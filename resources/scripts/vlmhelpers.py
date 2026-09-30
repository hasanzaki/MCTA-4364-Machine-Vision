"""
Vision-language helpers for Weeks 13A and 13B.

`load_clip()` returns a CLIP-style model with two methods,

    encode_images(list_of_bgr_images) -> (N, D) float32 array, unit length rows
    encode_text(list_of_strings)      -> (M, D) float32 array, unit length rows

so that cosine similarity is just `img_emb @ txt_emb.T`.

Backends
--------
"mobileclip" (default)  Apple MobileCLIP-B (LT), 86 M + 63 M parameters. The weights
                        are the TorchScript file that Ultralytics publishes on GitHub
                        for YOLOE (the same text encoder YOLOE uses), so it downloads
                        even where Hugging Face is blocked. The TorchScript file only
                        exposes the text tower; the image tower (a ViT-B/16 with a
                        convolutional stem) is re-implemented below and its weights are
                        read from the same file.
"openai"                OpenAI CLIP ViT-B/32 through Hugging Face `transformers`.

Licences: MobileCLIP weights - Apple (see github.com/apple/ml-mobileclip, research
and educational use); OpenAI CLIP - MIT.
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from cvhelpers import data_dir, download

MOBILECLIP_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/mobileclip_blt.ts"


def _tokenizer():
    """CLIP byte-pair tokenizer (77 tokens), from open_clip or the CLIP package."""
    try:
        import open_clip
        return lambda texts: open_clip.tokenize(texts)
    except ImportError:
        pass
    try:
        import clip
        return lambda texts: clip.tokenize(texts, truncate=True)
    except ImportError:
        pass
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
    import open_clip
    return lambda texts: open_clip.tokenize(texts)


def _to_list(images):
    if isinstance(images, np.ndarray) and images.ndim in (2, 3):
        return [images]
    return list(images)


def _prep(img, size=224):
    """BGR/grey uint8 -> RGB, shortest side resized to `size`, centre crop."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = size / min(h, w)
    img = cv2.resize(img, (max(size, round(w * s)), max(size, round(h * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    return img[y:y + size, x:x + size, ::-1]


class MobileCLIP:
    name = "MobileCLIP-B (LT)"

    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = download(MOBILECLIP_URL, data_dir("weights") / "mobileclip_blt.ts")
        ts = torch.jit.load(str(path), map_location=self.device).eval()
        self._text = ts.text_encoder
        self._w = {k[len("model."):]: v.detach() for k, v in ts.image_encoder.state_dict().items()}
        self.logit_scale = float(ts.logit_scale.detach().exp())
        self.dim = self._w["classifier.proj"].shape[1]
        self._tok = _tokenizer()

    def _bn(self, x, p):
        w = self._w
        return F.batch_norm(x, w[p + ".running_mean"], w[p + ".running_var"], w[p + ".weight"], w[p + ".bias"], False, 0.0, 1e-5)

    def _image_tower(self, x):
        w = self._w
        # convolutional stem: 4x4/4 -> 2x2/2 -> 2x2/2 gives 14 x 14 patches of 768 channels
        x = F.gelu(self._bn(F.conv2d(x, w["patch_emb.0.block.conv.weight"], stride=4), "patch_emb.0.block.norm"))
        x = F.gelu(self._bn(F.conv2d(x, w["patch_emb.1.block.conv.weight"], stride=2), "patch_emb.1.block.norm"))
        x = F.conv2d(x, w["patch_emb.2.block.conv.weight"], w["patch_emb.2.block.conv.bias"], stride=2)
        b = x.shape[0]
        x = x.flatten(2).transpose(1, 2) + w["pos_embed.pos_embed.pos_embed"][0]
        x = torch.cat([w["cls_token"].expand(b, -1, -1), x], 1)
        d = x.shape[-1]
        for i in range(sum(k.endswith("pre_norm_mha.0.weight") for k in w)):
            p = f"transformer.{i}."
            h = F.layer_norm(x, (d,), w[p + "pre_norm_mha.0.weight"], w[p + "pre_norm_mha.0.bias"])
            qkv = F.linear(h, w[p + "pre_norm_mha.1.qkv_proj.weight"], w[p + "pre_norm_mha.1.qkv_proj.bias"])
            qkv = qkv.view(b, -1, 3, 12, d // 12).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(b, -1, d)
            x = x + F.linear(a, w[p + "pre_norm_mha.1.out_proj.weight"], w[p + "pre_norm_mha.1.out_proj.bias"])
            h = F.layer_norm(x, (d,), w[p + "pre_norm_ffn.0.weight"], w[p + "pre_norm_ffn.0.bias"])
            h = F.gelu(F.linear(h, w[p + "pre_norm_ffn.1.weight"], w[p + "pre_norm_ffn.1.bias"]))
            x = x + F.linear(h, w[p + "pre_norm_ffn.4.weight"], w[p + "pre_norm_ffn.4.bias"])
        x = F.layer_norm(x, (d,), w["post_transformer_norm.weight"], w["post_transformer_norm.bias"])
        return x[:, 0] @ w["classifier.proj"]                    # class token -> shared 512-D space

    @torch.no_grad()
    def encode_images(self, images, batch_size=32):
        out = []
        imgs = _to_list(images)
        for i in range(0, len(imgs), batch_size):
            x = np.stack([_prep(im) for im in imgs[i:i + batch_size]]).astype(np.float32) / 255.0   # MobileCLIP: no mean/std
            f = self._image_tower(torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device))
            out.append(F.normalize(f, dim=-1).float().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def encode_text(self, texts, batch_size=256):
        texts = [texts] if isinstance(texts, str) else list(texts)
        out = []
        for i in range(0, len(texts), batch_size):
            f = self._text(self._tok(texts[i:i + batch_size]).to(self.device))
            out.append(F.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)


class HFCLIP:
    """OpenAI CLIP (or any CLIP checkpoint) through Hugging Face transformers."""

    def __init__(self, model_id="openai/clip-vit-base-patch32", device=None):
        from transformers import CLIPModel, CLIPProcessor
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self.name = model_id
        self.logit_scale = float(self.model.logit_scale.detach().exp())
        self.dim = self.model.config.projection_dim

    @torch.no_grad()
    def encode_images(self, images, batch_size=32):
        out = []
        imgs = _to_list(images)
        for i in range(0, len(imgs), batch_size):
            rgb = [cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) if im.ndim == 2 else im[..., ::-1].copy() for im in imgs[i:i + batch_size]]
            px = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(self.device)
            f = self.model.visual_projection(self.model.vision_model(pixel_values=px).pooler_output)
            out.append(F.normalize(f, dim=-1).float().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def encode_text(self, texts, batch_size=256):
        texts = [texts] if isinstance(texts, str) else list(texts)
        out = []
        for i in range(0, len(texts), batch_size):
            tok = self.proc(text=texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True).to(self.device)
            f = self.model.text_projection(self.model.text_model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).pooler_output)
            out.append(F.normalize(f, dim=-1).float().cpu().numpy())
        return np.concatenate(out)


def load_clip(backend="mobileclip", device=None):
    """Load a CLIP model: backend "mobileclip" (GitHub download) or "openai" (Hugging Face)."""
    if backend == "mobileclip":
        return MobileCLIP(device)
    if backend == "openai":
        return HFCLIP("openai/clip-vit-base-patch32", device)
    return HFCLIP(backend, device)


@torch.no_grad()
def yoloe_set_text_classes(yoloe, names, clip_model):
    """Give a YOLOE detector text prompts, encoded with our MobileCLIP.

    YOLOE was trained with exactly this MobileCLIP text encoder, so the result is identical to
    `yoloe.set_classes(names, yoloe.get_text_pe(names))`, without the extra package that
    Ultralytics installs from git for its own tokenizer.
    """
    names = list(names)
    p = next(yoloe.model.parameters())
    feats = torch.from_numpy(clip_model.encode_text(names))[None].to(p.device, p.dtype)
    yoloe.set_classes(names, yoloe.model.model[-1].get_tpe(feats))
    return yoloe
