import cv2
import torch
import torchvision
from PIL import Image

# Load pretrained model & its canonical transforms
# weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
# model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
preprocess = weights.transforms()
coco_labels = weights.meta["categories"]

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


cap = cv2.VideoCapture(0)  # change index if needed
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

CONF_THRESHOLD = 0.5

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

    # BGR -> RGB -> PIL for transforms
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    # To (1, C, H, W) on the same device as model
    input_tensor = preprocess(frame_pil)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_batch)

    output = outputs[0]  # first batch element

    boxes  = output["boxes"].cpu().numpy().astype(int)
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    # Draw results
    for box, cls_id, score in zip(boxes, labels, scores):
        if score < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame_bgr, f"{coco_labels[cls_id]}: {score:.2f}",
                    (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("faster rcnn", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
