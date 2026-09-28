"""Shared helpers for the Machine Vision course notebooks.

Usage (after the setup cell has cloned the repository):

    import sys
    sys.path.append("resources/scripts")
    from cvhelpers import show, concept_map, image_grid, interactive
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

try:
    import ipywidgets as widgets
    from ipywidgets import interact
    HAS_WIDGETS = True
except Exception:
    HAS_WIDGETS = False


def show(*images, titles=None, cmap=None, figsize=None):
    """Display one or more images (BGR or gray) side by side."""
    titles = titles or [""] * len(images)
    figsize = figsize or (5 * len(images), 5)
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        if img is None:
            plt.text(0.5, 0.5, "None", ha="center", va="center")
        elif img.ndim == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap=cmap or "gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def concept_map(steps, title="Concept map", color="#e8f0fe", edge="#1a73e8"):
    """Draw a vertical flow diagram (infographic) from a list of steps."""
    n = len(steps)
    fig, ax = plt.subplots(figsize=(9, 0.95 * n + 1))
    ax.axis("off")
    ys = np.linspace(0.92, 0.08, n)
    for i, s in enumerate(steps):
        ax.text(0.5, ys[i], s, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.55", fc=color, ec=edge, lw=1.6),
                wrap=True)
        if i < n - 1:
            ax.annotate("", xy=(0.5, ys[i + 1] + 0.055), xytext=(0.5, ys[i] - 0.055),
                        arrowprops=dict(arrowstyle="-|>", color="#5f6368", lw=1.8))
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def image_grid(images, cols=3, titles=None, figsize=(15, 5)):
    """Display a grid of images (useful for many small results)."""
    n = len(images)
    rows = int(np.ceil(n / cols))
    titles = titles or [""] * n
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if img.ndim == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_histogram(image, title="Histogram"):
    """Plot the intensity histogram of a gray or colour image."""
    plt.figure(figsize=(6, 4))
    if image.ndim == 2:
        plt.plot(cv2.calcHist([image], [0], None, [256], [0, 256]), color="k")
    else:
        for i, c in enumerate(["b", "g", "r"]):
            plt.plot(cv2.calcHist([image], [i], None, [256], [0, 256]), color=c)
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel count")
    plt.xlim([0, 256])
    plt.show()


# ---------------------------------------------------------------------------
# Course-wide utilities (paths, downloads, diagrams, quizzes, self-checks)
# ---------------------------------------------------------------------------

import hashlib
import os
import time
import urllib.request
from pathlib import Path

# CI sets MV_SMOKE=1: notebooks shrink datasets and epochs so every cell still
# runs end to end on a CPU runner. Students never need to set it.
SMOKE = os.environ.get("MV_SMOKE", "0") == "1"


def repo_root():
    """Absolute path of the repository root."""
    return Path(__file__).resolve().parents[2]


def data_dir(*parts):
    """Folder for downloaded datasets (git-ignored), created on demand."""
    path = repo_root().joinpath("data", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url, dest, retries=3):
    """Download `url` to `dest` once; later calls reuse the cached file."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length", 0))
                done, next_report = 0, 0.1
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total and done / total >= next_report:
                        print(f"  {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
                        next_report += 0.1
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - report and retry any network error
            print(f"  attempt {attempt} failed: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {url}. Check your internet connection.")


def pipeline_diagram(stages, title=None, colors=None, notes=None, ax=None,
                     figsize=None, fontsize=10):
    """Draw a left-to-right block diagram, e.g. ["image", "filter", "feature"].

    `notes` is an optional list of small captions drawn under each block.
    """
    n = len(stages)
    colors = colors or ["#e8f0fe"] * n
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=figsize or (2.3 * n, 2.0))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, s in enumerate(stages):
        ax.text(i + 0.5, 0.6, s, ha="center", va="center", fontsize=fontsize,
                bbox=dict(boxstyle="round,pad=0.5", fc=colors[i], ec="#5f6368", lw=1.3))
        if notes and notes[i]:
            ax.text(i + 0.5, 0.18, notes[i], ha="center", va="center",
                    fontsize=fontsize - 2, color="#5f6368", style="italic")
        if i < n - 1:
            ax.annotate("", xy=(i + 1.12, 0.6), xytext=(i + 0.88, 0.6),
                        arrowprops=dict(arrowstyle="-|>", color="#5f6368", lw=1.6))
    if title:
        ax.set_title(title, fontsize=fontsize + 2, fontweight="bold")
    if own_fig:
        plt.tight_layout()
        plt.show()
    return ax


def youtube(video_id, width=720, height=405):
    """Embed a YouTube video (shows a link instead when offline or static)."""
    from IPython.display import YouTubeVideo, display
    print(f"Video: https://www.youtube.com/watch?v={video_id}")
    display(YouTubeVideo(video_id, width=width, height=height))


def _answer_key(question, option):
    return hashlib.sha256(f"{question}|{option}".encode()).hexdigest()[:10]


def quiz(questions, title="Quick check"):
    """Auto-marked multiple-choice quiz.

    Each question is a dict with keys `q`, `options`, `answer` (the answer key
    of the correct option, so the answer is not readable in the notebook) and
    `explain` (shown after the student checks their choice).
    """
    if not HAS_WIDGETS or SMOKE:  # static text version (no front end in CI)
        print(title)
        for i, item in enumerate(questions, 1):
            print(f"\nQ{i}. {item['q']}")
            for opt in item["options"]:
                print(f"   - {opt}")
        return
    from IPython.display import display
    score = {}
    total = widgets.HTML()

    def refresh_total():
        total.value = (f"<b>Score: {sum(score.values())} / {len(questions)}</b>"
                       f" ({len(score)} answered)")

    rows = [widgets.HTML(f"<h3>{title}</h3>")]
    for i, item in enumerate(questions):
        choice = widgets.RadioButtons(options=item["options"], value=None,
                                      layout=widgets.Layout(width="max-content"))
        button = widgets.Button(description="Check", button_style="primary")
        feedback = widgets.HTML()

        def on_click(_, i=i, item=item, choice=choice, feedback=feedback):
            if choice.value is None:
                feedback.value = "<i>Pick an option first.</i>"
                return
            ok = _answer_key(item["q"], choice.value) == item["answer"]
            score[i] = int(ok)
            mark = ("<span style='color:#188038'><b>Correct.</b></span>" if ok
                    else "<span style='color:#d93025'><b>Not quite.</b></span>")
            feedback.value = f"{mark} {item.get('explain', '') if ok else 'Try again.'}"
            refresh_total()

        button.on_click(on_click)
        rows += [widgets.HTML(f"<b>Q{i + 1}. {item['q']}</b>"), choice,
                 widgets.HBox([button, feedback])]
    refresh_total()
    rows.append(total)
    display(widgets.VBox(rows))


def check(name, test):
    """Run a self-check for an exercise and print a friendly verdict.

    `test` is a function that raises AssertionError with a hint when the
    student's answer is wrong. An exercise still left as `raise
    NotImplementedError` is reported as not attempted instead of failing.
    """
    try:
        test()
        print(f"✅ {name}: passed")
        return True
    except NotImplementedError:
        print(f"⏳ {name}: not attempted yet - complete the TODO above and re-run.")
    except AssertionError as exc:
        print(f"❌ {name}: {exc}")
    except Exception as exc:  # noqa: BLE001 - show any error as feedback
        print(f"❌ {name}: your code raised {type(exc).__name__}: {exc}")
    return False


def interact(func, **kwargs):
    """Drop-in for `ipywidgets.interact` that also works without a live front end.

    With widgets available, this is exactly `ipywidgets.interact`. In SMOKE/CI
    mode (headless runners, where widget messages can stall execution) or
    without ipywidgets, `func` is called once with the default values, so the
    cell still produces a static figure.
    """
    if HAS_WIDGETS and not SMOKE:
        return widgets.interact(func, **kwargs)
    defaults = {}
    for name, spec in kwargs.items():
        if hasattr(spec, "value"):
            defaults[name] = spec.value
        elif isinstance(spec, (list, tuple)) and spec:
            defaults[name] = spec[0]
        else:
            defaults[name] = spec
    return func(**defaults)
