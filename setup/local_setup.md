# Running the Labs Locally (Optional)

Google Colab is the recommended environment for all labs in this course because it provides a free GPU and requires no installation. However, if you prefer to work locally (or need to use the lab computers), follow the steps below.

## 1. Install Python and a virtual environment

We recommend Python **3.10 - 3.12**. Create an isolated environment so the course packages do not conflict with other projects.

### Windows (PowerShell)
```powershell
git clone https://github.com/hasanzaki/MCTA-4364-Machine-Vision.git
cd MCTA-4364-Machine-Vision
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r setup/requirements.txt
```

### macOS / Linux
```bash
git clone https://github.com/hasanzaki/MCTA-4364-Machine-Vision.git
cd MCTA-4364-Machine-Vision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r setup/requirements.txt
```

## 2. Select the interpreter in your notebook

Open any notebook in `notebooks/` and select the kernel named **.venv** (the environment you just created).

## 3. Important notes for local use

| Topic | Colab | Local |
|---|---|---|
| `cv2.imshow` (interactive windows) | Not supported | Supported |
| `cv2.VideoCapture(0)` (webcam) | Not supported | Supported |
| GPU training | Free T4 GPU | Needs a CUDA-capable GPU |
| Large datasets | Upload or download on demand | Stored in `projects/datasets/` |

When running in Colab, replace any `cv2.imshow` call with `matplotlib` display code. The notebooks already provide a `show()` helper that works in both environments.

## 4. Troubleshooting

- **`cv2` import error:** run `pip install opencv-python`.
- **`ultralytics` cannot find a model:** the first run downloads weights automatically. Ensure you have internet access.
- **Out-of-memory during training:** reduce the batch size in the notebook's configuration cell.
