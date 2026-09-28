# Running the notebooks on your laptop

The notebooks run in **Google Colab** or **Kaggle** with no setup. To run them on your own laptop, which is faster
for deep-learning weeks and gives you a live webcam, follow these steps once.

## Minimum hardware
| | Minimum | Recommended |
|---|---|---|
| GPU | none (CPU works, with smaller training runs) | NVIDIA GPU with **4 GB VRAM** or more (e.g. RTX 3050/4050 laptop) |
| RAM | 8 GB | 16 GB |
| Disk | 10 GB free | 25 GB free (datasets and model weights are cached in `data/`) |

Every notebook detects your hardware and adjusts the batch size, image size and number of epochs automatically.

## 1. Install Python and get the course files
1. Install [Miniforge](https://github.com/conda-forge/miniforge) (or Anaconda), which works on Windows, macOS and Linux.
2. Clone the repository (or download it as a ZIP from GitHub):
   ```bash
   git clone https://github.com/hasanzaki/MCTA-4364-Machine-Vision.git
   cd MCTA-4364-Machine-Vision
   ```
3. Create an environment:
   ```bash
   conda create -n mv python=3.11 -y
   conda activate mv
   ```

## 2. Install PyTorch with GPU support (do this first)
Go to **[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)**, select *Stable → your OS → Pip → Python → CUDA*
(choose the newest CUDA version offered), and run the command it shows. It looks like:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
```
No NVIDIA GPU? Choose *CPU* instead.

> Keep your **NVIDIA driver** up to date (GeForce Experience / NVIDIA App on Windows). The CUDA libraries are bundled
> with PyTorch; you do **not** need to install the CUDA Toolkit separately.

## 3. Install the remaining packages
```bash
pip install -r requirements.txt
```

## 4. Check your setup
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
`True` and your GPU name means you are ready.

## 5. Open the notebooks
```bash
jupyter lab
```
Or open the folder in **VS Code** with the *Python* and *Jupyter* extensions and choose the `mv` kernel.

## Troubleshooting
| Problem | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` | You installed the CPU build. Reinstall with the CUDA command from step 2, and update your NVIDIA driver. |
| `CUDA out of memory` | Halve the batch size in the settings cell, or restart the kernel to free GPU memory held by earlier cells. |
| Sliders/widgets do not appear | `pip install ipywidgets`, then restart Jupyter/VS Code. |
| Downloads fail | Check your internet/proxy. Datasets and weights are cached after the first successful download. |
| Webcam not found | Close other apps using the camera; try `camera=1` in the capture helper. |
