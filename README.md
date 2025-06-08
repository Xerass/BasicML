# basicML

A student's projects around ML

## Libraries & Tools Used

- **Jupyter Notebook (`.ipynb`)** – Interactive coding and documentation.
- **Matplotlib** – Basic data visualization.
- **Seaborn** – Statistical data visualization.
- **PyTorch (with CUDA)** – Deep learning framework accelerated by NVIDIA GPUs.

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/basicML.git
cd basicML
```

### 2. Set Up a Python Environment (Recommended)

Using `venv` or `conda` is recommended to avoid conflicts:

**Using `venv`:**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

#### Option A: With pip (GPU support)

To install GPU-accelerated PyTorch with other libraries:

```bash
# First, install PyTorch with CUDA (for NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then, install the rest
pip install matplotlib seaborn notebook
```

> ⚠️ Replace `cu121` with your CUDA version (e.g., `cu118`, `cu117`) if needed. See: https://pytorch.org/get-started/locally/

#### Option B: With conda (Alternative)

```bash
conda create -n basicml python=3.10
conda activate basicml

# Install GPU-enabled PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Then install others
conda install matplotlib seaborn notebook
```

## Notes

- Make sure your system has **NVIDIA drivers and CUDA toolkit** installed for GPU support.
- For CPU-only PyTorch, use `pip install torch torchvision torchaudio` without the `--index-url` flag.

---


