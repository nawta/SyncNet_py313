# Installation Guide

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional, but recommended)
- FFmpeg

## Installation Steps

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/SyncNet_py313.git
cd SyncNet_py313
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

### 4. Download model weights

Place the following files in the `weights/` directory:
- `sfd_face.pth` - S3FD face detector weights
- `syncnet_v2.model` - SyncNet model weights

### 5. Verify installation

```bash
python scripts/run_example.py
```

## Troubleshooting

### CUDA not available

If you see CUDA-related errors, ensure:
1. NVIDIA drivers are installed
2. CUDA toolkit is installed
3. PyTorch is installed with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### FFmpeg not found

Install FFmpeg:
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from https://ffmpeg.org/download.html

### Missing dependencies

If you encounter import errors:
```bash
pip install --upgrade -r requirements.txt
```