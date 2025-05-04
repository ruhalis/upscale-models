# Upscale Models

A collection of models and utilities for image upscaling and super-resolution.

## Setup

### Environment Setup

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

Install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## RealESRGAN Integration

To use RealESRGAN models, you need to clone the Real-ESRGAN repository into this folder:

```bash
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install -r requirements.txt
python setup.py develop
cd ..
```

## Usage

Refer to individual model documentation for specific usage instructions.

## License

This project is distributed under the terms of the license specified in the LICENSE file.
