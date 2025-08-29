# Voice Cloning Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/)

A deep learning project for voice cloning using state-of-the-art TTS models. This repository contains implementations for fine-tuning and inference with Sesame CSM (1B) and Orpheus (3B) models.

## 📋 Table of Contents
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## ✨ Features
- Voice cloning using Sesame CSM (1B) model
- Support for fine-tuning on custom voice datasets
- Easy-to-use Jupyter notebooks for training and inference
- Integration with Hugging Face datasets
- Audio preprocessing and post-processing utilities

## 🛠 Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (Tesla T4 or better recommended)
- Google Colab Pro (recommended for training)
- Hugging Face account (for accessing models and datasets)

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/voice-cloning.git
   cd voice-cloning
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   For Google Colab, the notebooks will install the required packages automatically.

## 🚀 Usage

### 1. Setting Up Google Colab
1. Open either notebook in Google Colab:
   - `nb/Sesame_CSM_(1B)_TTS.ipynb`
   - `nb/Orpheus_(3B)_TTS.ipynb` (requires high-end GPU)

2. Mount your Google Drive (for saving models and datasets):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

### 2. Training on Custom Voice
1. Prepare your dataset in the following structure:
   ```
   dataset/
   ├── audio_files/
   │   ├── sample1.wav
   │   └── sample2.wav
   └── metadata.csv  # Format: audio_file,text
   ```

2. Update the dataset path in the notebook and run all cells.

### 3. Inference
Use the provided inference cells to generate speech from text using your trained model.

## 📁 Project Structure
```
voice-cloning/
├── nb/                          # Collab/Jupyter notebooks
│   ├── Sesame_CSM_(1B)_TTS.ipynb
│   └── Orpheus_(3B)_TTS.ipynb
├── outputs/                     # Generated audio samples
│   └── Sesame_CSM_(1B)_TTS/
│       ├── original.wav
│       └── cloned.wav
├── metadata_cleaning.py
├── .gitignore
└── README.md
```

## 📊 Results
- Audio samples are saved in the `outputs/` directory
- The model achieves good voice similarity with as little as 30 minutes of training data
- For best results, use at least 3 hours of high-quality audio

## 🔧 Troubleshooting

### Common Issues
1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Try a smaller model

2. **Invalid Notebook Error**
   ```bash
   python scripts/metadata_cleaning.py
   ```

3. **GPU Compatibility**
   - The Orpheus (3B) model requires high-end GPUs (A100/H100)
   - For Tesla T4, use the Sesame CSM (1B) model

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- [Hugging Face](https://huggingface.co/) for the Transformers library
- [MrDragonFox/Elise](https://huggingface.co/datasets/MrDragonFox/Elise) dataset
- Google Colab for providing free GPU resources
