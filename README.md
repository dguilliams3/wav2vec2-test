# AI Audio Project

This project fine-tunes Wav2Vec2 for Automatic Speech Recognition (ASR) using Mozilla's Common Voice dataset.  
It includes dataset preprocessing, model fine-tuning, and inference optimization.

---

## **📌 Project Setup**

### **1️⃣ Prerequisites**
- **Windows with WSL2**
- **Python 3.10+**
- **PyTorch with CUDA support**
- **Hugging Face Transformers and Datasets**
- **ONNX for model optimization**

### **2️⃣ Environment Setup**
Inside **WSL2**, run:
```bash
# Update and install system dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip git wget -y

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install torch torchvision torchaudio transformers datasets accelerate onnx onnxruntime torch-tensorrt

# Example script to test if CUDA is available
``` import torch
print(torch.cuda.is_available())  # Expected output: True

# Project Structure
ai-audio-project/
│── common_voice_dataset/   # Dataset storage (linked to OneDrive)
│── models/                 # Saved trained models
│── logs/                   # Training logs
│── scripts/                # Utility scripts
│── test_pipeline_from_dataset_to_finetuned_wac2vec.py  # Main training script
│── README.md               # This file
