# torch-config.py
import os
import torch

MODELS_DIR = "pretrained_models"

os.makedirs(MODELS_DIR, exist_ok=True)

os.environ['TORCH_HOME'] = MODELS_DIR
torch.hub.set_dir(MODELS_DIR)
