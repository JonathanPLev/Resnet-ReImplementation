from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

IMAGE_ROOT = PROJECT_ROOT / "PhC-C2DH-U373"
NUM_WORKERS = 12
BATCH_SIZE = 1  # batch size detailed in the paper
FLIP_PROBABILITY = 0.5
LEARNING_RATE = 0.01
MOMENTUM_TERM = 0.99  # detailed in paper
NUM_OUTPUT_CHANNELS = 2  # detailed in paper
EPOCHS = 400
WEIGHT_MAP_CACHE_DIR = PROJECT_ROOT / "weight_cache"
WEIGHT_MAP_W0 = 10.0
WEIGHT_MAP_SIGMA = 5.0
CROP_SIZE = 572
DEFORM_SIGMA = 10.0
DEFORM_POINTS = 3
WEIGHT_MAP_CACHE_EXTENSION = ".npy"
PLOT_DIR = PROJECT_ROOT / "plots"
RUN_DETAIL = "original_implementation"
MODEL_SAVE_PATH = PROJECT_ROOT / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
