from pathlib import Path

import torch

# paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PHC_IMAGE_ROOT = PROJECT_ROOT / "PhC-C2DH-U373"
DSB2018_TRAIN_ROOT = PROJECT_ROOT / "data-science-bowl-2018" / "stage1_train"
PLOT_DIR = PROJECT_ROOT / "plots"
MODEL_SAVE_PATH = PROJECT_ROOT / "checkpoints"
WEIGHT_MAP_CACHE_DIR = PROJECT_ROOT / "weight_cache"

# dataset selection
# options: "phc-u373" (default), "data-science-bowl-2018"
DATASET_CHOICE = "phc-u373"

# run save tag
RUN_DETAIL = "original_implementation"

# data loading
NUM_WORKERS = 12
BATCH_SIZE = 1  # batch size detailed in the paper
CROP_SIZE = 572

# augmentation
FLIP_PROBABILITY = 0.5
DEFORM_SIGMA = 10.0
DEFORM_POINTS = 3

# weight map
WEIGHT_MAP_W0 = 10.0
WEIGHT_MAP_SIGMA = 5.0
WEIGHT_MAP_CACHE_EXTENSION = ".npy"

# model/training (detailed in paper)
NUM_OUTPUT_CHANNELS = 2
LEARNING_RATE = 1e-4
EPOCHS = 170

# cuda configs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
