import torch
import torch.nn as nn
import torchvision
from torchvision import DataLoader
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import sys
import os
import numpy as np

NUM_WORKERS = 4
BATCH_SIZE = 4


# temporary
def transforms(image, mask):

    return np.array(image), np.array(mask)


class SegmentationDataset(Dataset):
    def __init__(self, image_root, mask_paths, transforms):
        self.image_root = image_root
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        label_path = self.mask_paths[idx]
        if "01_GT" in label_path:
            image_folder = "01"
        else:
            image_folder = "02"
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        label_filename = os.path.basename(label_path)
        base = label_filename.split(".")[0]
        frame_id = base.removeprefix("man_seg")

        image_path = os.path.join(self.image_root, image_folder, f"t{frame_id}.tif")

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        return x


def train_u_net(train_loader):
    net = Net()


if __name__ == "__main__":

    train_dataset = SegmentationDataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )  # pin_memory = True, not necessary unless training on Cuda GPU

    train_u_net(train_loader)
