""" Creates a custom dataset class for loading images and labels for a CustomYolov5 model."""
import os.path as osp
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, labels, images_path, transform=None):
        self.get_data_details()
        self.images_path = images_path
        self.labels = labels
        self.transform = transform

    def get_data_details(self):
        self.orig_size = (1280, 720)
        self.model_input_size = (640, 384)
        self.scale_x = self.model_input_size[0] / self.orig_size[0]
        self.scale_y = self.model_input_size[1] / self.orig_size[1]

        self.classes = ['car', 'traffic sign', 'traffic light', 'person', 'truck', 'bus',
                        'bike', 'rider', 'motor', 'train']
        self.nc = len(self.classes)
        self.class_indices = {c: i for i, c in enumerate(self.classes)}

        self.nx, self.ny = 40, 24  # grid design
        self.grid_cell_size = 16

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Implement data loading and preprocessing here

        labels = self.labels[idx]

        img_path = osp.join(self.images_path, labels['name'])
        img = Image.open(img_path)
        img = img.resize(self.model_input_size)
        if self.transform:
            img = self.transform(img)

        target_tensor = torch.zeros((self.ny, self.nx, 5 + self.nc))
        for obj in labels['labels']:
            if 'box2d' in obj:
                x1, y1, x2, y2 = obj['box2d']['x1'], obj['box2d']['y1'], obj['box2d']['x2'], \
                obj['box2d']['y2']
                x_center = ((x1 + x2) / 2) * self.scale_x
                y_center = ((y1 + y2) / 2) * self.scale_y
                width = (x2 - x1) * self.scale_x
                height = (y2 - y1) * self.scale_y

                category = obj['category']
                one_hot = np.eye(self.nc)[self.class_indices[category]]

                grid_x = int(x_center // self.grid_cell_size)
                grid_y = int(y_center // self.grid_cell_size)

                target_tensor[grid_y, grid_x, :5] = torch.tensor(
                    [x_center, y_center, width, height, 1])
                target_tensor[grid_y, grid_x, 5:] = torch.tensor(one_hot)

        sample = {'image': img, 'labels': target_tensor, 'img_path': img_path}

        return sample
