from utils import *

import random

from data import CustomDataset
from model import CustomYOLOv5

import torchvision.transforms as transforms

# --------------------------------
# Load labels
# --------------------------------
train_labels, val_labels = load_labels()


# --------------------------------
# Setup Dataset classes
# --------------------------------
preprocess_transform = transforms.Lambda(preprocess_image)
# Val Dataset class
val_labels_trimmed = trim_data(val_labels)
n = min(50, len(val_labels_trimmed))
val_labels_trimmed = random.sample(val_labels_trimmed, n)
val_dataset = CustomDataset(labels=val_labels_trimmed,
                            images_path=VAL_IMAGES_PATH,
                            transform=transforms.Compose([
                                preprocess_transform
                            ]))

# --------------------------------
# Setup DataLoaders
# --------------------------------
batch_size = 32
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# --------------------------------
# Define model
# --------------------------------
num_classes = 10
num_anchors = 1
anchors = [[8,8]]
model = CustomYOLOv5(num_classes, num_anchors, anchors)
checkpoint_path = './object_detection_model.pth'
model.load_state_dict(torch.load(checkpoint_path))

# Set the model to evaluation mode
model.eval()

for batch_idx, batch_data in enumerate(val_loader):
    images = batch_data['image']
    target = batch_data['labels']
    pred = model(images)

    nms_pred = []
    for pred_tensor in pred:
        nms_boxes = get_nms_boxes(pred)
        nms_pred.append(nms_boxes)

