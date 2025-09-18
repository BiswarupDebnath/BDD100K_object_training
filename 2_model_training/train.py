from utils import *

import random

from data import CustomDataset
from model import CustomYOLOv5
from loss import CustomLoss

import torchvision.transforms as transforms

# --------------------------------
# Load labels
# --------------------------------
train_labels, val_labels = load_labels()

# ----------------------------------
# Setup data augmentation techniques
# ----------------------------------
brightness_jitter = 0.3
contrast_jitter = 0.3
saturation_jitter = 0.3
hue_jitter = 0.0

color_jitter = transforms.ColorJitter(brightness=brightness_jitter, contrast=contrast_jitter,
                                      saturation=saturation_jitter, hue=hue_jitter)
preprocess_transform = transforms.Lambda(preprocess_image)


# --------------------------------
# Setup Dataset classes
# --------------------------------

# Train Dataset class
train_labels_trimmed = trim_data(train_labels)
n = min(10000, len(train_labels_trimmed))
train_labels_trimmed = random.sample(train_labels_trimmed, n)
train_dataset = CustomDataset(labels=train_labels_trimmed,
                              images_path=TRAIN_IMAGES_PATH,
                              transform=transforms.Compose([
                                  color_jitter,
                                  preprocess_transform
                              ]))


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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# --------------------------------
# Define model
# --------------------------------
num_classes = 10
num_anchors = 1
anchors = [[8,8]]
model = CustomYOLOv5(num_classes, num_anchors, anchors)

# --------------------------------
# Define loss_function, optimiser, LR scheduler
# --------------------------------
loss_fn = CustomLoss(num_anchors)
param_list = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = torch.optim.Adam(param_list, lr=0.001, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# --------------------------------
# Train the model
# --------------------------------
losses_train = []
losses_eval = []
num_epochs = 2
for epoch in range(num_epochs):

    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        images = batch_data['image']
        labels = batch_data['labels']
        y = model(images)
        train_loss = loss_fn(y, labels)
        losses_train.append(train_loss)
        if batch_idx % 4 == 0:
            print("Train loss: %s" % train_loss)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    lr_scheduler.step()

    model.eval()
    for batch_idx, batch_data in enumerate(val_loader):
        images = batch_data['image']
        labels = batch_data['labels']
        y = model(images)
        val_loss = loss_fn(y, labels)
        losses_eval.append(val_loss)
        if batch_idx % 4 == 0:
            print("Val loss: %s" % val_loss)

# Save the trained model
checkpoint_path = './object_detection_model.pth'
torch.save(model.state_dict(), checkpoint_path)