import os.path as osp
import io
import ujson as json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from torchvision.ops import box_convert, box_iou
import matplotlib.pyplot as plt

# Initialise data paths
DATASET_PATH = '/home/jovyan/assignment_data_bdd/'

IMAGES_PATH = osp.join(DATASET_PATH, 'bdd100k_images_100k/bdd100k/images/100k')
LABELS_PATH = osp.join(DATASET_PATH, 'bdd100k_labels_release/bdd100k/labels')

TRAIN_IMAGES_PATH = osp.join(IMAGES_PATH, 'train')
VAL_IMAGES_PATH = osp.join(IMAGES_PATH, 'val')
TEST_IMAGES_PATH = osp.join(IMAGES_PATH, 'test')

TRAIN_LABELS_PATH = osp.join(LABELS_PATH, 'bdd100k_labels_images_train.json')
VAL_LABELS_PATH = osp.join(LABELS_PATH, 'bdd100k_labels_images_val.json')

# --------------------------------------
# Code for loading the data labels
# --------------------------------------
def load_labels():
    with io.open(TRAIN_LABELS_PATH, 'r+', encoding='utf-8') as fp:
        train_labels = json.load(fp)
    with io.open(VAL_LABELS_PATH, 'r+', encoding='utf-8') as fp:
        val_labels = json.load(fp)
    return train_labels, val_labels


def preprocess_image(img):
    img = np.asarray(img)
    img = img.astype(np.float64)
    img = img.transpose((2, 0, 1))
    img /= 255
    img[0, :, :] -= 102.9801 / 255
    img[1, :, :] -= 115.9465 / 255
    img[2, :, :] -= 122.7717 / 255

    img_tensor = torch.from_numpy(img).float()
    return img_tensor


def inverse_preprocess(img_tensor):
    img_arr = img_tensor.clone().numpy()

    # Reverse the mean subtraction and scaling operations
    img_arr[0, :, :] += 102.9801 / 255
    img_arr[1, :, :] += 115.9465 / 255
    img_arr[2, :, :] += 122.7717 / 255
    img_arr *= 255
    # Reshape the image back to (C, H, W) format
    img_arr = img_arr.transpose((1, 2, 0))
    img_arr = np.round(img_arr)
    # Convert the pixel values back to uint8
    img_arr = img_arr.astype(np.uint8)
    img = Image.fromarray(img_arr)
    return img


def trim_data(raw_data_list):
    parsed_data = []
    for raw_img_data in raw_data_list:
        entry = {'name': raw_img_data['name'], 'labels': []}
        boxes = []
        for label in raw_img_data['labels']:
            if 'box2d' in label:

                # Filter out small boxes
                width = label['box2d']['x2'] - label['box2d']['x1']
                height = label['box2d']['y2'] - label['box2d']['y1']
                if width >= 20 and height >= 20:
                    boxes.append(label)
        if len(boxes) == 0:
            continue
        entry['labels'] = boxes
        parsed_data.append(entry)
    return parsed_data



def get_nms_boxes(pred, iou_thresh=0.2, score_thresh=0.1, max_output_boxes=100):
    na, h, w, c = pred.shape

    pred_sample = pred.clone().view(na * h * w, c)
    boxes = box_convert(pred_sample[:, :4], in_fmt='cxcywh', out_fmt='xyxy')

    boxes[:, 0] = torch.round(torch.clamp(boxes[:, 0], min=0, max=640))  # x1
    boxes[:, 2] = torch.round(torch.clamp(boxes[:, 2], min=0, max=640))  # x2

    # Clamp y values to [0, 384] and round to the nearest integer
    boxes[:, 1] = torch.round(torch.clamp(boxes[:, 1], min=0, max=384))  # y1
    boxes[:, 3] = torch.round(torch.clamp(boxes[:, 3], min=0, max=384))  # y2

    objness_scores = pred_sample[:, 4]
    vals, indixes = torch.max(pred_sample[:, 5:], axis=1)
    conf_scores = objness_scores * vals

    conf_scores_reshaped = conf_scores.view(-1, 1)
    indixes_reshaped = indixes.view(-1, 1)
    boxes = torch.cat((boxes, conf_scores_reshaped, indixes_reshaped), dim=1)

    boxes = boxes[boxes[:, 4] >= score_thresh]
    if boxes.numel() != 0:
        sorted_indices = boxes[:, 4].argsort(descending=True)
        boxes = boxes[sorted_indices]

    keep_boxes = []
    while len(boxes) > 0:
        # Select the box with the highest score
        current_box = boxes[0]
        keep_boxes.append(current_box)

        # Remove the current box from the list
        boxes = boxes[1:]
        #     scores = scores[1:]

        if len(boxes) == 0:
            break

        # Calculate IoU with remaining boxes and remove redundant ones
        iou_values = [box_iou(current_box[:4].view(1, -1), box[:4].view(1, -1)) for box in boxes]
        non_redundant_indices = [idx for idx, iou_val in enumerate(iou_values) if
                                 iou_val.item() < iou_thresh]

        boxes = boxes[non_redundant_indices]

    return keep_boxes[:max_output_boxes]


def draw_bboxes(img, boxes, fmt='cxcywh'):
    """
    Assuming bboxes are in cxcywh format
    """
    # Create a copy of the image to draw bounding boxes on
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)

    # Load a font for category labels
    font = ImageFont.load_default()

    if fmt == 'cxcywh':
        h, w, c = boxes.shape
        boxes = boxes.clone().view(h * w, c)

        boxes = box_convert(boxes[:, :4], in_fmt='cxcywh', out_fmt='xyxy')

        boxes[:, 0] = torch.round(torch.clamp(boxes[:, 0], min=0, max=640))  # x1
        boxes[:, 2] = torch.round(torch.clamp(boxes[:, 2], min=0, max=640))  # x2

        # Clamp y values to [0, 384] and round to the nearest integer
        boxes[:, 1] = torch.round(torch.clamp(boxes[:, 1], min=0, max=384))  # y1
        boxes[:, 3] = torch.round(torch.clamp(boxes[:, 3], min=0, max=384))  # y2
    else:
        boxes = boxes[:, :4]

    # Convert the tensor to a NumPy array
    boxes_array = boxes.detach().numpy()

    for box in boxes_array:
        draw.rectangle(box, outline='green', width=1)

    aspect_ratio = img.size[0] / img.size[1]
    figsize = 12
    plt.figure(figsize=(figsize, figsize / aspect_ratio))  # Set the desired figure size here
    plt.imshow(draw_img)
    plt.axis('off')  # Turn off axis labels
    plt.show()
    return boxes_array
