import pandas as pd
import numpy as np
import json as json
import os.path as osp
import io
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
from IPython.display import HTML

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

# ------------------------------------------
# Code for parsing the data into a dataframe
# ------------------------------------------
def parse_data(raw_data_list):
    """ Parse the raw data and extract objects with bounding box labels into a dataframe"""
    parsed_data = []
    for raw_img_data in raw_data_list:
        for label in raw_img_data['labels']:
            if 'box2d' in label:
                img_data = {}
                img_data['name'] = raw_img_data['name']
                img_data['category'] = label['category']
                for k, v in label['attributes'].items():
                    img_data[k] = v
                img_data['box2d'] = label['box2d']
                img_data['id'] = label['id']
                img_data['width'] = label['box2d']['x2'] - label['box2d']['x1']
                img_data['height'] = label['box2d']['y2'] - label['box2d']['y1']
                parsed_data.append(img_data)
    df = pd.DataFrame(parsed_data)
    return df


# ------------------------------------------
# Utility function for data analysis
# ------------------------------------------
def get_object_categories(data_df):
    """ Get only object categories"""
    category_counts = data_df['category'].value_counts()
    category_percentages = data_df['category'].value_counts(normalize=True) * 100
    categories_df = pd.DataFrame({'count': category_counts, 'percentage': category_percentages})
    categories_df.sort_values('count', ascending=False, inplace=True)
    return categories_df

def get_mean_object_size(df):
    """ Get mean object size"""
    print("Mean bbox HxW: (%s,%s)" %(df.width.mean(), df.height.mean()))



# ----------------------------------------------------------
# Utility function for pretty display of multiple dataframes
# ----------------------------------------------------------
def horizontal(dfs):
    html = '<div style="display:flex">'
    for df in dfs:
        html += '<div style="margin-right: 32px">'
        html += df.to_html()
        html += '</div>'
    html += '</div>'
    return display(HTML(html))


# -----------------------------------------------------------
# Utility function for visualizing bounding boxes on images
# -----------------------------------------------------------
category_colors = {
    'bike': 'orchid',
    'bus': 'darkorange',
    'car': 'red',
    'motor': 'blue',
    'person': 'purple',
    'rider': 'tan',
    'traffic light': 'green',
    'traffic sign': 'magenta',
    'train': 'darkgrey',
    'truck': 'brown'
}

def show_annotated_image(img_name, labels_df, split):
    if split == 'val':
        images_path = VAL_IMAGES_PATH
    else:
        images_path = TRAIN_IMAGES_PATH

    img_path = osp.join(images_path, img_name)
    img = Image.open(img_path)
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)

    # Load a font for category labels
    font = ImageFont.load_default()

    for index, row in labels_df.iterrows():
        if 'box2d' in row:
            color = category_colors.get(row['category'], 'red')
            x1, y1 = row['box2d']['x1'], row['box2d']['y1']
            x2, y2 = row['box2d']['x2'], row['box2d']['y2']
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Add category label
            label = row['category']

            label_width = draw.textlength(label, font=font)
            label_height = draw.textbbox((0, 0), label, font=font)[3]
            draw.rectangle([x1, y1 - label_height, x1 + label_width, y1], fill=color)
            draw.text((x1, y1 - label_height), label, fill='white', font=font)

    aspect_ratio = img.size[0] / img.size[1]
    figsize = 12
    plt.figure(figsize=(figsize, figsize / aspect_ratio))  # Set the desired figure size here
    plt.imshow(draw_img)
    plt.axis('off')  # Turn off axis labels
    plt.show()


def show_annotated_image_randomly(df, split, category=None, n=None, h=None, w=None):
    """
    Show a random annotated image from the dataframe.
    If category is provided, filter by that object class and ensure at least n instances.
    Also filter by height (h) and width (w) if provided.
    of that class in the image.
    """
    df_copy = df.copy()
    if category is not None:
        assert n is not None, "Please provide a value for n"
        df_copy = df_copy.query("category == @category")
        df_copy = df_copy.groupby('name').filter(lambda x: len(x) >= n)

    if h is not None:
        df_copy = df_copy.groupby('height').filter(lambda x: len(x) >= h)
        if n is not None:
            df_copy = df_copy.groupby('name').filter(lambda x: len(x) >= n)

    if w is not None:
        df_copy = df_copy.groupby('width').filter(lambda x: len(x) >= w)
        if n is not None:
            df_copy = df_copy.groupby('name').filter(lambda x: len(x) >= n)

    if len(df_copy)==0:
        print("No images found with the given conditions")
        return None

    img_name = random.choice(df_copy.name.unique().tolist())
    img_df = df.query("name == @img_name")
    show_annotated_image(img_name, img_df, split)
    return img_df


def show_annotated_image_by_name(img_name, df, split):
    """Show annotated image by name"""
    img_df = df.query("name == @img_name")
    if len(img_df) == 0:
        print(f"No image found with name {img_name}")
        return
    show_annotated_image(img_name, img_df, split)
    return img_df

def show_crops(df, split, category, n, context=1, crop_size=64):
    """
    Show n crops of the given category with context around the bounding box.
    Resize the crops to crop_size x crop_size for display.
    """
    if split == 'val':
        images_path = VAL_IMAGES_PATH
    else:
        images_path = TRAIN_IMAGES_PATH

    category_images = df.query("category == @category").sample(n)

    fig, axes = plt.subplots(int(np.ceil(n / 5)), 5, figsize=(20, 16))
    for idx, (_, row) in enumerate(category_images.iterrows()):
        img_path = osp.join(images_path, row['name'])
        img = Image.open(img_path)

        x1, y1, x2, y2 = (row['box2d']['x1'], row['box2d']['y1'],
                          row['box2d']['x2'], row['box2d']['y2'])
        width = x2 - x1
        height = y2 - y1

        # Calculate the cropped region with context
        x1_crop = max(0, int(x1 - 0.5 * context * width))
        y1_crop = max(0, int(y1 - 0.5 * context * height))
        x2_crop = min(img.width, int(x2 + 0.5 * context * width))
        y2_crop = min(img.height, int(y2 + 0.5 * context * height))

        # Extract the cropped region
        cropped_img = img.crop((x1_crop, y1_crop, x2_crop, y2_crop))

        # Draw the actual bounding box on the cropped region
        draw = ImageDraw.Draw(cropped_img)
        draw.rectangle([(x1 - x1_crop, y1 - y1_crop), (x2 - x1_crop, y2 - y1_crop)],
                       outline='red',
                       width=2)

        # Resize the cropped region to 64x64
        resized_img = cropped_img.resize((crop_size, crop_size))

        # Display the resized image
        ax = axes[idx // 5, idx % 5]
        ax.imshow(resized_img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
