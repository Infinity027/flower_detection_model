import argparse
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="flower.yaml", help="Path to data.yaml")
    parser.add_argument("--barplot", type=bool, default=False, help="draw barplot of data")
    parser.add_argument("--plot_image", type=bool, default=False, help="draw barplot of data")
    args = parser.parse_args()
    if not os.path.exists(args.data):
        print("Could not found yaml path!!! Give correct path adress...")
        exit(1)
    return args 

def data_count(img_dir:str,classes:dict):
    dic = {}
    for i in classes.values():
        dic[i] = 0
    for filename in os.listdir(img_dir):
        img_path = os.path.join(img_dir, filename)
        label_path = img_path.replace("images","labels").replace("jpg","txt")
        if not os.path.exists(img_path):
            print(img_path)
        with open(label_path, mode='r') as f:
            labels = [x.split() for x in f.read().splitlines()]
        for label in labels:
            dic[classes[int(label[0])]] += 1
    return dic

def bargraph(train_data, test_data):
    """
    draw bar graph of data present in each classes
    """
    class_names = list(train_data.keys())
    data = {    "train": [train_data[key] for key in train_data.keys()],
                "test": [test_data[key] for key in test_data.keys()]
            }
    x = np.arange(len(class_names))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects)
        multiplier += 1
    ax.set_ylabel('count')
    ax.set_title('Number of data present in each classes')
    ax.set_xticks(x + width, class_names)
    ax.legend(loc='upper left', ncols=2)

    ax.figure.set_size_inches(25, 10)
    fig.savefig("bargraph.png")

def plot_images(img_dir: str, classes: dict, k=4):
    filenames = rd.sample(os.listdir(img_dir), k)
    row = int(np.ceil(np.sqrt(k)))
    col = row if row * (row - 1) < k else row + 1

    fig, axes = plt.subplots(row, col, figsize=(col * 4, row * 4))

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, filename in enumerate(filenames):
        img_path = os.path.join(img_dir, filename)
        label_path = img_path.replace("images", "labels").replace("jpg", "txt")

        # Open image and resize to 224x224
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))
        draw = ImageDraw.Draw(img)

        # Load font (optional: needs a valid .ttf font file)
        try:
            font = ImageFont.truetype("arial.ttf", 10)  # Ensure arial.ttf is available
        except IOError:
            font = ImageFont.load_default()

        # Read label file
        with open(label_path, mode="r") as f:
            labels = [x.split() for x in f.read().splitlines()]

        # Draw bounding boxes with labels
        for label in labels:
            id, x_center, y_center, width, height = label
            id = int(id)
            x_center, y_center, width, height = map(float, [x_center, y_center, width, height])

            x1 = int((x_center - width / 2) * 224)
            y1 = int((y_center - height / 2) * 224)
            x2 = int((x_center + width / 2) * 224)
            y2 = int((y_center + height / 2) * 224)

            color = ((id*75)%255,(id*125)%255,(id*255)%255, 70)
            draw.rectangle([x1, y1, x2, y2], outline=color[:3], width=1)
            draw.text((x1, y1 - 12), f"{classes[id]}", fill=color, font=font)

        # Show image in subplot
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("plot_images.png")

if __name__=="__main__":
    args = parse_args()
    with open(args.data, mode="r") as f:
        data_item = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    val_path = data_item["VAL"]
    train_path = data_item["TRAIN"]
    classes = data_item["CLASS_INFO"]
    # print(classes,type(classes))
    val_instances = data_count(val_path, classes)
    train_instances = data_count(train_path, classes)
    print("Total number of Train instances:", sum(train_instances.values()))
    for key in train_instances.keys():
        print(f"{key}:{train_instances[key]}")
    print("Total number of validation instances:", sum(val_instances.values()))
    for key in val_instances.keys():
        print(f"{key}:{val_instances[key]}")
    if args.barplot:
        bargraph(train_instances,val_instances)
    if args.plot_image:
        plot_images(val_path,classes,k=4)


