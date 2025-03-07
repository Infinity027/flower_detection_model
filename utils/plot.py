import numpy as np
import matplotlib.pyplot as plt
import json
# from PIL import ImageDraw, ImageFont
from matplotlib.patches import Rectangle

def plot_image_with_label(images, labels, class_names, format = "xcycwh", save_name="results/plot_image_with_label.png"):
    """
    draw image and label using plot function
    """
    colors = ['red', 'blue', 'green','yellow','purple','white']
    class_names[-1] = "None"
    #only first 25 images draw
    if len(images)>16:
        images = images[:16]
        labels = labels[:16]

    col = int(np.ceil(np.sqrt(len(images))))
    row = (col-1) if col * (col - 1) >= len(images) else col
    fig, axes = plt.subplots(row, col, figsize=(col * 5, row * 5))

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    im_size = images[0].shape[1]
    for image, boxes, ax in zip(images, labels, axes):
        if image.shape[0] == 3:
            ax.imshow(image.permute(1,2,0))
        else:
            ax.imshow(image)
        for box in boxes:
            # print(box)
            if format == 'xyxy':
                # convert xyxy to xcycwh format
                box[-2] = max(box[-2] - box[-4], 0)
                box[-1] = max(box[-1] - box[-3], 0)
                box[-4] = min(box[-4] + box[-2]/2, 1)
                box[-3] = min(box[-3] + box[-1]/2, 1)

            if len(box)>5:
                cls_id, conf, xc, yc, w, h = box
                names = f"{class_names[int(cls_id)]}: {(conf*100).item():.2f}"
            else:
                cls_id, xc, yc, w, h = box
                names = f"{class_names[int(cls_id)]}"

            text_size = min(max(int(w*h*120),8),12)
            xc, yc, w, h = xc*im_size, yc*im_size, w*im_size, h*im_size
            rect = Rectangle((xc-w/2, yc-h/2), w, h, edgecolor=colors[int(cls_id)], facecolor='none')
            # print(text_size)
            ax.add_patch(rect)
            ax.text(xc-w/2, yc-h/2, names, fontsize=text_size, color=colors[int(cls_id)])
        ax.axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_name)
    plt.close()

def plot_loss(filename, save_name="results/loss.png"):
    """
    draw losses and accuarcy
    """
    with open(filename, "r") as f:
        result = json.load(f)
    if len(result.keys())>6:
        plt.figure(figsize=(14,12))
        row = 3
        col = 3
    else:
        plt.figure(figsize=(14,8))
        row = 2
        col = 3 

    for i, key in enumerate(result.keys()):
        plt.subplot(row,col,i+1)
        plt.plot(result[key])
        plt.title(key)
        plt.xlabel("epoch")

    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

if __name__=="__main__":
    train_loss = {
        'total_loss':[],
        'box_loss':[],
        'object_loss':[],
        'no_object_loss':[],
        'class_loss':[],
        'acc':[]
    }
    for i in range(100):
        train_loss['total_loss'].append(np.random.randint(100))
        train_loss['box_loss'].append(np.random.randint(100))
        train_loss['object_loss'].append(np.random.randint(100))
        train_loss['no_object_loss'].append(np.random.randint(100))
        train_loss['class_loss'].append(np.random.randint(100))
        train_loss['acc'].append(np.random.randint(100))

    plot_loss(result=train_loss)