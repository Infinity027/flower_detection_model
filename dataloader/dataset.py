import torch
from PIL import Image
import os
import yaml
from pathlib import Path
import sys
import numpy as np
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

class FlowerDataset(torch.utils.data.Dataset):
    def __init__(self, yaml_path, image_size=448, phase="train", transform = None):
        '''
        yaml_path: path yaml file contains details of train and val dataset
        phase: either train or val
        '''
        with open(yaml_path, mode="r") as f:
            data_item = yaml.load(f, Loader=yaml.FullLoader)
        f.close() 

        self.phase = phase
        self.class_dict = data_item["CLASS_INFO"]
        self.img_size = data_item["SIZE"]
        
        image_dir = data_item[self.phase.upper()]
        # load images path
        self.image_paths = [os.path.join(image_dir,fn) for fn in os.listdir(image_dir) if fn.lower().endswith(("png", "jpg", "jpeg"))]
        
        #load labels path
        self.label_paths = self.image2label_path(self.image_paths)
        self.generate_no_label(self.label_paths)
        self.size = image_size
        self.transform = transform

    def __len__(self): 
        return len(self.image_paths)

    def __getitem__(self, index):
        """Fetches the image and label by index"""

        im = Image.open(self.image_paths[index]).convert("RGB")
        im = im.resize((self.size, self.size))

        # load lables
        with open(self.label_paths[index], mode="r") as f:
            labels = [x.split() for x in f.read().splitlines()]
        f.close()

        labels = torch.tensor([[float(x) for x in label] for label in labels])
        if self.transform:
           im, labels = self.transform[self.phase](im, labels)

        return im, labels

    def image2label_path(self, image_paths):
        '''
        replace the image file to label file and also change extension of an image
        '''
        sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
        return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in image_paths]
    
    def generate_no_label(self, label_paths):
        '''
        if no label available for an image then create a new label for that image without any class
        '''
        for label_path in label_paths:
            if not os.path.isfile(label_path):
                f = open(str(label_path), mode="w")
                f.close()

    @staticmethod
    def collate_fn(minibatch):
        images, labels = zip(*minibatch)
        images = torch.stack(images, dim=0)
        # labels is a tuple of variable-length tensors; leave them as is or process accordingly.
        return images, labels

if __name__ == "__main__":
    from utils.plot import plot_image_with_label
    from dataloader.augmentation import RandomHorizontalFlip, RandomVerticalFlip

    yaml_path = ROOT / "data" / "flower.yaml"

    # define augmentation
    trans = {
        "train": v2.Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(brightness=0.5, contrast=0.5)], p=0.5),
            v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
            v2.RandomPosterize(bits=4, p=0.2),
            v2.RandomSolarize(threshold=128, p=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ]), 
        "val": v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    }

    traindata = FlowerDataset(yaml_path = yaml_path, phase="train", transform=trans)
    print("Total data:",traindata.__len__())
    k = 6
    # choose random k images
    idxs = np.random.choice(range(traindata.__len__()), k)

    images = []
    labels = []
    for idx in idxs:
        image, label = traindata.__getitem__(idx)
        images.append(image)
        labels.append(label)

    plot_image_with_label(images, labels, traindata.class_dict)

