from torchvision.transforms import functional as F, v2, transforms as T
import torch 
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            #target conatins -- [cls, x_center, y_center, w, h]
            if target is not None:
                target[:, 1] = 1 - target[:, 1]
        return img, target
    
class RandomVerticalFlip(T.RandomVerticalFlip):
    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            img = F.vflip(img)
            #target conatins -- [cls, x_center, y_center, w, h]
            if target is not None:
                target[:, 2] = 1 - target[:, 2]
        return img, target

class RandomRotation:
    def __init__(self, degrees, p=0.5):
        """
        Args:
            degrees (float): Maximum absolute rotation angle (in degrees).
            p (float): Probability of applying the rotation.
        """
        self.degrees = degrees
        self.p = p

    def __call__(self, img, target):
        """
        Args:
            img (PIL.Image or Tensor): Input image.
            target (Tensor): Tensor of shape [N, 5] with each row [cls, cx, cy, w, h] in normalized coordinates.
        Returns:
            img, target: Rotated image and adjusted bounding boxes.
        """
        if torch.rand(1) < self.p:
            # Sample a random angle between -degrees and +degrees.
            angle = torch.empty(1).uniform_(-self.degrees, self.degrees).item()
            # Rotate the image (using PIL image functionality if img is PIL, or tensor if already a tensor)
            img = F.rotate(img, angle)

            if target is not None:
                target = target.clone()  # Avoid in-place modifications on a view
                new_boxes = []
                theta = math.radians(angle)
                cos = math.cos(theta)
                sin = math.sin(theta)
                # Image center in normalized coordinates (0.5, 0.5)
                center = torch.tensor([0.5, 0.5])
                # Rotation matrix for a clockwise rotation by theta.
                R = torch.tensor([[cos, -sin],
                                  [sin, cos]])

                for box in target:
                    cls_id, cx, cy, w, h = box.tolist()
                    # Compute corners in normalized coordinates.
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    # Build a tensor of shape [4, 2] for the corners.
                    corners = torch.tensor([
                        [x1, y1],
                        [x1, y2],
                        [x2, y1],
                        [x2, y2]
                    ])
                    # Shift corners to be relative to the image center.
                    corners = corners - center
                    # Apply the rotation.
                    rotated = torch.matmul(corners, R.T)
                    # Shift back to normalized coordinates.
                    rotated = rotated + center
                    # Compute the new axis-aligned bounding box.
                    new_x1 = rotated[:, 0].min().item()
                    new_y1 = rotated[:, 1].min().item()
                    new_x2 = rotated[:, 0].max().item()
                    new_y2 = rotated[:, 1].max().item()
                    new_cx = (new_x1 + new_x2) / 2
                    new_cy = (new_y1 + new_y2) / 2
                    new_w = new_x2 - new_x1
                    new_h = new_y2 - new_y1
                    new_boxes.append([cls_id, new_cx, new_cy, new_w, new_h])
                # Convert list back to a tensor.
                target = torch.tensor(new_boxes, dtype=torch.float32)
        return img, target


if __name__ == "__main__":
    from utils.plot import plot_image_with_label
    from dataloader.dataset import FlowerDataset
    CLASS_INFO= {
        0: 'rose',
        1: 'lotus',
        2: 'hibiscuss',
        3: 'marigold',
        4: 'sunflower'
    }
    yaml_path = ROOT / "data" / "flower.yaml"
    
    trans = {
        "train": v2.Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=5, p=1),
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

    dataset = FlowerDataset(yaml_path=yaml_path, phase="train", transform=trans)
    k = 4
    # choose random k images
    idxs = np.random.choice(range(dataset.__len__()), k)

    images = []
    labels = []
    for idx in idxs:
        image, label = dataset.__getitem__(idx)
        images.append(image)
        labels.append(label)

    plot_image_with_label(images, labels, dataset.class_dict, save_name="results/augmentation.png")
    



