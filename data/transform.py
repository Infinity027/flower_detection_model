import cv2
import torch
import numpy as np


MEAN = 0.485, 0.456, 0.406 # RGB
STD = 0.229, 0.224, 0.225 # RGB

def to_tensor(image):
    # convert channel last to channel first, important for pytorch model training
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image).float()


def to_image(tensor, mean=MEAN, std=STD):
    #convert tensor to image 
    denorm_tensor = tensor.clone()
    for t, m, s in zip(denorm_tensor, mean, std):
        t.mul_(s).add_(m)
    denorm_tensor.clamp_(min=0, max=1.)
    denorm_tensor *= 255
    image = denorm_tensor.permute(1,2,0).numpy().astype(np.uint8)
    return image

def denormalize(image, mean=MEAN, std=STD):
    image *= std
    image += mean
    image *= 255.
    return image.astype(np.uint8)

class LetterBox:
    '''
    This class maintain the aspect ratio of an image after convert into specific dimension

    new_shape: new dimension shape as list
    color: specifiy the background color
    '''
    def __init__(self, new_shape=448, color=(0, 0, 0)):
        self.new_shape = new_shape
        self.color = color

    def __call__(self, image, boxes, labels=None):
        # Resize and pad image while meeting stride-multiple constraints
        shape = image.shape[:2]  # current shape [height, width]
        # Scale ratio (new / old)
        r = min(self.new_shape / shape[0], self.new_shape / shape[1])
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # [width, height]
        dw, dh = self.new_shape - new_unpad[0], self.new_shape - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        boxes[:, :2] = (boxes[:, :2] * (new_unpad[0], new_unpad[1]) + (left, top))
        boxes[:, :2] /= (image.shape[1], image.shape[0])
        boxes[:, 2:] /= (image.shape[1] / new_unpad[0], image.shape[0] / new_unpad[1])
        return image, boxes, labels

class BasicTransform:
    def __init__(self, input_size, mean=MEAN, std=STD):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        self.tfs = Compose([
            LetterBox(new_shape=input_size),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image, boxes, labels = self.tfs(image, boxes, labels)
        return image, boxes, labels
    
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes=None, labels=None):
        for tf in self.transforms:
            image, boxes, labels = tf(image, boxes, labels)
        return image, boxes, labels
    
class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image /= 255.
        image -= self.mean
        image /= self.std
        return image, boxes, labels
    
class ToX1Y1X2Y2:
    #convert xywh to x1y1x2y2
    def __call__(self, image, boxes, labels=None):
        x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
        x2y2 = boxes[:, :2] + boxes[:, 2:] / 2
        boxes = np.concatenate((x1y1, x2y2), axis=1).clip(min=0, max=1)
        return image, boxes, labels

class ToXYWH:
    #convert x1y1x2y2 to xywh
    def __call__(self, image, boxes, labels=None):
        wh = boxes[:, 2:] - boxes[:, :2]
        xy = boxes[:, :2] + wh / 2
        boxes = np.concatenate((xy, wh), axis=1).clip(min=0, max=1)
        return image, boxes, labels