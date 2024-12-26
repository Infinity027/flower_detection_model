import torch
import cv2
import os
import yaml
import numpy as np
from pathlib import Path
import sys
# from transform import BasicTransform, to_tensor



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, yaml_path, image_size=448, S=7, B=2, C=5, phase="train", transform = None):
        '''
        yaml_path: path yaml file contains details of train and val dataset
        phase: either train or val
        S: grid size (default=7)
        B: bounding box per cell (default=2)
        C: Number of classes (default=5)
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
        self.label_paths = self.image2label_path(self.image_paths)
        self.generate_no_label(self.label_paths)
        self.S = S
        self.B = B 
        self.C = C
        self.size = image_size
        self.transform = transform


    def __len__(self): return len(self.image_paths)

    def __getitem__(self, index):
        """Fetches the image and label by index"""
        image = self.get_image(index)
        label_matrix = self.get_label(index)

        return image, label_matrix
    
    def get_image(self, index):
        '''
        load image and convert to tensor format
        '''
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(self.size,self.size))
        im_tensor = torch.from_numpy(image / 255.).permute((2, 0, 1)).float()
        im_tensor_channel_0 = (torch.unsqueeze(im_tensor[0], 0) - 0.485) / 0.229
        im_tensor_channel_1 = (torch.unsqueeze(im_tensor[1], 0) - 0.456) / 0.224
        im_tensor_channel_2 = (torch.unsqueeze(im_tensor[2], 0) - 0.406) / 0.225
        im_tensor = torch.cat((im_tensor_channel_0,
                               im_tensor_channel_1,
                               im_tensor_channel_2), 0)
        return im_tensor

    def get_label(self, index):
        with open(self.label_paths[index], mode="r") as f:
            item = [x.split() for x in f.read().splitlines()]
        f.close()
        label = np.array(item, dtype=np.float32)
        label_matrix = np.zeros([self.S, self.S, 5+self.C],dtype=np.float32)
        for l in label:
            loc_i = int(l[1]*self.S)
            loc_j = int(l[2]*self.S)
            cls = int(l[0])
            if label_matrix[loc_i,loc_j,0]==0:
                label_matrix[loc_i,loc_j,cls]=1    #one-hot encoding
                label_matrix[loc_i,loc_j,self.C] = l[1]*self.S-loc_i  #relative x
                label_matrix[loc_i,loc_j,self.C+1] = l[2]*self.S-loc_j #relative y
                label_matrix[loc_i,loc_j,self.C+2] = l[3]  #width
                label_matrix[loc_i,loc_j,self.C+3] = l[4]  #height
                label_matrix[loc_i,loc_j,self.C+4] = 1     #objectness score
        return label_matrix    

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

if __name__ == "__main__":
    yaml_path = 'flower.yaml'
    traindata = YoloDataset(yaml_path = yaml_path)
    print("Total data:",traindata.__len__())