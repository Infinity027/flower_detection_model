import torch
import torchvision.ops as ops
import numpy as np
from pathlib import Path 
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.metrics import cxcywh_to_x1y1x2y2 
def decode_predictions(images, predictions, C, S=7, conf_thresh=0.5, nms_thresh=0.6, plot=None):
    """
    Convert model output to list of bounding boxes in cxcywh format.
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    pred_boxes = []
    
    for i in range(batch_size):
        #pred shape: [S*S, (C + 5*B)]
        pred = predictions[i]
        
        # Get class probabilities and bounding boxes
        class_probs = pred[:, :C]
        box_data = pred[:, C:].view(-1,2,5)

        #converting to absolute center point
        for i, box in enumerate(box_data):
            box[:,1] = (box[:,1] + i%S)/S
            box[:,2] = (box[:,2] + i//S)/S

        select_box = torch.where((box_data[:,0,0]>box_data[:,1,0]).unsqueeze(-1), box_data[:,0,:], box_data[:,1,:])
        mask = (select_box[:,0]>conf_thresh)

        boxes = select_box[mask,1:]
        scores = select_box[mask,0]
        labels = torch.argmax(class_probs[mask], dim=-1)

        #apply nms 
        if boxes.shape[0] > 0:
            # Convert xcycwh to x1y1x2y2 format
            boxes_xyxy = cxcywh_to_x1y1x2y2(boxes)

            keep_indices = ops.nms(boxes_xyxy, scores, nms_thresh)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
        else:
            boxes = torch.empty((0, 4))
            scores = torch.empty((0,))
            labels = torch.empty((0,), dtype=torch.int64)

        pred_boxes.append({
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        })

    select_pred = []
    if plot!=None:
        from utils.plot import plot_image_with_label
        idxs = np.random.choice(range(batch_size),6)
        for idx in idxs:
            select_pred.append(torch.cat((pred_boxes[idx]['labels'].unsqueeze(-1), 
                                         pred_boxes[idx]['scores'].unsqueeze(-1), 
                                         pred_boxes[idx]['boxes']), -1))
            
        class_name = {0:'rose',
                      1:'lotus',
                      2:'hibiscuss',
                      3:'marigold',
                      4:'sunflower'}
        save_name = f"results/pred_plot_{plot}.png"
        plot_image_with_label(images[idxs].to("cpu"), select_pred, class_name, save_name=save_name)

        # print(select_pred)

        return pred_boxes, idxs
    else:
        return pred_boxes

def process_targets(targets):
    """
    Convert targets to list of dictionaries with "boxes" and "labels".
    """
    true_boxes = []
    for target in targets:
        boxes = []
        labels = []
        for obj in target:
            #target format: [class, cx, cy, w, h]
            
            boxes.append([obj[1], obj[2], obj[3], obj[4]])
            labels.append(int(obj[0]))
        
        true_boxes.append({
            "boxes": torch.tensor(boxes),
            "labels": torch.tensor(labels),
        })
    return true_boxes

def accuracy(pred:torch.tensor, target:list, S=7, C=5, conf=0.5):
    """
    Calculate classification accuracy for objects in grid cells.
    
    Args:
        pred (Tensor): Model output of shape [batch_size, S*S, (C + 5*B)].
        target (list): List of object, each containing objects [cls_id, xc, yc, w, h].
        S (int): Grid size (default=7).
        C (int): Number of classes (default=5).
    
    Returns:
        float: Classification accuracy.
    """
    class_pred = []
    pred_cls = pred[..., :C].view(-1,S,S,5)
    box_data = pred[..., C:].view(-1,S,S,2,5)
    # print((box_data[...,0,0]>box_data[...,1,0]))
    # print(box_data[...,0,1])
    select_box = torch.where((box_data[...,0,0]>box_data[...,1,0]), 
                             box_data[...,0,0], 
                             box_data[...,1,0]).unsqueeze(-1) 
    # print(select_box.shape)
    mask = (select_box[...,0]>conf)
    pred_cls = pred_cls*mask.unsqueeze(-1)
    # print(pred_cls.shape)

    for batch_pred, batch_target in zip(pred_cls, target):
        for obj in batch_target:
            x_cell = min(int(obj[1]*S), S-1)
            y_cell = min(int(obj[2]*S), S-1)
            if batch_pred[x_cell, y_cell].sum().item() > 0:
            
                # Check if prediction matches target 
                class_pred.append(torch.argmax(batch_pred[x_cell, y_cell]).item() == obj[0].item())
    
    # Calculate accuracy
    correct = sum(class_pred)
    total1 = len(class_pred)
    total = mask.sum().item()

    # print(len(class_pred))
    # print(correct)
    # print(total)
    return correct / total1 if total1 > 0 else 0.0

if __name__=="__main__":

    from dataloader.dataset import FlowerDataset
    from torchvision.transforms import v2
    from torch.utils.data import DataLoader
    yaml_path = ROOT / "data" / "flower.yaml"

    trans = {
        "train": v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ]),
        "val": v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    }
    
    preds = torch.rand(4,49,15)
    val_data = FlowerDataset(yaml_path=yaml_path, phase="val", transform=trans)
    val_loader = DataLoader(val_data, 
                            collate_fn=FlowerDataset.collate_fn, 
                            batch_size=4,
                            shuffle=True)

    for i, (images, labels) in enumerate(val_loader):
        break


    print(accuracy(preds, labels, S=7, C=5, conf=0.5))
    print(decode_predictions(images, preds, C=5))
