import torch.nn as nn
import torch
import sys 
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=5, label_smooth=0.0, device="cpu"):
        super(YoloLoss, self).__init__()
        self.S, self.B, self.C = S, B, C
        self.device = device
        self.lambda_coord, self.lambda_noobj = 5, 1
        self.mse =  nn.MSELoss(reduction='none')
        self.class_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smooth)
        grid_x, grid_y = torch.meshgrid((torch.arange(self.S), torch.arange(self.S)), indexing="ij")
        self.grid_x = grid_x.contiguous().view((1, -1))
        self.grid_y = grid_y.contiguous().view((1, -1))


    def forward(self, preds:torch.tensor, targets:list):
        #reshape the prediction into proper dimension of target
        self.batch_size = preds.shape[0]
        preds = preds.to(self.device)
        #target shape: [N, C + 5]
        targets = self.batch_build_target(targets).to(self.device)

        #predictions shape: [batch_size, S*S, C + B*5] with B=2.
        # Split predictions: first C are class predictions, then for each of the 2 boxes, we have [conf, x, y, w, h]
        pred_cls = preds[..., :self.C].permute(0,2,1)
        pred_boxes = preds[..., self.C:].view(self.batch_size, -1, 2, 5)
        # pred_boxes[..., b, 0] is the confidence for box b and pred_boxes[..., b, 1:5] is the bbox (cx,cy,w,h)

        # Build target components.
        target_obj = (targets[..., 1] == 1).float()        #only conatins True and False values [N, 49]
        target_noobj = (targets[..., 1] == 0).float()       #only conatins True and False values
        target_cls = targets[..., 0].long()                # [N, S*S]
        with torch.no_grad():
            #target last dim 2-6 and pred_box 1-5
            iou1 = self.calculate_iou(pred_boxes[...,0, 1:], targets[..., 2:])
            iou2 = self.calculate_iou(pred_boxes[...,1, 1:], targets[..., 2:])
        
        # For cells with an object, select the box with higher IoU.
        best_box_mask = (iou1 >= iou2).unsqueeze(-1)  # [N, S*S, 1]
        # print(best_box_mask)
        # print(best_box_mask.shape)
        best_pred_box = torch.where(best_box_mask, pred_boxes[..., 0, 1:], pred_boxes[..., 1, 1:])
        best_pred_conf = torch.where(best_box_mask.squeeze(-1), pred_boxes[..., 0, 0], pred_boxes[..., 1, 0])

        # ======= Object loss [only for the responsible box] =======
        obj_loss = self.mse(best_pred_conf, torch.max(iou1, iou2)) * target_obj
        obj_loss = obj_loss.sum()/self.batch_size

        # ======= No Object loss [penalize the other box's confidence] =======
        noobj_loss_b1 = self.mse(pred_boxes[...,0, 0], targets[..., 1]*0) * target_noobj
        noobj_loss_b2 = self.mse(pred_boxes[...,1, 0], targets[..., 1]*0) * target_noobj
        noobj_loss = (noobj_loss_b1.sum() + noobj_loss_b2.sum())/self.batch_size

        # ======= box loss =======
        # Box (localization) loss: use the best predicted box and sqrt of width and height
        pred_wh = torch.sqrt(best_pred_box[..., 2:])
        targets_wh = torch.sqrt(targets[..., 4:])
        box_loss_xy = self.mse(best_pred_box[..., :2], targets[..., 2:4]).sum(dim=-1)
        box_loss_wh = self.mse(pred_wh, targets_wh).sum(dim=-1)
        box_loss = ((box_loss_xy+box_loss_wh)* target_obj).sum() / self.batch_size

        # ======= class loss =======
        cls_loss = self.class_loss(pred_cls, target_cls) * target_obj
        cls_loss = cls_loss.sum() / self.batch_size

        total_loss = obj_loss + self.lambda_noobj*noobj_loss + self.lambda_coord*box_loss + cls_loss

        # print(f"Total loss: {total_loss} | obj_loss: {obj_loss} | noobj_loss: {noobj_loss} | box_loss: {box_loss} | cls_loss: {cls_loss}")
        return [total_loss, obj_loss, noobj_loss, box_loss, cls_loss]
    
    def build_target(self, label):
        #create label matrix for each image - [grid_size, grid_size, 1 + bounding box + confidence]
        target = torch.zeros((self.S, self.S, 1+4+1), dtype=torch.float32)
        # print(target.shape)
        for box in label:
            x_cell, y_cell = int(box[1]*self.S), int(box[2]*self.S)
            if target[y_cell, x_cell, 1] == 0:
                target[y_cell, x_cell, 1] = 1
                target[y_cell, x_cell, 2:4] = torch.Tensor([box[1]*self.S - x_cell, box[2]*self.S - y_cell])
                target[y_cell, x_cell, 4:] = box[3:5]
                target[y_cell, x_cell, 0] = box[0]
            else:
                if target[y_cell, x_cell, 4]*target[y_cell, x_cell, 5] < box[3]*box[4]:
                    target[y_cell, x_cell, 2:4] = torch.Tensor([box[1]*self.S - x_cell, box[2]*self.S - y_cell])
                    target[y_cell, x_cell, 4:] = box[3:5]
                    target[y_cell, x_cell, 0] = box[0]
           
        return target
    
    def batch_build_target(self, labels):
        batch_target = torch.stack([self.build_target(label) for label in labels], dim=0)
        return batch_target.view(self.batch_size, -1,  1+4+1)
    
    def calculate_iou(self, pred_box_cxcywh, target_box_cxcywh):
        pred_box_x1y1x2y2 = self.transform_cxcywh_to_x1y1x2y2(pred_box_cxcywh)
        target_box_x1y1x2y2 = self.transform_cxcywh_to_x1y1x2y2(target_box_cxcywh)

        x1 = torch.max(pred_box_x1y1x2y2[..., 0], target_box_x1y1x2y2[..., 0])
        y1 = torch.max(pred_box_x1y1x2y2[..., 1], target_box_x1y1x2y2[..., 1])
        x2 = torch.min(pred_box_x1y1x2y2[..., 2], target_box_x1y1x2y2[..., 2])
        y2 = torch.min(pred_box_x1y1x2y2[..., 3], target_box_x1y1x2y2[..., 3])
        
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        union = abs(pred_box_cxcywh[..., 2] * pred_box_cxcywh[..., 3]) + abs(target_box_cxcywh[..., 2] * target_box_cxcywh[..., 3]) - inter
        inter[inter.gt(0)] = inter[inter.gt(0)] / union[inter.gt(0)]
        return inter
    
    def transform_cxcywh_to_x1y1x2y2(self, boxes):
        xc = (boxes[..., 0] + self.grid_x.to(self.device)) / self.S
        yc = (boxes[..., 1] + self.grid_y.to(self.device)) / self.S
        x1 = xc - boxes[..., 2] / 2
        y1 = yc - boxes[..., 3] / 2
        x2 = xc + boxes[..., 2] / 2
        y2 = yc + boxes[..., 3] / 2
        return torch.stack((x1, y1, x2, y2), dim=-1)

if __name__ == "__main__":
    from dataloader.dataset import FlowerDataset
    from torchvision.transforms import v2
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
    
    loss = YoloLoss(device="cuda")
    preds = torch.rand(4,49,15)
    val_data = FlowerDataset(yaml_path=yaml_path, phase="val", transform=trans)
    val_loader = DataLoader(val_data, 
                            collate_fn=FlowerDataset.collate_fn, 
                            batch_size=4,
                            shuffle=True)

    for i, (images, labels) in enumerate(val_loader):
        break

    total_loss, obj_loss, noobj_loss, box_loss, cls_loss = loss(preds=preds,targets=labels)
    print("Totol loss:", total_loss)
    print("box loss:", box_loss)
    print("obj loss:", obj_loss)
    print("noobj loss:", noobj_loss)
    print("cls loss:", cls_loss)
