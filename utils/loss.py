import torch.nn as nn
import torch
from utils.metrics import iou

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=5):
        super(YoloLoss, self).__init__()
        self.S, self.B, self.C = S, B, C
        self.lambda_coord, self.lambda_noobj = 5, .5
        self.mse = nn.MSELoss(reduction="sum")


    def forward(self, preds, target):
        #reshape the prediction into proper dimension of target
        self.batch_size = preds.shape[0]
        ious = torch.cat(
            [iou(preds[...,7:11], target[...,6:10]).unsqueeze(0),
            iou(preds[...,11:15], target[...,6:10]).unsqueeze(0)], dim=0   
        )
        # print(ious.shape)
        _, bestbox = torch.max(ious, dim=0) # return the best bounding box among two
        exists_box = target[...,5:6]  #in paper this is Iobj_i
        # print(bestbox.shape)    
        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        # select the best box out of two boxes
        box_preds =  exists_box * ( (1-bestbox)*preds[...,7:11] + bestbox*preds[...,11:15] )
        # Take sqrt of width, height of boxes, may be minus values comes thats why abs and sign method use
        box_preds[...,2:4] = torch.sign(box_preds[...,2:4])*torch.sqrt(torch.abs(box_preds[...,2:4]+1e-6))

        box_targets = exists_box * target[...,6:10] # if object present then only take the bounding box
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4]) #target weight and height square root

        box_loss = self.mse(
            torch.flatten(box_preds, end_dim=-2),  # 
            torch.flatten(box_targets, end_dim=-2)
        ) / self.batch_size

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_score =   bestbox * preds[..., 6:7] + (1 - bestbox) * preds[..., 5:6]
        object_loss = self.mse(
            torch.flatten(exists_box*pred_score), torch.flatten(exists_box)
        ) / self.batch_size

         # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss=self.mse(
            torch.flatten((1-exists_box) * preds[...,5:6], end_dim=-2),
            torch.flatten((1-exists_box) * target[...,5:6], end_dim=-2)
        ) / self.batch_size
        no_object_loss += self.mse(
            torch.flatten((1-exists_box) * preds[...,6:7], end_dim=-2),
            torch.flatten((1-exists_box) * target[...,5:6], end_dim=-2)
        ) / self.batch_size

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * preds[...,:5], end_dim=-2),
            torch.flatten(exists_box * target[...,:5], end_dim=-2)
        ) / self.batch_size
        
        total_loss = (self.lambda_coord*box_loss + object_loss + self.lambda_noobj*no_object_loss + class_loss)
        return total_loss, self.lambda_coord*box_loss, object_loss + self.lambda_noobj*no_object_loss, class_loss

if __name__ == "__main__":
    loss = YoloLoss()
    preds = torch.rand(4,7,7,15)
    target = torch.rand(4,7,7,10)
    total_loss, box_loss, obj_loss, cls_loss = loss(preds=preds,target=target)
    print("Totol loss:", total_loss)
    print("box loss:", box_loss)
    print("obj loss:", obj_loss)
    print("cls loss:", cls_loss)
