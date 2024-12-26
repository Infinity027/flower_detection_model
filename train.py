import os
import numpy as np
import torch
import sys
import argparse
from collections import defaultdict
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.yolo_resnet50 import YOLO_resnet50
from data.dataset import YoloDataset
from utils.metrics import ( cellboxes_to_boxes,
                            transform_pred_box,
                            non_max_suppression,
                            load_checkpoint,
                            mean_average_precision
                        )
from utils.loss import YoloLoss


seed = 123
torch.manual_seed(seed)
LEARNING_RATE = 0.001
DEVICE = "cpu"
BATCH_SIZE = 16 
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "save.pt"
IMG_DIR = "data/flower_dataset/images"
LABEL_DIR = "data/flower_dataset/labels"

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    loss_type = ["total_loss", "box_loss", "obj_loss", "cls_loss"]
    losses = defaultdict(float)
    model.train()
    optimizer.zero_grad()
    for _, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)

        for loss_name, loss_value in zip(loss_type, loss):
            if not torch.isfinite(loss_value) and loss_name != "total_loss":
                print(f"############## {loss_name} Loss is Nan/Inf ! {loss_value} ##############")
                sys.exit(0)
            else:
                losses[loss_name] += loss_value.item()
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss[0].item())
    # del images, predictions
    # torch.cuda.empty_cache()
    for loss_name in loss_type:
        losses[loss_name] /= len(train_loader)
    return losses

@torch.no_grad()
def validate(args, val_loader, model, loss_fn):
    loop = tqdm(val_loader, leave=True)
    all_pred_boxes = []
    all_true_boxes = []
    train_idx = 0
    model.eval()
    losses = {"total_loss":0.0, "box_loss":0.0, "obj_loss":0.0, "cls_loss":0.0}
    for _, (x,y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            pred = model(x)
        batch_size = x.shape[0]
        _, target_label = y[..., :5].max(dim=-1)
        target_score = y[..., 5:6]
        target_box = transform_pred_box(y[..., 6:])
        y = torch.cat((target_label.unsqueeze(-1), target_score, target_box), dim=-1)
        true_bboxes = cellboxes_to_boxes(y)
        bboxes = cellboxes_to_boxes(pred)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=args.nms_thres,
                threshold=args.conf_thres,
                box_format='midpoint'
            )
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > args.conf_thres:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1
    mean_avg_prec = mean_average_precision(all_pred_boxes, all_true_boxes, iou_threshold=0.5, box_format="midpoint")
    print(f"Train mAP: {mean_avg_prec}")


def parse_args(make_dirs=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="log file name")
    parser.add_argument("--data", type=str, default="flower.yaml", help="Path to data.yaml")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--base-lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--lr-decay", nargs="+", default=[40, 45], type=int, help="Epoch to learning rate decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="Threshold to filter confidence score")
    parser.add_argument("--nms-thres", type=float, default=0.6, help="Threshold to filter Box IoU of NMS process")
    parser.add_argument("--resume", action="store_true", help="Name to resume path")
    
    args = parser.parse_args()
    args.data = os.path.join("data",args.data)
    return args

def main():
    args = parse_args(make_dirs=True)
    model = YOLO_resnet50(im_size=448, num_classes=5, device = DEVICE).to(DEVICE)
    train_dataset = YoloDataset(yaml_path=args.data, phase="train")
    val_dataset = YoloDataset(yaml_path=args.data, phase="val")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    loss_fn = YoloLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay, gamma=0.1)
    if LOAD_MODEL:
        load_checkpoint(torch.load(args.load_path), model, optimizer)
    loss = []
    # args.class_list = train_dataset.class_list
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}/{args.num_epochs}:")
        losses = train_fn(train_loader, model, optimizer, loss_fn)
        print(f"Total loss:{losses["total_loss"]} | box loss:{losses["box_loss"]} | obj loss:{losses["obj_loss"]} | cls loss:{losses["cls_loss"]}")
        loss.append(losses.values())
        validate(args, val_loader, model, loss_fn)
        if epoch+1==10:
            print("=> Saving checkpoint")
            torch.save(model, args.weight_dir / "best.pt")
        scheduler.step()

if __name__ == "__main__":
    main()