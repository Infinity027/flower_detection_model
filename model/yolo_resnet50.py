import torch.nn as nn
import torchvision.models as models
import torch

class YOLOResnet50(nn.Module):
    """
    Model with three components:
    1. Backbone of resnet50 pretrained on 224x224 images from Imagenet
    2. 4 Conv,Batchnorm,LeakyReLU Layers for Yolo Detection Head
    3. Fc layers with final layer having S*S*(5B+C) output dimensions
        Final layer predicts [
            x_offset_box1,y_offset_box1,sqrt_w_box1,sqrt_h_box1,conf_box1, # box-1 params
            ...,
            x_offset_boxB,y_offset_boxB,sqrt_w_boxB,sqrt_h_boxB,conf_boxB, # box-B params
            p1, p2, ...., pC-1, pC  # class conditional probabilities
        ] for each S*S grid cell
    """
    def __init__(self, im_size, num_classes, S=7, B=2, device = "cpu"):
        super(YOLOResnet50, self).__init__()
        self.im_size = im_size
        self.im_channels = 3
        self.B = B
        self.S = S
        self.C = num_classes
        self.device = device

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the last fully connected layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Freeze the base model
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Define additional layers
        self.yolo_conv_layers = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
        
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
        
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1)
        )

        self.yolo_fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.im_size//32 * self.im_size//32 * 512, 496),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),
                nn.Linear(496, self.S * self.S * (5 * self.B + self.C))
            )

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.yolo_fc_layers(self.yolo_conv_layers(self.backbone(x)))
        pred_cls = out[...,:self.C]
        pred_obj_box = torch.sigmoid(out[...,self.C:])
        out = torch.cat((pred_cls, pred_obj_box), dim=-1)

        #shape = [batch_size, -1, classes+(conf + xc + yc + wh)*self.B]
        return out.reshape(batch_size,-1,self.B*5+self.C)

if __name__ == "__main__":
    input_size = 448
    num_classes = 5
    inp = torch.randn(2, 3, input_size, input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLOResnet50(im_size=input_size, num_classes=num_classes, device = device).to(device)
    model.train()
    out = model(inp.to(device))
    print(out.shape)