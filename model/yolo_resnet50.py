import torch.nn as nn
import torchvision.models as models
import torch

class YOLO_resnet50(nn.Module):
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
    def __init__(self, im_size, num_classes, device = "cpu"):
        super(YOLO_resnet50, self).__init__()
        self.im_size = im_size
        self.im_channels = 3
        self.B = 2
        self.S = 7
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

        self.fc_yolo_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.im_size//32 * self.im_size//32 * 512, 4096),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.5),
                nn.Linear(4096, self.S * self.S * (5 * self.B + self.C))
            )
        
    def transform_pred_box(self, pred_box):
        shifts_x = torch.arange(0, self.S, dtype=torch.float32, device=self.device) / self.S
        shifts_y = torch.arange(0, self.S, dtype=torch.float32, device=self.device) / self.S
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        xc = (pred_box[..., 0] + shifts_x) / self.S
        yc = (pred_box[..., 1] + shifts_y) / self.S
        w = pred_box[..., 2]
        h = pred_box[..., 3]
        return torch.stack((xc, yc, w, h), dim=-1)
    
    def yolo_reshape(self, x):
        x = x.reshape((-1,self.S,self.S,self.B*5+self.C))
        pred_cls = torch.softmax(x[..., :self.C], dim=-1)
        pred_score = torch.sigmoid(x[..., self.C:self.C+self.B])
        pred_box = torch.sigmoid(x[..., self.C+self.B:])
        if self.training:
            return torch.cat((pred_cls, pred_score, pred_box), dim=-1)
        else:
            best_score, best_ind = torch.max(pred_score,dim=-1, keepdim=True)
            finale_box = (1-best_ind) * pred_box[..., :4]+ best_ind * pred_box[..., 4:]
            finale_box = self.transform_pred_box(finale_box)
            _, pred_label = pred_cls.max(dim=-1)
            # return torch.Size[batch_size, S, S, 1+1+4]
            return torch.cat((pred_label.unsqueeze(-1), best_score, finale_box), dim=-1)


    def forward(self, x):
        out = self.backbone(x)
        out = self.yolo_conv_layers(out)
        out = self.fc_yolo_layers(out)
        out = self.yolo_reshape(out)
        return out


if __name__ == "__main__":
    input_size = 448
    num_classes = 5
    inp = torch.randn(2, 3, input_size, input_size)
    device = 'cpu'

    model = YOLO_resnet50(im_size=input_size, num_classes=num_classes, device = device).to(device)
    model.train()
    out = model(inp.to(device))
    print(out.shape)

    model.eval()
    out = model(inp.to(device))
    print(out.device)
    print(out.shape)