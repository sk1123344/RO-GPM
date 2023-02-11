from torchvision.models import resnet18
import torch
import torch.nn as nn


class IncrementalBaseModel(nn.Module):
    def __init__(self):
        super(IncrementalBaseModel, self).__init__()
        self.backbone = None
        self.fc = None

    def incremental_fc(self, incremental_num_classes):
        weight = self.fc.weight.data
        bias = self.fc.bias.data if hasattr(self.fc, 'bias') else None
        in_feature, out_feature = self.fc.in_features, self.fc.out_features

        self.fc = nn.Linear(in_feature, incremental_num_classes + out_feature, bias=True if bias is not None else False)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        if bias is not None:
            self.fc.bias.data[:out_feature] = bias[:out_feature]

    def freeze_all(self):
        for param_ in self.backbone.parameters():
            param_.requires_grad = False
        for param_ in self.fc.parameters():
            param_.requires_grad = False
        self.freeze_bn()
        self.backbone.eval()
        self.fc.eval()

    def freeze_bn(self):
        for n, p in self.backbone.named_modules():
            if isinstance(p, nn.BatchNorm2d):
                p.eval()
                p.weight.requires_grad = False
                p.bias.requires_grad = False

    def feature_extract(self, input_):
        return self.backbone(input_)

    def get_feature_length(self):
        return self.fc.in_features


class ResNet18Fundus(IncrementalBaseModel):
    def __init__(self, num_classes=50, bias=True, use_pretrained_backbone=False):
        super(ResNet18Fundus, self).__init__()
        self.backbone = resnet18(num_classes=1000, pretrained=use_pretrained_backbone)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, num_classes, bias=bias)

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)


