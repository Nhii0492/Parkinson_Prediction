"""
Model Architecture Definition
Reuse ResNet18_CBAM architecture from train.py to load model checkpoint
"""
import torch
import torch.nn as nn
from torchvision import models


class ChannelAttention(nn.Module):
    """Channel Attention Module of CBAM"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module of CBAM with 7x7 kernel"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        x_out = self.bn(x_out)
        return self.sigmoid(x_out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ResNet18_CBAM(nn.Module):
    """
    ResNet18 với CBAM Attention Modules
    
    Architecture:
    - ResNet18 backbone (ImageNet pretrained)
    - CBAM modules sau mỗi residual layer (layer1, layer2, layer3, layer4)
    - Classifier: Dropout -> Linear(512, 256) -> ReLU -> Dropout -> Linear(256, 2)
    """
    def __init__(self, num_classes=2, dropout_prob=0.5):
        super(ResNet18_CBAM, self).__init__()

        # Load pretrained ResNet18
        try:
            from torchvision.models import ResNet18_Weights
            base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except:
            base_model = models.resnet18(pretrained=True)

        # Extract ResNet18 layers
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # CBAM modules with 7x7 kernel
        self.cbam1 = CBAM(64, kernel_size=7)
        self.cbam2 = CBAM(128, kernel_size=7)
        self.cbam3 = CBAM(256, kernel_size=7)
        self.cbam4 = CBAM(512, kernel_size=7)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.cbam1(x)

        x = self.layer2(x)
        x = self.cbam2(x)

        x = self.layer3(x)
        x = self.cbam3(x)

        x = self.layer4(x)
        x = self.cbam4(x)

        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def load_model(model_path: str, device: torch.device = None):
    """
    Load trained model from checkpoint file
    
    Args:
        model_path: Path to .pth file
        device: torch.device (default: cuda if available, else cpu)
    
    Returns:
        model: ResNet18_CBAM model with loaded weights in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with same architecture as training
    model = ResNet18_CBAM(num_classes=2, dropout_prob=0.5)
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

