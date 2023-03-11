import torch
import torch.nn as nn

from torchvision import models, transforms

def add_channels(model, channels):
    weight_indices = {'red': 0, 'green': 1, 'blue': 2, 'nir': 0, 'red_edge': 0}
    weight = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(len(channels), 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        for i, channel in enumerate(channels):
            model.conv1.weight[:, i] = weight[:, weight_indices[channel]]
    return model

def full_precision_model(channels):
    model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    model = add_channels(model, channels)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model