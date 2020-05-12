
from src.config import cfg
from torchvision import models
from torch import nn
import torch
from collections import OrderedDict
import os

os.environ['TORCH_HOME'] = cfg.pretrained_model_path


def get_resnet_model():
    resnet50 = models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.requires_grad = False
    fc = nn.Sequential(OrderedDict([
        ("output", nn.Linear(2048, 1))
    ]))
    resnet50.fc = fc
    # Saving model for training multiple times
    # torch.save(resnet50.state_dict(), cfg.init_model_path)
    return resnet50
