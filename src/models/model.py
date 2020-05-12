
from src.config import cfg
from torchvision import models
from collections import OrderedDict
import os
os.environ['TORCH_HOME'] = cfg.pretrained_model_path


def get_resnet_model():
    resnet50 = models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.requires_grad = False

    return resnet50


def get_f2b_model(ignore_age_weights=False):
    model = get_resnet_model()
    return model
