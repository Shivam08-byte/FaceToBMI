
from config import cfg
from torchvision import models
from torch import nn
import torch
from collections import OrderedDict
import os
from os.path import join

os.environ['TORCH_HOME'] = cfg.pretrained_model_path
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device('cpu')


def get_model(target, type, is_continue):
    resnet50 = models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.requires_grad = False
    fc = nn.Sequential(OrderedDict([
        ("output", nn.Linear(2048, 1))
    ]))
    resnet50.fc = fc

    if is_continue:
        file_name = target + '_' + type + '_best_model.pt'
        print(f"Loading {file_name}")
        file_address = join(cfg.trained_model_path, file_name)
        resnet50.load_state_dict(torch.load(file_address, map_location=device))
        print("Getting model for continue training!")
        return resnet50

    print("Getting new model!")
    return resnet50
