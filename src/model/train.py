import cv2
import torch
import numpy as np
from config import cfg
from data.data import train_val_test_split

train_on_gpu = torch.cuda.is_available()


def train_top_layer(model):
    pass


def train_all_layers(model):
    pass


def test_model(model):
    pass
