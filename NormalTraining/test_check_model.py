import argparse
import os
import logging
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from conf import global_settings
from utils_ai import build_network, get_training_dataloader, get_test_dataloader, WarmUpLR, load_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument('-net', default = 'vgg16', type=str, help='net type')
parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
parser.add_argument('-b', type=int, default=48, help='batch size for dataloader')
parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
parser.add_argument('-warm', type=int, default=2, help='warm up training phase')
parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
args = parser.parse_args()
args_dict = vars(parser.parse_args())

###     
def load_checkpoint(filepath):
    # checkpoint = torch.load(filepath)
    # model = checkpoint['model']
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(torch.load(filepath), args.gpu)
    model.eval()
    print(model)
    for parameter in model.parameters():
        parameter.requires_grad = False

    
    return model

load_checkpoint('vgg16-84-best.pth')