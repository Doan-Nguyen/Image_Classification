import logging
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import flask
from flask import request, jsonify
from PIL import Image
import json

import torch
from torchvision import transforms
from torchvision import datasets

from conf import settings
from utils_ai import build_network, predict_author_single_img

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)

logging.info('Start program')
handler = logging.FileHandler('infor.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)
   
if __name__ == '__main__':
    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default= 'squeezenet', help='net type')
    parser.add_argument('-weights', type=str, 
                        default='./checkpoint/results/stamp_squeezenet-280-regular.pth',
                        help='the weights file path you want to test')
    
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    # keep for training new author
    parser.add_argument('-num_w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    
    args_dict = vars(parser.parse_args())
    logger.info(args_dict)

    net_type = args_dict['net']
    use_gpu = args_dict['gpu']
    net = build_network(archi = net_type, use_gpu=use_gpu ) 
    logger.info(net)

    net.load_state_dict(torch.load(args_dict['weights']), args_dict['gpu'])
    net.eval()

    test_image_dir = 'D:/SealProjectOLD/Datasets/images/val'
    dataset = datasets.ImageFolder(test_image_dir, transform= None)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    count_authors = len([name for name in os.listdir(test_image_dir) if os.path.isdir(os.path.join(test_image_dir, name))])
    json_file = open('author_data.json')
    list_author = json.load(json_file)
    for subfolder in list_author.keys():
        subfolder_path = os.path.join(test_image_dir, subfolder)

        for img in os.listdir(subfolder_path):
            test_img_path = os.path.join(subfolder_path, img)
            author, confidence = predict_author_single_img(net, idx_to_class=idx_to_class, image_path=test_img_path)
            print("Author: ", subfolder, ", Predict: ",author, ", Confidence: ", confidence)
            