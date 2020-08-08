#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model
author baiyu
"""

import argparse
from torchvision import datasets
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from conf import settings
from utils_ai import get_test_dataloader, build_network
import os
from PIL import Image
from torch import unsqueeze
import time
import json

import warnings
warnings.filterwarnings('ignore')
import logging
import sqlite3

import time

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.CRITICAL)

logging.info('Start program')
handler = logging.FileHandler('test.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the file handler to the logger
logger.addHandler(handler)

### Predict images
def predict(model, idx_to_class, image, subfolder):
    txt_file = open('resutls_predict_signature.txt', 'a+')
    device = torch.device("cuda")
    image_transforms = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)])

    test_img = Image.open(image)
    test_img_tensor = image_transforms(test_img)

    if torch.cuda.is_available():
        test_img_tensor = test_img_tensor.view(1, 3, 112, 112).cuda()
    else:
        test_img_tensor = test_img_tensor.view(1, 3, 112, 112)
    i = 0
    with torch.no_grad():
        model.eval()
        # model ouputs log probabilities
        out = model(test_img_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(3, dim=1)
        sum_topk = int(topk.cpu().numpy()[0][0]) + int(topk.cpu().numpy()[0][1]) + int(topk.cpu().numpy()[0][2])
        print("Author: ", subfolder, ", Predict: ", idx_to_class[topclass.cpu().numpy()[0][0]], ", Confidence: ", int(topk.cpu().numpy()[0][0])/sum_topk)
        line = "Author: " + subfolder + ", Predict: " + str(idx_to_class[topclass.cpu().numpy()[0][0]]) + ", Confidence: "+ str((topk.cpu().numpy()[0][0])/sum_topk) +'\n'
        if (str(subfolder) == str(idx_to_class[topclass.cpu().numpy()[0][0]])):
            i =  1
        txt_file.writelines(line)
    return i , (topk.cpu().numpy()[0][0])/sum_topk
    
    
if __name__ == '__main__':
    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default= 'squeezenet', help='net type')
    ### 
    # parser.add_argument('-weights', type=str, default='./checkpoint/results/sign_squeezenet-280-regular.pth', help='the weights file path you want to test')
    parser.add_argument('-weights', type=str, default='./checkpoint/results/squeezenet-51-best.pth', help='the weights file path you want to test')
    
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    args = parser.parse_args()
    args_dict = vars(parser.parse_args())
    logger.info(args_dict)
    net_type = args_dict['net']
    use_gpu = args_dict['gpu']
    
    standard_folder = "D:/SealProjectOLD/Datasets/images/train"
    list_author = next(os.walk(standard_folder))[1]
    num_class = len(list_author)
    net = build_network(archi = net_type, use_gpu=use_gpu, num_class=num_class)
    logger.info(net)
    
    net.load_state_dict(torch.load(args.weights), args.gpu)
    net.eval()

    test_image_dir = 'D:/SealProjectOLD/Datasets/images/val'

    ###
    sum_acc_average = 0
    sum_confiden_average = 0
    count_authors = len([name for name in os.listdir(test_image_dir) if os.path.isdir(os.path.join(test_image_dir, name))])
    
    dataset = datasets.ImageFolder(test_image_dir, transform= None)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    
    for subfolder in list_author:
        subfolder_path = os.path.join(test_image_dir, subfolder)
        count_items = len([name for name in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, name))])
        print(subfolder)
        sum_acc = 0
        sum_confiden = 0

        for img in os.listdir(subfolder_path):
            test_img_path = os.path.join(subfolder_path, img)
            i, confiden = predict(net, idx_to_class, test_img_path, subfolder) #, confiden
            sum_acc += i 
            sum_confiden += confiden
        ###
        result_acc = sum_acc/count_items
        result_confiden = sum_confiden/count_items
        ###
        sum_acc_average += result_acc
        sum_confiden_average += result_confiden
        ###
        print("True predicts/ Totals: {}/{} ({})".format(sum_acc, count_items, result_acc))
        print("The average confidence: {}".format(result_confiden))
    ###
    result_acc_average = sum_acc_average/count_authors
    result_confiden_average = sum_confiden_average/count_authors
    print("The average accuracy of testsets: {}".format(result_acc_average))
    print("The average confidence of testsets: {}".format(result_confiden_average))        