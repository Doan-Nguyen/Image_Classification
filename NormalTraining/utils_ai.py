""" helper function

author baiyu
"""
import logging
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import random 
import cv2 
import numpy as np
from numpy import dot
from numpy.linalg import norm
from PIL import Image

import torch
from torchvision import transforms
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import global_settings
from imgaug import augmenters as iaa

def build_network(args):
    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    ### custom network start here
    elif args.net == 'shallow_squeezenet':
        from models.shallow_squeezenet import shallow_squeezenet
        net = shallow_squeezenet()
    elif args.net == 'shallow_resnet18':
        from models.shallow_resnet import shallow_resnet18
        net = shallow_resnet18()
    ### custom network end here
    
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34 
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50 
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101 
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:
        net = net.cuda()

    return net

def get_training_dataloader(
            mean, 
            std, 
            train_folder= global_settings.TRAIN_FOLDER, 
            batch_size=global_settings.batch_size, 
            num_workers=global_settings.num_workers, 
            shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of stamp training dataset
        std: std of stamp training dataset
        path: path to stamp training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_path = train_folder
    stamp_training = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
    stamp_training_loader = torch.utils.data.DataLoader(
        stamp_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return stamp_training_loader

def get_test_dataloader(mean, 
                        std,
                        test_folder=global_settings.TEST_FOLDER, 
                        batch_size=global_settings.batch_size, 
                        num_workers=global_settings.num_workers, 
                        shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of stamp test dataset
        std: std of stamp test dataset
        path: path to stamp test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: stamp_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        # transforms.ToPILImage(),
	    transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_path = test_folder
    stamp_test = torchvision.datasets.ImageFolder(root=test_path, transform=transform_test)
    stamp_test_loader = DataLoader(
        stamp_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    idx_to_class = {v: k for k, v in stamp_test.class_to_idx.items()}

    return stamp_test_loader, idx_to_class

def compute_mean_std(stamp_dataset):
    """compute the mean and std of stamp dataset
    Args:
        stamp_training_dataset or stamp_test_dataset
        witch derived from class torch.utils.data
    
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([stamp_dataset[i][1][:, :, 0] for i in range(len(stamp_dataset))])
    data_g = np.dstack([stamp_dataset[i][1][:, :, 1] for i in range(len(stamp_dataset))])
    data_b = np.dstack([stamp_dataset[i][1][:, :, 2] for i in range(len(stamp_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    
def predict_author_single_img(model, idx_to_class=None, input_image=None, image_path=None):
    """
    """
    # device = torch.device("cpu")
    device = torch.device("gpu")

    image_transforms =  transforms.Compose([
                        transforms.Resize((112, 112)),
                        transforms.ToTensor(),
                        transforms.Normalize(settings.TRAIN_MEAN, 
                                             settings.TRAIN_STD)])

    if input_image is not None:
        img = input_image
    else:
        img = Image.open(image_path).convert('RGB')
    img_tensor = image_transforms(img)
    img_tensor = img_tensor.view(1, 3, 112, 112)
    with torch.no_grad():
        model.eval()
        # model ouputs log probabilities
        out = model(img_tensor)  # <class 'torch.Tensor'>  torch.Size([1, 58])
        ps = torch.exp(out) #  <class 'torch.Tensor'> torch.Size([1, 58])
        topk, topclass = ps.topk(3, dim=1)
        
        # sum_topk = int(topk.cpu().numpy()[0][0]) + int(topk.cpu().numpy()[0][1]) + int(topk.cpu().numpy()[0][2])
        sum_topk = int(topk.gpu().numpy()[0][0]) + int(topk.gpu().numpy()[0][1]) + int(topk.gpu().numpy()[0][2])
    # return idx_to_class[topclass.cpu().numpy()[0][0]], (topk.cpu().numpy()[0][0])/sum_topk
    return idx_to_class[topclass.gpu().numpy()[0][0]], (topk.gpu().numpy()[0][0])/sum_topk

def get_feature_single_img(net, input_image=None, image_path=None):
    # device = torch.device("cpu")
    device = torch.device("gpu")
    image_transforms =  transforms.Compose([
                        transforms.Resize((112, 112)),
                        transforms.ToTensor(),
                        transforms.Normalize(settings.TRAIN_MEAN, 
                                             settings.TRAIN_STD)])

    if input_image is not None:
        img = input_image
    else:
        img = Image.open(image_path).convert('RGB')
    img_tensor = image_transforms(img)
    model = net
    img_tensor = img_tensor.view(1, 3, 112, 112)
    with torch.no_grad():
        model.eval()
        # model ouputs log probabilities
        out = model(img_tensor)  # <class 'torch.Tensor'>  torch.Size([1, 58])
        ps = torch.exp(out) #  <class 'torch.Tensor'> torch.Size([1, 58])
        # feature = ps.cpu().numpy()[0]
        feature = ps.gpu().numpy()[0]
    return feature
   
def compare_similarity(net, image_1, image_2):
    feature_1 = get_feature_single_img(net, image_1)
    feature_2 = get_feature_single_img(net, image_2)
    cos_sim = dot(feature_1, feature_2)/(norm(feature_1)*norm(feature_2))
    return cos_sim


def create_training_data_for_new_author(author_name, logger):
    logger.info("Create dataset for author: ", author_name)
    standard_folder = "../datasets/Stamp_Recognition/Original_Datasets"
    result_folder = "../datasets/Stamp_Recognition/To_Use_Datasets/train"
    src_author_folder = os.path.join(standard_folder, author_name)
    dst_author_folder = os.path.join(result_folder, author_name)
    list_img=next(os.walk(src_author_folder))[2]
    if len(list_img) == 0:
        logger.error("This source contain no image. Exit!")
        sys.exit()
    if not os.path.isdir(dst_author_folder):
        os.mkdir(dst_author_folder)
    # https://colab.research.google.com/drive/1Ru7ghfYNsQ3T0l6NZT8cdY62ttliC6hH?authuser=1#scrollTo=bdGRjWUaQrCg
    dict_augmentation = {
        "crop_0_16_gauss_3": iaa.Sequential([
                iaa.Crop(px=(0, 16)), 
                iaa.GaussianBlur(sigma=(0, 3.0))]),
        "contrast_normalize_04_20":iaa.Sequential([
                iaa.ContrastNormalization((0.4, 2.0))]),
        "AdditiveGaussianNoise_001_007":iaa.Sequential([
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.01*255, 0.07*255), per_channel=0.5)]),
        "Multiply_01_11_03":iaa.Sequential([
            iaa.Multiply((0.7, 1.1), per_channel=0.3),]),
        "Affine_shear_y":iaa.Sequential([
            iaa.Affine(
                scale={"x": (1, 1), "y": (1, 1)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                shear=(0, 8))]),
        "Dropout_003_015":iaa.Sequential([
            iaa.Dropout((0.03, 0.15), per_channel=0.5),]),
        "Invert_02":iaa.Sequential([
            iaa.Invert(0.2, per_channel=True)]), 
        "ElasticTransformation_05_25_05":iaa.Sequential([
            iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.5)]),
        "PiecewiseAffine_004_008":iaa.Sequential([
            iaa.PiecewiseAffine(scale=(0.04, 0.08))]),
        "crop_5_18":iaa.Sequential([
            iaa.Crop(px=(5, 18)),])
    }
    for img_name in list_img:
        img_path = os.path.join(src_author_folder, img_name)
        for method_key in dict_augmentation.keys():
            method = dict_augmentation[method_key]
            _img = cv2.imread(img_path)
            img = list()
            img.append(_img)
            images_aug = method(images=img)
            for img in images_aug:
                new_name = img_name[:-4] + str(method_key) + ".jpg"
                new_path = os.path.join(dst_author_folder, "train", author_name, new_name)
                new_img = images_aug[0]
                cv2.imwrite(new_path, new_img)
    list_result_img = next(os.walk(dst_author_folder))[2]
    if len(list_result_img) == 0:
        logger.error("Error, Created 0 images, recheck please")
        sys.exit()
    logger.info("Created " + len(list_result_img) + " images of author: " + author_name)
    if len(list_result_img) > 500:
        num_to_remove = len(list_result_img) - 500
        logger.info("Truncated " + num_to_remove + " images of author: " + author_name)
        list_keep = random.choice(list_result_img, k = 500)
        list_remove_img = list(set(list_result_img).difference(list_keep))
        for img_name in list_remove_img:
            img_path = os.path.join(dst_author_folder, img_name)
            os.remove(img_path)

