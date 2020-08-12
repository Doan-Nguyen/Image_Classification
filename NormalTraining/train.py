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
import models


### Logging 
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)

logging.info('Start training process')
handler = logging.FileHandler('train_log.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

def training_loop(epoch):
    # net.cuda()
    # net.train()
    train_running_loss = 0.0
    train_running_correct = 0.0
    for batch_index, (images, labels) in enumerate(stamp_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()
        #
        images, labels = Variable(images), Variable(labels)
        images, labels = images.cuda(), labels.cuda()
        # 
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        train_running_loss += loss.item()
        _, preds = outputs.max(1)
        train_running_correct += (preds == labels).sum()

        loss.backward()
        optimizer.step()

        logger.info('Training Epoch: \
            {epoch} [{trained_samples}/{total_samples}]\t \
            Loss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(stamp_training_loader.dataset)
        ))
    train_loss = train_running_correct/len(stamp_training_loader.dataset)
    train_accuracy = 100.*train_running_correct/len(stamp_training_loader.dataset)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]

    return train_loss, train_accuracy

def eval_training():
    net.eval()

    val_running_loss = 0.0
    val_running_correct = 0.0

    for (images, labels) in stamp_test_loader:
        images, labels = Variable(images), Variable(labels)
        images, labels = images.cuda(), labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        val_running_loss += loss.item()
        _, preds = outputs.max(1)
        val_running_correct += (preds == labels).sum()

    logger.info('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        val_running_loss / len(stamp_test_loader.dataset),
        val_running_correct.float() / len(stamp_test_loader.dataset)
    ))
    val_loss = val_running_correct/len(stamp_training_loader.dataset)
    val_accuracy = 100.*val_running_correct/len(stamp_training_loader.dataset)

    return val_loss, val_accuracy

def retrain(model_ft, optimizer):
    ###         Initiliaze the model
    net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    ###         Load checkpoint 
    checkpoint = torch.load(global_settings.CHECKPOINT_TRANSFER)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', default = 'vgg16', type=str, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-r', type=bool, default=False, help='retrain')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=48, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=2, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    ###         Datasets loader
    stamp_training_loader = get_training_dataloader(
        global_settings.TRAIN_MEAN,
        global_settings.TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    stamp_test_loader, idx_to_class = get_test_dataloader(
        global_settings.TRAIN_MEAN,
        global_settings.TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    ###         Initialize the model  
    net_type = args_dict['net']
    use_gpu = args_dict['gpu']
    standard_folder = global_settings.TRAIN_FOLDER
    list_author = next(os.walk(standard_folder))[1]
    num_classes = len(list_author)
    net = build_network(archi=net_type, use_gpu=use_gpu, num_classes=num_classes)
    net.cuda() 
    
    ###         Optimizer & compute loss function
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    #   Log infor modularity to Pytorch models & optimizer
    for param_tensor in net.state_dict():       
        logger.info("Param tensor: {param_tensor} \t {net.state_dict()[param_tensor].size()}")
    for var_name in optimizer.state_dict():
        logger.info("{var_name} \t {optimizer.state_dict()[var_name]}")
    #   
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=global_settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(stamp_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = global_settings.CHECKPOINT_PATH
    
    ###         Create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    ###         Training 
    best_acc = 0.0
    train_loss, train_accuracy = [], []
    for epoch in range(1, global_settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train_epoch_loss, train_epoch_accuracy =  training_loop(epoch)
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss, val_accuracy = eval_training()

        #start to save best performance model after learning rate decay to 0.01
        if best_acc < val_accuracy:
            # torch.save(net.state_dict(), 
            #     checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            torch.save({
                'epoch':global_settings.EPOCH,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_function
            }, checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            logger.info("Saving at epoch: " + str(epoch) + " with accuracy: " +  str(acc))

    