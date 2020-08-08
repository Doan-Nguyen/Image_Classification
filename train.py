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
from torch.autograd import Variable

# from conf import settings
from conf import global_settings
from utils_ai import build_network, get_training_dataloader, get_test_dataloader, WarmUpLR

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

def train(epoch):
    net.cuda()
    net.train()
    for batch_index, (images, labels) in enumerate(stamp_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
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

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]

def eval_training():
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in stamp_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    logger.info('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(stamp_test_loader.dataset),
        correct.float() / len(stamp_test_loader.dataset)
    ))

    return correct.float() / len(stamp_test_loader.dataset)

if __name__ == '__main__':
    
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
    net_type = args_dict['net']
    use_gpu = args_dict['gpu']
    standard_folder = global_settings.TRAIN_FOLDER
    list_author = next(os.walk(standard_folder))[1]
    print(list_author)
    num_classes = len(list_author)
    net = build_network(archi=net_type, use_gpu=use_gpu, num_classes=num_classes) 
    #data preprocessing:
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
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=global_settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(stamp_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = './checkpoint/'
    
    input_tensor = torch.Tensor(12, 3, 32, 32).cuda()

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, global_settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training()

        #start to save best performance model after learning rate decay to 0.01
        if best_acc < acc:
            torch.save(net.state_dict(), 
            checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            logger.info("Saving at epoch: " + str(epoch) + " with accuracy: " +  str(acc))

    # writer.close()