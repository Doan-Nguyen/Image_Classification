""" configurations for this project

author baiyu
"""
import os
from datetime import datetime


##          Directory to save weights file & datasets  
CHECKPOINT_PATH = './checkpoint'
DATASET_FOLDER_LAP = "/media/doannn/data/Projects/Works/SealProject/Datasets/data_from_internet/cifar100_imgs"
DATASET_FOLDER_COLAB = "/content/drive/My Drive/Data_Backup/Image_classification/Datasets/From_internet/cifar100_imgs"

# TRAIN_FOLDER = os.path.join(DATASET_FOLDER_LAP, "train")
# TEST_FOLDER = TRAIN_FOLDER = os.path.join(DATASET_FOLDER_LAP, "test")

TRAIN_FOLDER = os.path.join(DATASET_FOLDER_COLAB, "train")
TEST_FOLDER = TRAIN_FOLDER = os.path.join(DATASET_FOLDER_COLAB, "test")


###             Model parameters

#  mean and std of STAMP dataset
TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# 
NUMB_CLASSES = 100
#  Dataloader
batch_size = 48 # 32
num_workers = 4
#  total training epoches
EPOCH = 10
MILESTONES = [60, 120, 160]
#  time of we run the script
TIME_NOW = datetime.now().isoformat()
#  save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 2
# start epoch for resume training
START_EPOCH = 101

### Checkpoint path 
CHECKPOINT_TRANSFER = 'checkpoint/vgg16-84-best.pth'
#tensorboard log dir
LOG_DIR = 'runs'


