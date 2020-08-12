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