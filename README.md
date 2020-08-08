## Acces to server
sudo ssh -i "seal_key.pem" ubuntu@44.230.156.111

git clone http://Nguyen_Van_Thanh:53ki32ng30c1q@git.deha.vn:1080/Nguyen_Van_Thanh/SignatureAPI.git

http://git.deha.vn:1080 Nguyen_Van_Thanh 53ki32ng30c1q

## Requirements
- python3.6
- pytorch1.3.1
- cuda10.1
- cudnnv7
conda install -ya pillow=6.1 
pip install flask-cors imgaug

## This project have different docker for each task stamp vs sign
# For stamp 
cd SignatureAPI/Stamp

for local: 
sudo docker build -t stamp_docker_image .
for server: 
docker build -t stamp_docker_image .

for local
sudo docker run -it --privileged=true --name stamp_docker --shm-size 8G stamp_docker_image
sudo docker rm -f stamp_docker

for server
docker run -it --privileged=true --name stamp_docker --shm-size 32G stamp_docker_image
docker rm -f stamp_docker

docker update --restart unless-stopped stamp_docker_id 

# For sign READ NOTE first
cd SignatureAPI/Sign

for local: 
sudo docker build -t sign_docker_image .
for server: 
docker build -t sign_docker_image .

for local
sudo docker run -it --privileged=true --name sign_docker --shm-size 8G -p 8888:8888 sign_docker_image
sudo docker rm -f sign_docker

for server
docker run -it --privileged=true --name sign_docker --shm-size 32G -p 8888:8888 sign_docker_image
docker rm -f sign_docker

docker update --restart unless-stopped 2597ff7d33f3 

# Note: currently, my mistake for copy, so in real env for now for sign
for local: 
sudo docker build -t seal_docker_image .
for server: 
docker build -t seal_docker_image .

for local
sudo docker run -it --privileged=true --name seal_docker --shm-size 8G -p 8888:8888 seal_docker_image
sudo docker rm -f sign_docker

for server
docker run -it --privileged=true --name seal_docker --shm-size 32G -p 8888:8888 seal_docker_image
docker rm -f sign_docker

docker update --restart unless-stopped 2597ff7d33f3 


## Build docker 
for local: 
sudo docker build -t seal_docker_image .
for server: 
docker build -t seal_docker_image .

docker run -it --init \
  --name image_recognition\
  --runtime=nvidia \
  --ipc=host \
  -p 2347:2347\
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/workspace" \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  SignatureAPI/Sign python3 sign_api.py


docker update --restart unless-stopped 5686628847b5


sudo docker run -it  --name sign_docker --shm-size 8G -p 8888:8888 seal_docker_image

nvidia-docker run -it --name stamp_recognition --shm-size 32G  -p 8889:8889 -v /home/ubuntu:/workspace/share nvcr.io/nvidia/pytorch:19.08-py3

docker update --restart unless-stopped abb8136ba5df

## auto start cron service
sudo systemctl enable my-cron.service
sudo systemctl disable my-cron.service


## Usage

sudo docker run -it --runtime=nvidia --privileged=true --name sign_docker --shm-size 8G -p 8888:8888 seal_docker_image python sign_api.py


sudo docker rm -f sign_docker

### 2.Dataset & data augmentation
Seal Dataset from customer

#### 2.1 Data augmentation
+ Lib: Augmentor
  ```
  $ pip install Augmentor
  ```

+ Some techniques:

  - rotate

  - flip_left_right/left (0.8)

  - flip_top/bottom

  - angles, lightings, and miscellaneous distortions ...

+ **Note**: 
  - imbalanced data case -> the number of each author's images same. (suggests :600 images)

### 3. train the model
python train.py -net squeezenet 
train_accu not saved
accuracy 80% 

python train.py -net shallow_squeezenet -w 2
train_accu 88%
accuracy 82%

python train.py -net resnet18 -w 2

python train.py -net shallow_resnet18 -w 2


### 4. test model 
python test_sign.py -net shallow_squeezenet -weights ./checkpoint/results/shallow_squeezenet-51-best.pth 
I will update readme.

You need to specify the net you want to train using arg -net

```bash
$ python train.py -net vgg16
```

sometimes, you might want to use warmup training by set ```-warm``` to 1 or 2, to prevent network
diverge during early training phase.

The supported net args are:
squeezenet



### 5. test the model
Test the model using test.py
```bash
$ python test.py -net vgg16 -weights path_to_vgg16_weights_file
```

## Training Details
I didn't use any training tricks to improve accuray, if you want to learn more about training tricks,
please refer to my another [repo](https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks), contains
various common training tricks and their pytorch implementations.


I follow the hyperparameter settings in paper [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1701.06548v1), which is init lr = 0.1 divide by 5 at 60th, 120th, 160th epochs, train for 200
epochs with batchsize 128 and weight decay 5e-4, Nesterov momentum of 0.9. You could also use the hyperparameters from paper [Regularizing Neural Networks by Penalizing Confident Output Distributions](https://arxiv.org/abs/1701.06548) and [Random Erasing Data Augmentation](https://arxiv.org/abs/1701.06548), which is initial lr = 0.1, lr divied by 10 at 150th and 225th epochs, and training for 300 epochs with batchsize 128, this is more commonly used. You could decrese the batchsize to 64 or whatever suits you, if you dont have enough gpu memory.

You can choose whether to use TensorBoard to visualize your training procedure

## Results
The result I can get from a certain model, since I use the same hyperparameters to train all the networks, some networks might not get the best result from these hyperparameters, you could try yourself by finetuning the hyperparameters to get
better result.

|dataset|network|params|top1 err|top5 err|memory|epoch(lr = 0.1)|epoch(lr = 0.02)|epoch(lr = 0.004)|epoch(lr = 0.0008)|total epoch|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
|cifar100|mobilenet|3.3M|34.02|10.56|0.69GB|60|60|40|40|200|
|cifar100|mobilenetv2|2.36M|31.92|09.02|0.84GB|60|60|40|40|200|

