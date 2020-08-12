import logging
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
from datetime import datetime
import boto3
import botocore
import cv2
import base64
import urllib

import flask
from flask import request, jsonify
from PIL import Image
from flask_cors import CORS

import torch
from torchvision import transforms
from torchvision import datasets

from conf import settings
from utils_ai import build_network, compare_similarity, predict_author_single_img, create_training_data_for_new_author
from utils_common import get_name_from_time, get_url

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

app = flask.Flask(__name__)
CORS(app)

app.config["DEBUG"] = False

RECOGNITION_FOLDER = "./recognition_folder"
COMPARE_FOLDER = "./compare_folder"
EXAMPLE_FOLDER = "./example_dataset_structure"
BUCKET_NAME = "reldentify-seal-production"
ACCESS_ID = "AKIA5KBD4DULSIXHTOON"
ACCESS_KEY = "cTZ3OW0pySkeK52j7M4JEyKtEhH9KfDcdhHm7JxG"

if not os.path.isdir(RECOGNITION_FOLDER):
    os.mkdir(RECOGNITION_FOLDER)
if not os.path.isdir(COMPARE_FOLDER):
    os.mkdir(COMPARE_FOLDER)

@app.route('/stamp_author_recognition', methods=["POST"])
def stamp_author_recognition():
    global net
    global idx_to_class
    global logger
    logger.info("request addr: " + request.remote_addr)
    if request.method == "POST":
        if request.is_json:
            current_folder = os.path.join(RECOGNITION_FOLDER, str(request.remote_addr))
            if not os.path.isdir(current_folder):
                os.mkdir(current_folder)
            req = request.get_json()
            list_img_content = list()
            list_keys = list()
            results = list()
            for key in req.keys():
                b64_string = req[key]
                img_data = base64.b64decode(b64_string)
                list_img_content.append(img_data)
                list_keys.append(key)
            for i in range(len(list_img_content)):
                img_path = get_name_from_time(current_folder)
                with open(img_path, 'wb') as f:
                    f.write(list_img_content[i])
                author, confidence = predict_author_single_img(net, idx_to_class=idx_to_class, image_path=img_path)
                result = str(author) + "_" + str(confidence)
                results.append(result)
            result_dict = dict()
            for i in range(len(list_keys)):
                result_dict[list_keys[i]] = str(results[i])
                logger.info(results[i])
            return jsonify(result_dict)
        else:
            logger.error("Request is not json")
            feed_back = {"message ": "This is not json format"}
            return jsonify(feed_back)
    else:
        logger.error("Request is not POST method")
        feed_back = {"message ": "This is not POST method"}
        return jsonify(feed_back)

@app.route('/compare_stamp', methods=["POST"])
def compare_stamp():
    global net
    global idx_to_class
    global logger
    s3 = boto3.resource('s3', aws_access_key_id=ACCESS_ID,
                        aws_secret_access_key= ACCESS_KEY)
    logger.info("request addr: " + request.remote_addr)
    if request.method == "POST":
        if request.is_json:
            current_folder = os.path.join(COMPARE_FOLDER, str(request.remote_addr))
            if not os.path.isdir(current_folder):
                os.mkdir(current_folder)
            req = request.get_json()
            list_img_content = list()
            list_keys = list()
            results = dict()
            ori_img_url = get_url(req["ori_url"])

            # for real
            ori_img_name = get_name_from_time(current_folder)
            try:
                s3.Bucket(BUCKET_NAME).download_file(ori_img_url, ori_img_name)
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    logger.error("The object does not exist.")
                else:
                    raise
            # for local
            # ori_img_name = ori_img_url
            ori_img = Image.open(ori_img_name).convert('RGB')
            for key in req.keys():
                if key == "ori_url":
                    continue
                img_url = get_url(req[key])
                img_name = get_name_from_time(current_folder)
                # for real
                try:
                    s3.Bucket(BUCKET_NAME).download_file(img_url, img_name)
                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == "404":
                        logger.error("The object does not exist.")
                    else:
                        raise
                # for local
                # img_name = img_url
                img_data = Image.open(img_name).convert('RGB')
                list_img_content.append(img_data)
                list_keys.append(key)
            for i in range(len(list_img_content)):
                similarity_confidence = compare_similarity(net, ori_img, list_img_content[i])
                results[list_keys[i]] = str(similarity_confidence)
                logger.info(list_keys[i] + "_sim_" + str(similarity_confidence))
            return jsonify(results)
        logger.error("Request is not json")
        feed_back = {"message ": "This is not json format"}
        return jsonify(feed_back)
    logger.error("Request is not POST method")
    feed_back = {"message ": "This is not POST method"}
    return jsonify(feed_back)

@app.route('/update_author', methods=["POST"])
def update_author():
    logger.info("request addr: " + request.remote_addr)
    if request.method == "POST":
        if request.is_json:
            req = request.get_json()
            list_img_content = list()
            list_keys = list()
            results = dict()
            author_name = req["author_name"]
            author_folder = os.path.join(EXAMPLE_FOLDER, author_name)
            if not os.path.isdir(author_folder):
                os.mkdir(author_folder)
                logger.info("Update new author: " + author_name)
            # for real
            # ori_img_name = get_name_from_time(current_folder)
            # urllib.request.urlretrieve(ori_img_url, ori_img_name)
            # for local
            ori_img_name = ori_img_url
            ori_img = Image.open(ori_img_name)
            for key in req.keys():
                if key == "author_name":
                    continue
                img_url = req[key]
                img_name = get_name_from_time(author_folder)
                # for real
                # urllib.request.urlretrieve(img_url, img_name)
                # for local
                img_name = img_url
                list_keys.append(key)
                logger.info("write: " + len(list_keys) + " images")
                logger.info("Finish writinge image")
            create_training_data_for_new_author(author_name, looger)
            os.system("python train.py -net shallow_squeezenet -w 2")
            return jsonify(results)
        logger.error("Request is not json")
        feed_back = {"message ": "This is not json format"}
        return jsonify(feed_back)
    logger.error("Request is not POST method")
    feed_back = {"message ": "This is not POST method"}
    return jsonify(feed_back)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default= 'squeezenet', help='net type')
    parser.add_argument('-weights', type=str, default='./checkpoint/results/squeezenet-139-best.pth', help='the weights file path you want to test')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')

    args_dict = vars(parser.parse_args())
    logger.info(args_dict)

    net_type = args_dict['net']
    # device = torch.device("cpu")
    device = torch.device("gpu")
    list_author = next(os.walk(EXAMPLE_FOLDER))[1] # 53
    num_classes = len(list_author)
    net = build_network(archi = net_type, use_gpu=False, num_classes=num_classes)
    logger.info(net)

    # net.load_state_dict(torch.load(args_dict['weights'], map_location=torch.device('cpu')))
    net.load_state_dict(torch.load(args_dict['weights'], map_location=torch.device('gpu')))
    net.eval()

    example_image_dir = './example_dataset_structure'
    dataset = datasets.ImageFolder(example_image_dir, transform= None)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    app.run(host='0.0.0.0', port=8889)