import logging
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
from datetime import datetime
import boto3 
import botocore 
import base64
import urllib

import flask
from flask import request, jsonify
from PIL import Image

from conf import settings
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
app.config["DEBUG"] = False

RECOGNITION_FOLDER = "./recognition_folder"
COMPARE_FOLDER = "./compare_folder"
BUCKET_NAME = "reldentify-seal"
ACCESS_ID = "AKIA5KBD4DULSIXHTOON"
ACCESS_KEY = "cTZ3OW0pySkeK52j7M4JEyKtEhH9KfDcdhHm7JxG"

if not os.path.isdir(RECOGNITION_FOLDER):
    os.mkdir(RECOGNITION_FOLDER)
if not os.path.isdir(COMPARE_FOLDER):
    os.mkdir(COMPARE_FOLDER)

@app.route('/compare_sign', methods=["POST"])
def compare_sign():
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
            ori_img = Image.open(ori_img_name)
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
                img_data = Image.open(img_name)
                list_img_content.append(img_data)
                list_keys.append(key)
            for i in range(len(list_img_content)):
                similarity_confidence = 1.0
                results[list_keys[i]] = str(similarity_confidence)
                logger.info(list_keys[i] + "_sim_" + str(similarity_confidence))
            return jsonify(results)
        logger.error("Request is not json")
        feed_back = {"message ": "This is not json format"}
        return jsonify(feed_back)
    logger.error("Request is not POST method")
    feed_back = {"message ": "This is not POST method"}
    return jsonify(feed_back)

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8888)
    app.run(host='127.0.0.1', port=8888)

