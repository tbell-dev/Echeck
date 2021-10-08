# -*- coding: utf-8 -*-
from ctypes import *
import cv2, json, os
import json
from datetime import datetime

import flask
from flask import Flask, request
import requests

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import darknet
import base64
from PIL import Image
import io
import numpy as np
from PIL import ExifTags
from tfServing import meter_predict as meter_pred
from configparser import ConfigParser

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)

def darknet_inference(image):
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    img_for_detect = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(img_for_detect, image_resized.tobytes())

    detections = darknet.detect_image(network, class_names, img_for_detect, thresh= .8)

    crop_batch = []
    predict_list = []
    conf = []
    for label, confidence, bbox in detections:
        _, _, w, h = bbox
        x, y, _, _ = darknet.bbox2points(bbox)

        crop_image = image_resized[y:y+int(h), x:x+int(w)]
        crop_image = cv2.resize(crop_image, (100, 100)) / 255.
        crop_batch.append(crop_image)
        conf.append(confidence)

    image_arr = np.array(crop_batch)

    if not len(image_arr) == 0:
        predict = LeNet_inference(image_arr)

        for idx, num in enumerate(predict):
            predict_list.append([x, y, w, h, num, conf[idx]])

        darknet.free_image(img_for_detect)
    else:
        predict = []

    return predict_list

def LeNet_load_model(model):
    global seg_model
    seg_model = tf.keras.models.load_model(model)
    seg_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics = ['accuracy'])

def LeNet_inference(image_arr):
    predict = seg_model.predict(image_arr)
    result = [str(np.argmax(x)) for x in predict]
    return result

# Darknet + LeNet inference > Response JSON
def predict2json(predict):

    result_json = {
        "response_code" : "",
        "device_id" : "",
        "predict_num" : [],
        "bbox" : []
    }

    if not len(predict) == 0:
        # [x, y, w, h, num, conf]
        for idx, pred in enumerate(predict):
            if idx == 5:
                break
            else:
                result_json['predict_num'].append({
                    'num_seq' : str(idx),
                    'predict' : str(pred[4]),
                    'confidence' : str(pred[5])
                })

                result_json['bbox'].append({
                    'num_seq' : str(idx),
                    'x' : str(pred[0]),
                    'y' : str(pred[1]),
                    'w' : str(pred[2]),
                    'h' : str(pred[3])
                })
    else:
        pass

    return result_json

def meter_preprocess(images):

    # images = images.convert("rgb")
    image_array = tf.keras.preprocessing.image.img_to_array(images)
    image_resize = tf.image.resize(image_array, [608, 608])
    image_resize = image_resize / 255.
    images_batch = []

    for i in range(1):
        images_batch.append(image_resize)

    imagesArray = np.asarray(images_batch).astype(np.float32)

    return imagesArray, (images.width, images.height)

# Flask
app = Flask(__name__)
@app.route('/meter/ocr', methods=['POST'])
def main():
    if request.method == 'POST':

        # json 이미지 읽기
        im_b64 = request.json['imBytearray']
        img_bytes = base64.b64decode(im_b64.encode('utf-8'))
        input_image = Image.open(io.BytesIO(img_bytes))

        ########## 이미지 회전 #######################################
        if "exif" in input_image.info:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation]=='Orientation':
                    break
            exif = dict(input_image._getexif().items())
            # print(exif[orientation])
            if exif[orientation] == 3:
                input_image = input_image.rotate(180, expand = True)
            elif exif[orientation] == 6:
                input_image = input_image.rotate(0, expand = True)
            elif exif[orientation] == 8:
                input_image = input_image.rotate(90, expand = True)

        elif input_image.width > input_image.height:
            input_image = input_image.rotate(90, expand=True)
        ###########################################################

        ############### 기계식 ######################
        if request.json['meter_id'] == '1':
            imgArray, im_shape = meter_preprocess(input_image)
            data = json.dumps({
                'instances' : imgArray.tolist()
            })

            # tfserving
            meter_predict = requests.post(meter_URL, data = data.encode('utf-8'))
            meter_predict = json.loads(meter_predict.text)

            if not len(meter_predict['predictions'][0]) == 0:
                result = meter_pred(meter_predict, im_shape)
            else:
                result = {
                    "response_code" : "",
                    "device_id" : request.json['device_id'],
                    "predict_num" : [],
                    "bbox" : []
                }

        ############### 전자식 ######################
        elif request.json['meter_id'] == '2':
            
            predict = darknet_inference(input_image) # [x, y, w, h, num, conf]
            result = predict2json(predict)
            result['device_id'] = request.json['device_id']

        # Response JSON 처리, 인식 숫자 길이 처리
        model_predict = result['predict_num']

        # 이미지 저장 파일명
        save_filename = '_'.join([
                        result['device_id'],  # 디바이스명
                        ''.join([x['predict'].split('.')[0] for x in result['predict_num']]), # 예측결과
                        datetime.now().strftime('%Y%m%d%H%M%S') # 년월일시분초
                        ])

        input_image.save(os.path.join(API_Save_dir_path, save_filename + '.jpg'), 'JPEG')

        # 기계식
        if request.json['meter_id'] == '1' and len(model_predict) == 4:
            result['response_code'] = '0000'

        # 전자식
        elif request.json['meter_id'] == '2' and len(model_predict) == 5:
            result['response_code'] = '0000'

        else:
            result['response_code'] = '9999'

        result['device_id'] = request.json['device_id'] #
        result['request_time'] = request.json['request_time'] #

        print(save_filename.split('_'))

        return flask.jsonify(result)

if __name__== '__main__':

    # 설정 파일 parser
    parser = ConfigParser()
    parser.read('config.ini')

    # tfServing
    meter_URL = parser.get('Tf-Serving', 'meter')

    # API 수신 저장
    API_Save_dir = datetime.now().strftime('%Y%m%d')
    API_Save_path = parser.get('API_Save', 'path')
    API_Save_dir_path = os.path.join(API_Save_path, API_Save_dir)

    if not os.path.exists(API_Save_dir_path):
        os.mkdir(API_Save_dir_path)
    else:
        pass

    # Model
    class_model = parser.get('Classifier', 'model')
    LeNet_load_model(class_model)

    network, class_names, class_colors = darknet.load_network(
        parser.get('Detector', 'conf'),
        parser.get('Detector', 'data'),
        parser.get('Detector', 'weight'),
        batch_size=1
    )

    width = darknet.network_width(network)
    height = darknet.network_height(network)

    app.run(debug=False, 
            host = parser.get('Flask', 'host'), 
            port = parser.getint('Flask', 'port')
            )