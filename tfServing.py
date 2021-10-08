import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf 
import numpy as np
import PIL.Image as Image
import io

input_size = 608

## TFServing JSON > Response JSON 반환
def meter_predict(json_predict, im_shape):

    result_json = {
        "response_code" : "",
        "device_id" : "",
        "predict_num" : [],
        "bbox" : []
    }

    boxes = []
    pred_conf = []

    ## 박스 좌표, confidence 스코어 반환
    for i in json_predict['predictions'][0]:
        boxes.append(i[:4])
        pred_conf.append(i[4:])

    boxes = tf.convert_to_tensor(tf.constant(boxes), dtype = tf.float32)
    boxes = tf.expand_dims(boxes, 0)

    pred_conf = tf.convert_to_tensor(tf.constant(pred_conf), dtype = tf.float32)
    pred_conf = tf.expand_dims(pred_conf, 0)

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.1
    )

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    out_boxes, out_scores, out_classes, num_boxes = pred_bbox
    image_h, image_w = im_shape

    data = []

    ## 원본 이미지사이즈로 박스 크기 변환
    for i in range(num_boxes[0]):

        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h) # y1
        coor[2] = int(coor[2] * image_h) # y2
        coor[1] = int(coor[1] * image_w) # x1
        coor[3] = int(coor[3] * image_w) # x2

        width = coor[3] - coor[1]
        height = coor[2] - coor[0]
        data.append([coor[1], coor[0], width, height, out_scores[0][i], out_classes[0][i]])

    data = sorted(data, key = lambda x : x[0], reverse = False)

    for seq, d in enumerate(data):

        if seq == 5:
            break

        else:

            result_json['predict_num'].append({
                'num_seq' : str(seq),
                'predict' : str(d[5]),
                'confidence' : str(d[4])
                })

            result_json['bbox'].append({
                'num_seq' : str(i),
                'x' : str(d[0]),
                'y' : str(d[1]),
                'width' : str(d[2]),
                'height' : str(d[3])
                })

    return result_json