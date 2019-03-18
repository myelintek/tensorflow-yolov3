#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
#================================================================

import cv2
import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils


IMAGE_H, IMAGE_W = 416, 416
video_path = "/video_cutted.mp4"
#video_path = 0 # use camera
classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)
input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(),
                                                            "./checkpoint/yolov3_cpu_nms.pb",
                                                            ["Placeholder:0", "concat_9:0", "mul_6:0"])

if not os.path.isfile(video_path):
    raise ValueError("Video file does not exist")
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
#cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('video_with_boxes_1.avi',fourcc, 30.0, (416,416))

with tf.Session() as sess:
    vid = cv2.VideoCapture(video_path)
    while True:
        flag, frame = vid.read()
        if flag:
            print("Frame read")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = image.resize(size=(IMAGE_H, IMAGE_W))
            img_resized = np.array(image, dtype=np.float32)
            img_resized = img_resized / 255.
            prev_time = time.time()

            boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
            boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)
            image = utils.draw_boxes(image, boxes, scores, labels, classes, (IMAGE_H, IMAGE_W), show=False)
            #image.save("/yolo_from_video.jpg")
            #break
            curr_time = time.time()
            exec_time = curr_time - prev_time
            result = np.asarray(image)
            info = "time: %.2f ms" %(1000*exec_time)
            cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=2)
#            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
#            cv2.imshow("result", result)
            out.write(result)
           

        else:
            print("No frame")
            break
out.release()
vid.release()
