#!/usr/bin/python3
# coding: utf-8
#
# Copyright (C) 2023 hidenorly
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import cv2
import os
import numpy as np
import face_detection
from mtcnn import MTCNN

class FaceDetector:
  def __init__(self):
    print("setup")

  def detectFace(self, image):
    faces = [] # [x,y,width,height]
    return faces

  def getEnsureDetectionResult(self, left, top, bbox_width, bbox_height, width, height):
    if left<0:
      left=0
    if left>width:
      left=width
    if top<0:
      top=0
    if top>height:
      top=height
    if (left+bbox_width)>width:
      bbox_width = width - left
    if (top+bbox_height)>height:
      bbox_height = height - top

    return [left, top, bbox_width, bbox_height]

  def resizeImage(self, image, minWidth=1080, minHeight=1080, maxWidth=1920, maxHeight=1080):
    result = image

    height, width = image.shape[:2]
    targetWidth=None
    targetHeight=None

    if width<minWidth and height<maxWidth:
      targetWidth = minWidth
      targetHeight = maxWidth
    elif width>height:
      if width>maxWidth:
        targetWidth=maxWidth
        targetHeight=maxWidth*height/width
      elif height>maxHeight:
        targetHeight=maxHeight
        targetWidth=maxHeight*width/height

    if targetWidth and targetHeight:
      result = cv2.resize(image, (targetWidth, targetHeight))

    return image

class HaarsFaceDetector(FaceDetector):
  def __init__(self):
    self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

  def detectFace(self, image):
    faces = self.detector.detectMultiScale( image, scaleFactor=1.3, minNeighbors=5 )
    return faces


class SsdFaceDetector(FaceDetector):
  PROTOTXT_PATH = "deploy.prototxt"
  MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
  CONF_THRESHOLD = 0.8  # 信頼度の閾値

  def __init__(self):
    self.net = cv2.dnn.readNetFromCaffe(self.PROTOTXT_PATH, self.MODEL_PATH)

  def detectFace(self, image):
    faces = []
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    self.net.setInput(blob)
    detections = self.net.forward()

    height, width = image.shape[:2]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > self.CONF_THRESHOLD:
            bbox = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            left, top, right, bottom = bbox.astype(int)

            faces.append( self.getEnsureDetectionResult(left, top, right, bottom, width, height) )

    return faces


class YoloFaceDetector(FaceDetector):
  MODEL_PATH = "yolov3.weights"
  CONFIG_PATH = "../darknet/cfg/yolov3.cfg"
  CONF_THRESHOLD = 0.8  # 信頼度の閾値

  def __init__(self):
    self.net = cv2.dnn.readNetFromDarknet(self.CONFIG_PATH, self.MODEL_PATH)

  def detectFace(self, image):
    faces = []

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    self.net.setInput(blob)
    output_layers = self.net.getUnconnectedOutLayersNames()
    layer_outputs = self.net.forward(output_layers)

    height, width = image.shape[:2]

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > self.CONF_THRESHOLD and class_id == 0:  # class_id==0 means face
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                bbox_width = int(detection[2] * width)
                bbox_height = int(detection[3] * height)
                left = int(center_x - bbox_width / 2)
                top = int(center_y - bbox_height / 2)

                faces.append( self.getEnsureDetectionResult(left, top, bbox_width, bbox_height, width, height) )

    return faces


class RetinaFaceDetector(FaceDetector):
  MODEL = "RetinaNetResNet50"
  NMS_IOU_THRESHOLD = 0.3
  CONF_THRESHOLD = 0.1  # 信頼度の閾値

  def __init__(self):
    self.net = face_detection.build_detector(self.MODEL, confidence_threshold=self.CONF_THRESHOLD, nms_iou_threshold=self.NMS_IOU_THRESHOLD)

  def detectFace(self, image):
    faces = []

    inputImage = self.resizeImage(image, 128, 128, 1920, 1080)
    height, width = inputImage.shape[:2]

    detections = self.net.detect(inputImage)
    for detection in detections:
        x, y, w, h = detection[:4]
        faces.append( self.getEnsureDetectionResult( x, y, w, h, width, height) )

    return faces


class MtCnnFaceDetector(FaceDetector):
  def __init__(self):
    self.detector = detector = MTCNN()

  def detectFace(self, image):
    faces = []

    height, width = image.shape[:2]
    results = self.detector.detect_faces(image)

    for result in results:
        x, y, w, h = result['box']
        faces.append( self.getEnsureDetectionResult( x, y, w, h, width, height) )

    return faces


class GfxUtil:
  def bitbltWithAlpha(dstImage, srcImage, x, y):
    height, width = srcImage.shape[:2]
    alpha = srcImage[:, :, 3] / 255.0
    foreground = srcImage[:, :, :3]

    srcRegion = alpha[:, :, np.newaxis] * foreground
    bgRegion = (1 - alpha[:, :, np.newaxis]) * dstImage[y:y + height, x:x + width]

    blended = (srcRegion + bgRegion).astype(np.uint8)

    dstImage[y:y + height, x:x + width] = blended

    return dstImage


def replaceFace(detector, inputPath, outputPath, icon):
  image = cv2.imread(inputPath)

  faces = detector.detectFace(image)

  for (x, y, w, h) in faces:
    #print("x="+str(x)+",y="+str(y)+",w="+str(w)+",h="+str(h))
    resizedIcon = cv2.resize(icon, (w, h))
    GfxUtil.bitbltWithAlpha(image, resizedIcon, x, y)

  cv2.imwrite(outputPath, image)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", help="Specify input path")
  parser.add_argument("-o", "--output", help="Specify output path")
  parser.add_argument("-f", "--icon", help="Specify replace icon image")
  parser.add_argument("-d", "--detector", default="haars", help="Specify detector. haars or ssd or yolo or retina or mtcnn")
  args = parser.parse_args()

  icon = cv2.imread(args.icon, cv2.IMREAD_UNCHANGED)

  detector = HaarsFaceDetector
  if args.detector == "ssd":
    detector = SsdFaceDetector
  elif args.detector == "yolo":
    detector = YoloFaceDetector
  elif args.detector == "retina":
    detector = RetinaFaceDetector
  elif args.detector == "mtcnn":
    detector = MtCnnFaceDetector

  detector = detector()

  files = []
  for file in os.listdir(args.input):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
      files.append(file)

  for file in files:
    replaceFace( detector, os.path.join(args.input, file), os.path.join(args.output, file), icon )