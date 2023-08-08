#!/usr/bin/env python3
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
from faceDetector import HaarsFaceDetector
from faceDetector import SsdFaceDetector
from faceDetector import YoloFaceDetector
from faceDetector import RetinaFaceDetector
from faceDetector import MtCnnFaceDetector
from faceDetector import RetinaFaceDetector


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

  def resizeImage(image, targetWidth, targetHeight):
    result = image

    height, width = image.shape[:2]
    if targetWidth!=width and targetHeight!=height:
      result = cv2.resize(image, (targetWidth, targetHeight))

    return image



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