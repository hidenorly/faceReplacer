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

def replaceFace(inputPath, outputPath, icon):
  image = cv2.imread(inputPath)

  detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
  faces = detector.detectMultiScale( image, 1.3, 5 )

  for (x, y, w, h) in faces:
    resizedIcon = cv2.resize(icon, (w, h))
    alpha = resizedIcon[:, :, 3] / 255.0
    foreground = resizedIcon[:, :, :3]

    iconRegion = alpha[:, :, np.newaxis] * foreground
    bgRegion = (1 - alpha[:, :, np.newaxis]) * image[y:y + h, x:x + w]

    blended = (iconRegion + bgRegion).astype(np.uint8)

    image[y:y + h, x:x + w] = blended

  cv2.imwrite(outputPath, image)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", help="Specify input path")
  parser.add_argument("-o", "--output", help="Specify output path")
  parser.add_argument("-f", "--icon", help="Specift replace icon image")
  args = parser.parse_args()

  icon = cv2.imread(args.icon, cv2.IMREAD_UNCHANGED)

  files = []
  for file in os.listdir(args.input):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
      files.append(file)

  for file in files:
    replaceFace( os.path.join(args.input, file), os.path.join(args.output, file), icon )