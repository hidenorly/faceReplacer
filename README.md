# faceReplacer

## setup for general

```
pip install numpy opencv-python
```

## setup for RetinaFace and mtcnn

```
pip install face-detection torch torchvision numpy opencv-python
pip install mtcnn tensorflow
```

## setup for haarcascade

https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

## setup for ssd

https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel
https://github.com/keyurr2/face-detection/blob/master/deploy.prototxt.txt

## setup for yolov3

git clone https://github.com/pjreddie/darknet.git
wget https://pjreddie.com/media/files/yolov3.weights


## help

```
usage: faceReplacer.py [-h] [-i INPUT] [-o OUTPUT] [-f ICON] [-d DETECTOR]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Specify input path
  -o OUTPUT, --output OUTPUT
                        Specify output path
  -f ICON, --icon ICON  Specify replace icon image
  -d DETECTOR, --detector DETECTOR
                        Specify detector. haars or ssd or yolo or retina or mtcnn
```