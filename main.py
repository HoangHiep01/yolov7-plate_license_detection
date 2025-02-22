import os
import sys

import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from keras.models import load_model
from sklearn.cluster import KMeans
from joblib import load

from tkinter import *
from tkinter import filedialog
import tkinter as tk
from PIL import ImageTk, Image

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def crop_sample(img, list_point, sample_size=(30, 60)):
  sample = []
  for point in list_point:
    area = cv2.resize(img[point[1]:point[3], point[0]:point[2]],sample_size, interpolation=cv2.INTER_AREA)
    exchange_range = cv2.bitwise_not(area)
    # cv2.imshow("Exchange image",area)
    # print(point, ((point[3]-point[1]) / (point[2] - point[0])))
    # cv2.waitKey(0)
    sample.append(exchange_range)
  return sample

def reoder_sample(list_point, list_center_y):
  thres = sum(list_center_y) // len(list_center_y)
  list_top_row = []
  list_down_row = []

  for point in list_point:
    if (point[1] + point[3]) // 2 < thres:
      list_top_row.append(point)
    else:
      list_down_row.append(point)

  list_point = sorted(list_top_row, key=lambda x:x[0]) + sorted(list_down_row, key=lambda x:x[0])
  return list_point

def gray_img_binarization(img, K = 5):

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  X = gray.reshape((gray.shape[0]*gray.shape[1], 1))
  kmeans = KMeans(n_clusters=K).fit(X)
  label = kmeans.predict(X)

  idx = np.argmax(kmeans.cluster_centers_[:])
  ret = int(min(X[label == idx]))
  print(ret)
  ret, thresh = cv2.threshold(gray, ret, 255, cv2.THRESH_BINARY)

  return thresh

def filter_img_and_binarization(img):

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # blur = cv2.GaussianBlur(gray, ksize=(3,3), sigmaX=cv2.BORDER_DEFAULT)
  bilateral = cv2.bilateralFilter(gray, 10, 35, 25)
  ret, thres = cv2.threshold(bilateral, 185, 255, cv2.THRESH_BINARY)
  return thres

def show_img_seg(list_img):
  print(len(list_img), )
  _, axes = plt.subplots(nrows=1, ncols=len(list_img), figsize=(25, 10))
  for ax, image, label in zip(axes, list_img, range(len(list_img))):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"K = {label}")
  print("end")

def image_processing(img, sample_size=(60, 30)):

  # thres = filter_img_and_binarization(img)
  thres = gray_img_binarization(img, K = 3)
  contours, hierarchies = cv2.findContours(thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  # cv2.imshow("Thershold",thres)
  # sample = []
  list_point = []
  list_center_y = []
  for ct in contours:

    min_x, min_y, max_x, max_y = thres.shape[1], thres.shape[0], 0, 0
    for point in ct:
      min_x = min(min_x, point[0][0])
      min_y = min(min_y, point[0][1])
      max_x = max(max_x, point[0][0])
      max_y = max(max_y, point[0][1])
    
    acspect = ((max_x - min_x) * (max_y - min_y)) / (thres.shape[1] * thres.shape[0])

    if acspect >= 0.015 and acspect <= 0.07 and (max_x - min_x) < (max_y - min_y) and ((max_y-min_y) / (max_x - min_x)) <= 4.5:
      list_point.append((min_x, min_y, max_x, max_y))
      list_center_y.append((min_y + max_y) // 2)

  reoder_list = reoder_sample(list_point, list_center_y)
  return crop_sample(thres, reoder_list, sample_size)

def classification(sample):
  
  label_dict_ex = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 
                   7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 
                   14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'K', 19: 'L', 
                   20: 'M', 21: 'N', 22: 'P', 23: 'R', 24: 'S', 25: 'T', 
                   26: 'U', 27: 'V', 28: 'X', 29: 'Y', 30: 'Z'}
  # label_dict_ex = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
  #             13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
  #             25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

  model = load_model("model/15_epochs_model2.h5")
  
  sample = np.array(sample, "float")
  sample = sample.reshape(sample.shape[0], sample.shape[1], sample.shape[2], 1) / 255

  results = model.predict(sample)
  label = []
  for result in results:
    if np.argmax(result) != 31:
        label.append(label_dict_ex[np.argmax(result)])
    print(label_dict_ex[np.argmax(result)], result[np.argmax(result)]*100)

  if len(label) == 9:
    label.insert(-2,".")
    
  label.insert(2,"-")
  label.insert(5," ")

  return label

def classification2(sample):
  
  label_dict_ex = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 
                   7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 
                   14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'K', 19: 'L', 
                   20: 'M', 21: 'N', 22: 'P', 23: 'R', 24: 'S', 25: 'T', 
                   26: 'U', 27: 'V', 28: 'X', 29: 'Y', 30: 'Z'}
  SVM = load("model/SVM3.joblib")
  
  sample = np.array(sample, np.uint8)
  sample = sample.reshape(sample.shape[0], -1)

  results = SVM.predict(sample)
  label = []
  for result in results:
    label.append(label_dict_ex[result])
  print(np.max(SVM.predict_proba(sample), axis=(1))*100)
  print(np.argmax(SVM.predict_proba(sample), axis=(1)))
  print(SVM.predict_proba(sample))

  if len(label) == 9:
    label.insert(-2,".")
    
  label.insert(2,"-")
  label.insert(5," ")
  return label

def char_digital_classification(img, xyxy, sample_size=(60,30), clf_mode="CNN"):

  # extract top-left point anh bottom-right point
  pt1 = (int(xyxy[0].item()),int(xyxy[1].item()))
  pt2 = (int(xyxy[2].item()),int(xyxy[3].item()))
  # cropping plate's area
  img_cropped = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]

  # cv2.imshow("Image cropped",img_cropped)
  # image processing and cropping digital charater
  sample = image_processing(img_cropped, sample_size)
  show_img_seg(sample )
  cv2.waitKey(0)
  # classification image
  if clf_mode == 0 or clf_mode == "CNN":
    return classification(sample)
  if clf_mode == 1 or clf_mode == "SVM":
    return classification2(sample)


def detect_plate(opt, source_image_path, sample_size=(60,30), clf_mode="CNN"):
  with torch.no_grad():
    weights, imgsz = opt['weights'], opt['img-size']
    set_logging()
    device = select_device(opt['device'])
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
      model.half()

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != 'cpu':
      model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    img0 = cv2.imread(source_image_path)
    img_backup = img0.copy()
    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
      img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment= False)[0]

    # Apply NMS
    classes = None
    if opt['classes']:
      classes = []
      for class_name in opt['classes']:

        classes.append(opt['classes'].index(class_name))


    pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes= classes, agnostic= False)
    t2 = time_synchronized()
    for i, det in enumerate(pred):
      s = ''
      s += '%gx%g ' % img.shape[2:]  # print string
      gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
      if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        for c in det[:, -1].unique():
          n = (det[:, -1] == c).sum()  # detections per class
          s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
      
        for *xyxy, conf, cls in reversed(det):
          
          res = char_digital_classification(img0, xyxy, sample_size, clf_mode=clf_mode)
          # label = f'{names[int(cls)]} {conf:.2f}'
          label = ("").join(res)
          plot_one_box(xyxy, img_backup, label=label, color=colors[int(cls)], line_thickness=3)
  if label:        
    displayOnTextLabel(label)        
  return cv2.resize(img_backup,(640, 640), interpolation=cv2.INTER_AREA)


# GUI FUNCTION
def displayOnImgLabel(myImg):
  imgLabel.configure(image=myImg)
  imgLabel.image = myImg

  textLabel.configure(text="Image changed")
  textLabel.text = "Image changed"


def openImage(urDir):
  myImg = Image.open(urDir)
  resizeImg = myImg.resize((640,395), Image.Resampling.LANCZOS)
  convertedImg = ImageTk.PhotoImage(resizeImg)
  displayOnImgLabel(convertedImg)


def chooseImg():
  root.filedir = filedialog.askopenfilename(
    initialdir="test/", 
    title="Select Image", 
    filetypes=(("Select Image","*.*"),("JPG","*.jpg"),("JPEG","*.jpeg"),("PNG","*png")))
  if root.filedir != "":
    # obj.setSrcImg(root.filedir)
    global source_image_path
    source_image_path = root.filedir
    openImage(root.filedir)

def displayOnTextLabel(message):
  textLabel.configure(text= f"Result: {message}")
  textLabel.text = f"Result: {message}"\

def displayErrOnTextLabel(message):
  textLabel.configure(text= message)
  textLabel.text = message
# END GUI FUNCTION


if __name__ == "__main__":
  
  classes_to_filter = None  #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]

  opt  = {
    
    "weights": "runs/train/exp14/weights/best.pt", # Path to weights file default weights are for nano model
    "yaml"   : "data/mydataset.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.7, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter  # list of classes to filter or None
        }
  source_image_path = None
  sample_size = (12, 28)
  # mul_plate = "D:\\program\\python\\PlateLicenseDetection\\yolov7\\test\\test4.jpg" 
  # img_result = detect_plate(opt, mul_plate, sample_size, clf_mode="SVM")
  # cv2.imshow("Result",img_result)
  # cv2.waitKey(0)
  root = Tk()
  root.title("Plate")
  # root.iconbitmap()
  root.geometry("700x500")

  frame = Frame(root)
  frame.pack(side=BOTTOM, padx=15, pady=15)

  img1 = Image.open("images/default.png")
  resizeImg = img1.resize((640,395), Image.Resampling.LANCZOS)
  convertedImg = ImageTk.PhotoImage(resizeImg)

  imgLabel = Label(root, image=convertedImg, width=640, height=395)
  textLabel = Label(root, text="Select Image")

  chooseImgBtn = Button(frame, text="Select Image", command=chooseImg)
  exitBtn = Button(frame, text="Exit", command=root.quit)
  detectBtn = Button(frame, text="Detect", command= lambda: detect_plate(opt, source_image_path, sample_size, clf_mode="SVM"))


  imgLabel.pack(side=TOP, ipadx=5, ipady= 5)
  textLabel.pack(side=TOP, ipadx=5, ipady= 5)

  chooseImgBtn.pack(side=tk.LEFT, padx= 10)
  detectBtn.pack(side=tk.LEFT, padx= 10)
  exitBtn.pack(side=tk.LEFT, padx= 10)

  root.mainloop()
