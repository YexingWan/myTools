
import os
import cv2
import numpy as np


def generate_img(img_dir):
  for (dirpath, dirnames, filenames) in os.walk(img_dir):
    for file in filenames:
      ext = os.path.splitext(file)[-1]
      if ext not in ('.png', '.jpg', '.JPEG'): continue
      img_path = os.path.join(dirpath, file)
      img = cv2.imread(img_path)
      img = cv2.resize(img, (640, 640))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      mean = np.array([123.68, 116.779, 103.939])
      img = img - mean
      img = np.expand_dims(img, 0)
      
      yield img
