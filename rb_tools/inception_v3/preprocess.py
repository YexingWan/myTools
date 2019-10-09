# Copyright (c) 2016 Corerain Technologies. All rights reserved.
# No part of this document, either material or conceptual may be
# copied or distributed, transmitted, transcribed, stored in a retrieval
# system or translated into any human or computer language in
# any form by any means, electronic, mechanical, manual
# or otherwise, or disclosed to third parties without
# the express written permission of Corerain Technologies,
# 502 Section b, 2305 Zuchongzhi Road, Zhangjiang Hi-Tech Park,
# Shanghai 201203, China.

import os
import cv2
import numpy as np

ALLOW_IMG = ('.jpg', '.JPEG', '.png')

def generate_img(imgdir):
  for (dirpath, dirnames, filenames) in os.walk(imgdir):
    for file in filenames:
      ext = os.path.splitext(file)[-1]
      if ext not in ALLOW_IMG: continue
      img_path = os.path.join(dirpath, file)
      img = preprocess(img_path)
      # Add batch dimension.
      img = np.expand_dims(img, 0)

      yield img

def preprocess(img):
  img = cv2.imread(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_h, img_w, channel = img.shape

  central_fraction = 0.875
  y = int((img_h - img_h * central_fraction) / 2)
  x = int((img_w - img_w * central_fraction) / 2)
  h = img_h - 2 * y
  w = img_w - 2 * x

  crop_img = img[y:y+h, x:x+w] / 255
  re_img = (cv2.resize(crop_img, (299, 299)) - 0.5) * 2
  
  return re_img