import os,sys, logging
sys.path.append("./")
import cv2
import numpy as np


def resize_img(img, h_resize, w_resize):
    h, w, _ = img.shape
    ratio_h = h_resize / float(h)
    ratio_w = w_resize / float(w)
    img = cv2.resize(img, (w_resize, h_resize), interpolation=cv2.INTER_LINEAR)
    return img, ratio_h, ratio_w


def crop(img, crop):
    # [left, top, right, bottom] 720*300
    assert (len(crop) == 4)
    h, w, _ = img.shape
    # logging.debug(img.shape)
    img = img[crop[1]:h - crop[3], crop[0]:w - crop[2], :]

    return img


def gamma(img, gamma=0.2):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, lookUpTable)
    return res

def shapen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) # Laplace filter
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst

def slice_otsu(img,num_slice_x,num_slice_y):
    print("in")
    if img.shape[2] != 1:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4 :
            img = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
        else:
            raise TypeError("Input image must be 1 or 3 or 4 channels, but is {} channels".format(img.shape[2]))
    slice_h = int(img.shape[0]/num_slice_y)
    slice_w = int(img.shape[1]/num_slice_x)

    for x in range(num_slice_x):
        x_tail = (x+1)*slice_w if (x == num_slice_x-1) else -1
        for y in range(num_slice_y):
            y_tail = (y + 1) * slice_h if (y == num_slice_y - 1) else -1
            img[y*slice_h:y_tail, x*slice_w:x_tail] = cv2.threshold(img[y*slice_h:y_tail, x*slice_w:x_tail], 0, 255, cv2.THRESH_OTSU)[1]
    return img

# specify loader for ORC project
def extract_rect_zone(dir: str,ratio:float):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG', 'bmp']
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    for i, f in enumerate(files):
        img = cv2.imread(f)

        # do crop, convert to gray, blur
        img = crop(img, crop=[500, 100, 500, 100])

        tem_img = img.astype(np.int)
        # logging.debug(img)
        gray_img = np.abs(tem_img[:,:,0]-tem_img[:,:,2])*2
        gray_img = np.clip(gray_img,0,255)
        gray_img = gray_img.astype(np.uint8)
        gray_img_blur = cv2.medianBlur(gray_img, 3)
        gray_img_blur = gamma(gray_img_blur,2)
        cv2.imshow(f,gray_img_blur)
        cv2.waitKey(0)

        # cv2.imwrite("./bur_{}{}".format(i,".jpg"),gray_img_blur)

        # otsu_binaries, erode, contours, get biggest contour, find hull, find min area rectangle
        gray_img_2 = cv2.threshold(gray_img_blur, 0, 255, cv2.THRESH_OTSU)[1]
        # gray_img_2 = slice_otsu(img,5,5)
        gray_img_2 = cv2.erode(gray_img_2, np.ones((7, 7)))
        cv2.imshow("binary",gray_img_2)
        cv2.waitKey(0)

        # cv2.imwrite("./bi_{}{}".format(i,".jpg"),gray_img_2)

        contours = cv2.findContours(gray_img_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # print(contours)
        max_index = np.argmax([cv2.contourArea(c) for c in contours])
        contours[max_index] = cv2.convexHull(contours[max_index])
        rect = cv2.minAreaRect(cv2.approxPolyDP(contours[max_index], 10, True))
        box = cv2.boxPoints(rect)

        # draw box and save blackboard
        # blackboard = np.zeros_like(img)
        # cv2.drawContours(blackboard,contours,max_index,[0,255,0])
        # cv2.imwrite("./blackboard_{}{}".format(i,".jpg"),blackboard)

        # order box by: lt, rt, rb. lb
        box_s = np.squeeze(np.array(box)).tolist()
        box_s = sorted(box_s, key=lambda x: x[1])

        if box_s[0][0] >= box_s[1][0]:
            box_s[0], box_s[1] = box_s[1], box_s[0]
        if box_s[2][0] <= box_s[3][0]:
            box_s[2], box_s[3] = box_s[3], box_s[2]
        w = int(abs(box_s[1][0] - box_s[0][0])* ratio)
        h = int(abs(box_s[1][1] - box_s[2][1])* ratio)
        box_t = np.array([[0, 0],
                          [w, 0],
                          [w, h],
                          [0, h]], dtype=np.float32)
        box_s = np.array(box_s, dtype=np.float32)

        # get affine transfer matrix, do transfer
        mat = cv2.getPerspectiveTransform(box_s, box_t)
        result = cv2.warpPerspective(img, mat, (w, h))
        # result = shapen(result)
        ori_save_dir = "./result"
        if not os.path.exists(ori_save_dir):
            os.mkdir(ori_save_dir)
        cv2.imshow("result",result)
        cv2.waitKey(0)
        cv2.imwrite(os.path.join(ori_save_dir,"ori_result_{}{}".format(i,".jpg")),result)

        # #select blue channel, gamma transfer
        # result_processed = result[:,:,0]
        # result_processed = gamma(result_processed,1.3)
        # result_processed = shapen(result_processed)
        # gray_save_dir = "/home/yx-wan/newhome/workspace/myTools/cv_tools/OCR/gray"
        # if not os.path.exists(gray_save_dir):
        #     os.mkdir(gray_save_dir)
        # cv2.imwrite(os.path.join(gray_save_dir,"ori_result_{}{}".format(i,".jpg")),result_processed)

        # IMG_MEAN = np.array([123.68, 116.78, 103.94], dtype=np.float32)
        # batch = np.expand_dims(result - IMG_MEAN, 0)
        # batch_processed = np.expand_dims(cv2.cvtColor(result_processed,cv2.COLOR_GRAY2BGR)-IMG_MEAN,0)

        yield box_s, f, img, result

for _,_,_,_ in extract_rect_zone(dir = "/home/kevin/workspace/pycharm-remote/Corerain/myTools/cv_tools/test_crop_rotate",ratio=1.3):
    pass








