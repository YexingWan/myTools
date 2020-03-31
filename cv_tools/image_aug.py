import numpy as np
import cv2
import random
from PIL import ImageEnhance
import PIL.Image as Image


def resize_img(img, height, width, inter=cv2.INTER_AREA):
    h, w, _ = img.shape
    ratio_h = height / float(h)
    ratio_w = width / float(w)
    img = cv2.resize(img, (width, height), interpolation=inter)
    return img, ratio_h, ratio_w


def resize_img_short_edge(img, short_edge: int, inter=cv2.INTER_AREA):
    h, w, _ = img.shape
    if h > w:
        return resize_img_keep_ratio(img, width=short_edge, inter=inter)
    else:
        return resize_img_keep_ratio(img, height=short_edge, inter=inter)


def resize_img_keep_ratio(img, height=None, width=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    h, w, _ = img.shape

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return img

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized, ratio_h, ratio_w = resize_img(img, dim[1], dim[0], inter=inter)

    # return the resized image
    return resized, ratio_h, ratio_w


def crop(img, crop):
    # [left, top, right, bottom] 720*300
    assert (len(crop) == 4)
    h, w, _ = img.shape
    # logging.debug(img.shape)
    img = img[crop[1]:h - crop[3], crop[0]:w - crop[2], :]

    return img


# Gamma correction to input image
def gamma(img, gamma=0.2):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, lookUpTable)
    return res


def shapen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # Laplace filter
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst


def slice_otsu(img, num_slice_x, num_slice_y):
    print("in")
    if img.shape[2] != 1:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            raise TypeError("Input image must be 1 or 3 or 4 channels, but is {} channels".format(img.shape[2]))
    slice_h = int(img.shape[0] / num_slice_y)
    slice_w = int(img.shape[1] / num_slice_x)

    for x in range(num_slice_x):
        x_tail = (x + 1) * slice_w if (x == num_slice_x - 1) else -1
        for y in range(num_slice_y):
            y_tail = (y + 1) * slice_h if (y == num_slice_y - 1) else -1
            img[y * slice_h:y_tail, x * slice_w:x_tail] = \
            cv2.threshold(img[y * slice_h:y_tail, x * slice_w:x_tail], 0, 255, cv2.THRESH_OTSU)[1]
    return img


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = random.uniform(0.001,0.005)
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape((row, col, ch))
        noisy = image + gauss
        noisy = np.clip(noisy,0,1)
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.01
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape((row, col, ch))
        noisy = image + image * gauss
        return noisy

def hue_jitter(img,rate,alpha=0.25):

    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_HSV)
    H = H.astype(np.int64)

    if random.random() < rate:
        H = (H + random.randint(0, int(alpha * 180))) % 180
        print("max H :{}".format(np.max(H)))

    H = H.astype(np.uint8)
    HSV = np.concatenate((np.expand_dims(H, 0), np.expand_dims(S, 0), np.expand_dims(V, 0)), axis=0)
    HSV = np.transpose(HSV, (1, 2, 0))
    HSV = HSV.astype(np.uint8)
    img = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

    return img



class ColorJitter(object):
    def __init__(self, transform_dict):
        transform_type_dict = dict(
            brightness=ImageEnhance.Brightness, contrast=ImageEnhance.Contrast,
            sharpness=ImageEnhance.Sharpness, color=ImageEnhance.Color
        )
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]

    def __call__(self, img):
        out = img
        rand_num = np.random.uniform(0, 1, len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_num[i] * 2.0 - 1.0) + 1  # r in [1-alpha, 1+alpha)
            out = transformer(out).enhance(r)

        return out



def random_aug(img,
               hue_r = .5,
               noise_r = .5,
               sharp_r = .5,
               blur_r=.5,
               gamma_r=.5
               ):
    """
    :param img: image array with shape (H,W,3)
    :param H_r: probability for random adjust hue
    :param S_r: probability for random adjust saturation
    :param V_r: probability for random adjust brightness
    :param noise_r: probability for random add gaussian noise
    :param sharp_r: probability for random do sharpen
    :param blur_r: probability for random do blur
    :param gamma_rm: probability for random do gamma
    :return:
    """

    random_seed = [random.random() for _ in range(4)]
    if random_seed[0] < blur_r:
        img = cv2.blur(img,(3,3))
    if random_seed[1] < gamma_r:
        img = gamma(img,gamma = 0.5)
    if random_seed[2] < sharp_r:
        img = shapen(img)
    if random_seed[3] < noise_r:
        img = img / 255
        img = noisy("gauss",img)
        img = img * 255

    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    _transform_dict = {'brightness': 0.5, 'contrast':0.5, 'sharpness': 0.5, 'color': 0.5}
    cj = ColorJitter(_transform_dict)
    img = cj(img)

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    img = hue_jitter(img,hue_r,0.25)

    return img
