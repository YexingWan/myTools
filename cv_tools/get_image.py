import cv2
import os

def get_images(image_dir):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(image_dir):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    for p in files:
        img_ori = cv2.imread(p)  # BGR
        # img = image_preprocess(img_ori)
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        yield img