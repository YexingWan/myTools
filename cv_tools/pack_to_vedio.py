import cv2
import os




def get_images(image_dir):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(image_dir):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    ori = os.path.join(parent, filename)

                    # files.append(ori)

                    # build sequance
                    tep_0 = os.path.splitext(filename)
                    tep_1 = tep_0[0].split("_")
                    surfix,idx = "_".join(tep_1[:-1]), str(tep_1[-1]).zfill(6)
                    re = surfix+ "_" + idx + tep_0[1]
                    re = os.path.join(parent, re)


                    os.rename(ori,re)
                    print("{} => {}".format(ori,re))
                    files.append(re)
                    break

    files = sorted(files)
    print('Find {} images'.format(len(files)))
    for p in files:
        img_ori = cv2.imread(p)  # BGR
        yield img_ori


DIR = "/home/yx-wan/newhome/dataset/face_dataset/boyaFaceRec-data/frames_result"
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
videoWriter = cv2.VideoWriter('HRNET_tiny_resnet101_new.avi',fourcc , 25, (1920,1080))
from tqdm import tqdm
for img in tqdm(get_images(DIR)):
    videoWriter.write(img)
videoWriter.release()


