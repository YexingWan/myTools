# coding: UTF-8
from ..image_aug import shapen,extract_rect_zone
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import string
import cv2
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ETimport
import numpy as np
import os,sys, logging
sys.path.append("./")



def high_fix_resize(img:np.ndarray,h_fix):
    h_fix = h_fix+np.random.randint(-5,5)
    ratio_w = h_fix/img.shape[0]
    w_fix = int(img.shape[1]*ratio_w)
    img = cv2.resize(img,(w_fix,h_fix))
    return img


def iter_extract_list(num_img,h = 288, w = 736):
    extract_list = [(box, path, ori_img, result) for box, path, ori_img, result
                    in
                    extract_rect_zone("/mnt/newhome/yexing/workspace/myTools/cv_tools/OCR/train_crop_rotate/", h, w)]
    index = np.random.randint(0,len(extract_list)-1,num_img)
    for i in index:
        yield extract_list[i]


def generate_background(h = 288, w = 736):
    # R60 G120 B180
    R = np.random.uniform(45,75,(h,w,1))
    G = np.random.uniform(110,140,(h,w,1))
    B = np.random.uniform(160,190,(h,w,1))
    background = np.concatenate([B,G,R],axis=2).astype(np.uint8)
    return background


def generate_new_data(num_img,h = 288, w = 736):
    """
    generate data by stick,
    根据方法中设定位置+扰动的为欧洲贴从原图扣下来的字符/图片
    这个方法写的时候是往扣下来的名牌或generate_background方法生成的名牌上贴

    :param num_img:
    :param h:
    :param w:
    :return:
    """
    training_data_dir = "/mnt/newhome/yexing/workspace/myTools/cv_tools/OCR/train_crop_rotate"
    save_dir = "/mnt/newhome/yexing/workspace/myTools/cv_tools/OCR/fake_training_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    class_file = os.path.join(training_data_dir,"classes.txt")
    label_file = os.path.join(training_data_dir,"train.txt")
    f = open(class_file,"r")
    class_list = f.readlines()
    f.close()
    label_dict = defaultdict(list)
    h_letter_capital = 63
    h_letter_small = 35
    h_letter_sig_small = 15
    h_letter_sig_big = 25

    f = open(label_file, "r")
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(" ")
        img_name = line[0]
        img_path = os.path.join(training_data_dir,img_name)
        img = cv2.imread(img_path)

        img = shapen(img) if np.random.rand() > 0.5 else img
        for bbox in line[1:]:
            xmin, ymin, xmax, ymax, id = bbox.split(",")
            xmin, ymin, xmax, ymax, id = int(xmin), int(ymin), int(xmax), int(ymax), int(id)
            name = class_list[id].strip(" ")
            name = name.strip("\n")
            crop = img[ymin:ymax + 1, xmin:xmax + 1, :]
            # if not path.exists(name):
            #     os.mkdir(name)
            # cv2.imwrite("./{}/{}_{}.jpg".format(name,name,np.random.rand()),crop)

            if name.isupper() or name.isnumeric():
                h_resize = h_letter_capital
            elif name.islower():
                h_resize = h_letter_small
            elif name == "-":
                h_resize = h_letter_sig_small
            else:
                h_resize = h_letter_sig_big

            label_dict[name].append(high_fix_resize(crop, h_resize))
            # label_dict[name].append(crop)
    class_list = [c.strip(" ").strip("\n") for c in class_list]
    print("origin class list:{}".format(class_list))

    class_list = list(label_dict.keys())
    print("exist class list:{}".format(class_list))

    label_file = open(os.path.join(save_dir,"new_classes.txt"),"w")
    for c in class_list:
        label_file.writelines(c+"\n")
    label_file.close()



    file = open(os.path.join(save_dir,"new_train.txt"), "w")


    # GENERATE NEW DATA
    for index in range(num_img):
        bg = generate_background(h,w)

        # line 1 and line 2 length setting
        l1_short_len = np.random.randint(1,4)
        l2_short_1_len = np.random.randint(1,4)
        l2_short_2_len = np.random.randint(1,4)

        ran1 = np.random.randint(0,len(class_list)-1,14)
        ran2 = np.random.randint(0,len(class_list)-1,14)

        name1 = [class_list[i].strip(" ").strip("\n") for i in ran1]
        name2 = [class_list[i].strip(" ").strip("\n") for i in ran2]

        logging.debug(ran1)
        logging.debug(ran2)
        logging.debug(name1)
        logging.debug(name2)

        random_sample_class_l1 = [label_dict[name] for name in name1]
        random_sample_class_l2 = [label_dict[name] for name in name2]

        l1 = [c[np.random.randint(0,len(c))] for c in random_sample_class_l1]
        l2 = [c[np.random.randint(0,len(c))] for c in random_sample_class_l2]

        file.write("generate_sample_{}.jpg ".format(index))

        cur_lb_y = int(1/5 * h)+h_letter_capital+np.random.randint(-5,5)
        max_y = -1
        cur_lt_x = int(1/8 * w)+np.random.randint(-5,5)

        for l, id, name in zip(l1[:l1_short_len],ran1[:l1_short_len],name1[:l1_short_len]):
            if name in string.punctuation:
                bg[int(cur_lb_y - h_letter_capital / 2 - l.shape[0] / 2):int(cur_lb_y - h_letter_capital / 2 + l.shape[0] / 2),
                cur_lt_x:cur_lt_x + l.shape[1], :] = l

                file.write("{},{},{},{},{} ".format(cur_lt_x,
                                                    int(cur_lb_y - h_letter_capital/2 - l.shape[0]/2),
                                                    cur_lt_x+l.shape[1],
                                                    int(cur_lb_y - h_letter_capital/2 + l.shape[0]/2),
                                                    id))
            else:
                bg[cur_lb_y - l.shape[0]:cur_lb_y, cur_lt_x:cur_lt_x + l.shape[1], :] = l
                file.write("{},{},{},{},{} ".format(cur_lt_x,cur_lb_y-l.shape[0],cur_lt_x+l.shape[1],cur_lb_y,id))

            cur_lb_y = cur_lb_y + np.random.randint(-3,3)
            if max_y < cur_lb_y:
                max_y = cur_lb_y
            cur_lt_x = cur_lt_x+l.shape[1] + np.random.randint(-3,3)

        cur_lt_x = cur_lt_x+30
        for l,id,name in zip(l1[l1_short_len:],ran1[l1_short_len:], name1[l1_short_len:]):
            if name in string.punctuation:
                bg[int(cur_lb_y - h_letter_capital / 2 - l.shape[0] / 2):int(cur_lb_y - h_letter_capital / 2 + l.shape[0] / 2),
                cur_lt_x:cur_lt_x + l.shape[1], :] = l

                file.write("{},{},{},{},{} ".format(cur_lt_x,
                                                    int(cur_lb_y - h_letter_capital / 2 - l.shape[0] / 2),
                                                    cur_lt_x + l.shape[1],
                                                    int(cur_lb_y - h_letter_capital / 2 + l.shape[0] / 2),
                                                    id))
            else:
                bg[cur_lb_y - l.shape[0]:cur_lb_y, cur_lt_x:cur_lt_x + l.shape[1], :] = l
                file.write(
                    "{},{},{},{},{} ".format(cur_lt_x, cur_lb_y - l.shape[0], cur_lt_x + l.shape[1], cur_lb_y, id))

            cur_lb_y = cur_lb_y + np.random.randint(-3, 3)
            if max_y < cur_lb_y:
                max_y = cur_lb_y
            cur_lt_x = cur_lt_x+l.shape[1] + np.random.randint(-3,3)


        cur_lb_y = max_y+h_letter_capital+30+np.random.randint(-5,5)
        cur_lt_x = int(1/8 * w)+np.random.randint(-5,5)
        for l,id,name in zip(l2[:l2_short_1_len],ran2[:l2_short_1_len],name2[:l2_short_1_len]):
            if name in string.punctuation:
                bg[int(cur_lb_y - h_letter_capital / 2 - l.shape[0] / 2):int(cur_lb_y - h_letter_capital / 2 + l.shape[0] / 2),
                cur_lt_x:cur_lt_x + l.shape[1], :] = l


                file.write("{},{},{},{},{} ".format(cur_lt_x,
                                                    int(cur_lb_y - h_letter_capital / 2 - l.shape[0] / 2),
                                                    cur_lt_x + l.shape[1],
                                                    int(cur_lb_y - h_letter_capital / 2 + l.shape[0] / 2),
                                                    id))
            else:
                bg[cur_lb_y - l.shape[0]:cur_lb_y, cur_lt_x:cur_lt_x + l.shape[1], :] = l
                file.write(
                    "{},{},{},{},{} ".format(cur_lt_x, cur_lb_y - l.shape[0], cur_lt_x + l.shape[1], cur_lb_y, id))

            cur_lb_y = cur_lb_y + np.random.randint(-3, 3)
            cur_lt_x = cur_lt_x+l.shape[1]+ np.random.randint(-3,3)

        cur_lt_x = cur_lt_x+20
        for l,id,name in zip(l2[l2_short_1_len:l2_short_1_len+l2_short_2_len],
                        ran2[l2_short_1_len:l2_short_1_len+l2_short_2_len],
                        name2[l2_short_1_len:l2_short_1_len+l2_short_2_len]):
            if name in string.punctuation:
                bg[int(cur_lb_y - h_letter_capital / 2 - l.shape[0] / 2):int(cur_lb_y - h_letter_capital / 2 + l.shape[0] / 2),
                cur_lt_x:cur_lt_x + l.shape[1], :] = l


                file.write("{},{},{},{},{} ".format(cur_lt_x,
                                                    int(cur_lb_y - h_letter_capital / 2 - l.shape[0] / 2),
                                                    cur_lt_x + l.shape[1],
                                                    int(cur_lb_y - h_letter_capital / 2 + l.shape[0] / 2),
                                                    id))
            else:
                bg[cur_lb_y - l.shape[0]:cur_lb_y, cur_lt_x:cur_lt_x + l.shape[1], :] = l
                file.write(
                    "{},{},{},{},{} ".format(cur_lt_x, cur_lb_y - l.shape[0], cur_lt_x + l.shape[1], cur_lb_y, id))

            cur_lb_y = cur_lb_y + np.random.randint(-3, 3)
            cur_lt_x = cur_lt_x+l.shape[1]+ np.random.randint(-3,3)

        cur_lt_x = cur_lt_x+20
        for l,id,name in zip(l2[l2_short_1_len+l2_short_2_len:],
                             ran2[l2_short_1_len+l2_short_2_len:],
                             name2[l2_short_1_len+l2_short_2_len:]):
            if name in string.punctuation:
                bg[int(cur_lb_y - h_letter_capital / 2 - l.shape[0] / 2):int(cur_lb_y - h_letter_capital / 2 + l.shape[0] / 2),
                cur_lt_x:cur_lt_x + l.shape[1], :] = l



                file.write("{},{},{},{},{} ".format(cur_lt_x,
                                                    int(cur_lb_y - h_letter_capital / 2 - l.shape[0] / 2),
                                                    cur_lt_x + l.shape[1],
                                                    int(cur_lb_y - h_letter_capital / 2 + l.shape[0] / 2),
                                                    id))
                cur_lb_y = cur_lb_y + np.random.randint(-3, 3)
                cur_lt_x = cur_lt_x + l.shape[1] + np.random.randint(-3, 3)

            else:
                bg[cur_lb_y - l.shape[0]:cur_lb_y, cur_lt_x:cur_lt_x + l.shape[1], :] = l

                file.write(
                    "{},{},{},{},{} ".format(cur_lt_x, cur_lb_y - l.shape[0], cur_lt_x + l.shape[1], cur_lb_y, id))
                cur_lb_y = cur_lb_y + np.random.randint(-3, 3)
                cur_lt_x = cur_lt_x + l.shape[1] + np.random.randint(-3, 3)

        file.write("\n")
        cv2.imwrite(os.path.join(save_dir,"generate_sample_{}.jpg".format(index)), bg)
    file.close()


def generate_strong_random_new_data_PIL(num_img,h = 288, w = 736,
                                        angle_ramdom_range = (-30,30),
                                        num_letter_random_range = (24,28),
                                        size_random_range=(35,60),R_range = (0,256),
                                        G_range = (0,256), B_range = (0,256),
                                        width_range = (1,1),
                                        length_range = (1,1),
                                        random_blur = 0.0,
                                        random_block = 0.0,
                                        random_width = 0.0,
                                        random_length = 0.0,
                                        random_location = True,
                                        num_row = 2,
                                        num_letter_pre_row = 14,
                                        splite = 0.2):
    """
    更多设定的,用PIL做字符样本生成的接
    这个方法写的时候是往扣下来的名牌或generate_background方法生成的名牌上贴

    :param num_img:
    :param h:
    :param w:
    :param angle_ramdom_range:
    :param num_letter_random_range:
    :param size_random_range:
    :param R_range:
    :param G_range:
    :param B_range:
    :param width_range:
    :param length_range:
    :param random_blur:
    :param random_block:
    :param random_width:
    :param random_length:
    :param random_location:
    :param num_row:
    :param num_letter_pre_row:
    :param splite:
    :return:
    """
    font_path = '/home/yx-wan/newhome/workspace/myTools/random_sample_gen/fonts/'
    class_file = "/mnt/newhome/yexing/workspace/myTools/cv_tools/random_sample_gen/train_crop_rotate/classes.txt"
    save_dir = "/mnt/newhome/yexing/workspace/myTools/cv_tools/random_sample_gen/fake_training_data_PIL_random"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    class_file = open(class_file,"r")
    class_list = class_file.readlines()
    class_list = [c.strip(" ").strip("\n") for c in class_list]
    lable_file = open(os.path.join(save_dir,"new_train.txt"),"w")
    print("fine {} letter.".format(len(class_list)))

    font_files = []
    exts = ['ttf']
    for parent, dirnames, filenames in os.walk(font_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    font_files.append(os.path.join(parent, filename))
                    break
    print('Find {} ttf file'.format(len(font_files)))
    assert(len(font_files) > 0)

    print("generating samples...")
    for index in range(num_img):

        lable_file.write("generate_PIL_random_{}.jpg ".format(index))

        if random_location:
            num_letter = np.random.randint(num_letter_random_range[0],num_letter_random_range[1]+1)
        else:
            num_letter = num_letter_pre_row*num_row
            cur_lt_y = int(1 / 5 * h) + np.random.randint(-5, 5)
            max_y = -1
            cur_lt_x = int(1 / 8 * w) + np.random.randint(-5, 5)

        letter_random_idx = np.random.randint(0,len(class_list),num_letter)

        mask_empty = np.ones((h,w))
        continue_flag = False

        bg = generate_background(h,w)
        # bg = cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)
        bg = Image.fromarray(bg)
        bg_draw = ImageDraw.Draw(bg)


        for cur_number,idx in enumerate(letter_random_idx):

            # random size, color, font and angle
            size_random = np.random.randint(size_random_range[0], size_random_range[1]+1)
            R_random = np.random.randint(R_range[0], R_range[1]+1)
            G_random = np.random.randint(G_range[0], G_range[1]+1)
            B_random = np.random.randint(B_range[0], B_range[1]+1)
            angle_random = np.random.randint(angle_ramdom_range[0], angle_ramdom_range[1]+1)
            font = ImageFont.truetype(font_files[np.random.randint(0,len(font_files))], size_random)

            text = class_list[idx]
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), text, font=font, fill=(R_random, G_random, B_random, 255))
            # img.save("./{}.png".format(text))

            img = img.rotate(angle_random, expand=1)
            logging.debug("rotated size : {}".format(img.size))
            # img.save("./{}_rotated.png".format(text))

            img_np = np.asarray(img)
            layer_A = img_np[:, :, 3]
            letter_pixel = np.array(np.where(layer_A > 0))
            tly, tlx, bry, brx = min(letter_pixel[0]), min(letter_pixel[1]), max(letter_pixel[0]), max(letter_pixel[1])
            img_crop = img.crop((tlx, tly, brx, bry))
            letter_pixel = letter_pixel-np.array([[tly],[tlx]])
            logging.debug("rotated and crop size : {}".format(img_crop.size))
            # img_crop.save("./{}_rotated_crop.png".format(text))


            # randomize length width blur and block
            width_raito  = 1
            length_raito = 1
            if random_width > 0 and random_width > np.random.rand():
                if img_crop.size[0] >= 20 and img_crop.size[1] >= 20:
                    width_raito = np.random.uniform(width_range[0],width_range[1])
                    if text == "U" or "V":
                        width_raito = width_raito*0.8

            if random_length > 0 and random_length > np.random.rand():
                if img_crop.size[0] >= 20 and img_crop.size[1] >= 20:
                    length_raito = np.random.uniform(length_range[0], length_range[1])

            img_crop = img_crop.resize((int(width_raito*img_crop.size[0]),int(length_raito*img_crop.size[1])))

            if random_blur > 0 and random_blur > np.random.rand():
                if img_crop.size[0] >= 10 and img_crop.size[1] >=10:
                    img_crop = img_crop.filter(ImageFilter.GaussianBlur(2))

            if random_block > 0 and random_block > np.random.rand():
                block = Image.fromarray(np.zeros((5,5,4)),mode = "RGBA")
                times = np.random.randint(1,4)
                random_letter_pixel_idx = np.random.randint(0,letter_pixel.shape[1],times)
                for t in range(times):
                    if letter_pixel[1,random_letter_pixel_idx[t]]+5 >= img_crop.size[0] or letter_pixel[0,random_letter_pixel_idx[t]] + 5 >= img_crop.size[1]:
                        # logging.info("skip: {} vs {}".format(img_crop.size,(letter_pixel[1,random_letter_pixel_idx[t]],letter_pixel[0,random_letter_pixel_idx[t]])))
                        continue
                    r_x =letter_pixel[1,random_letter_pixel_idx[t]]
                    r_y =letter_pixel[0,random_letter_pixel_idx[t]]
                    img_crop.paste(block,(r_x,r_y))



            bg_tly_random,bg_tlx_random = -1, -1

            # place the letter randomly
            if random_location:
                for _ in range(50):
                    bg_tly_random = np.random.randint(0,h-(img_crop.size[1]))
                    bg_tlx_random = np.random.randint(0,w-(img_crop.size[0]))
                    if np.all(mask_empty[bg_tly_random:bg_tly_random + img_crop.size[1],bg_tlx_random:bg_tlx_random + img_crop.size[0]]):
                        continue_flag = True
                        mask_empty[bg_tly_random:bg_tly_random + img_crop.size[1],bg_tlx_random:bg_tlx_random + img_crop.size[0]] = 0
                        break
                if not continue_flag:
                    break
                continue_flag = False
                bg.paste(img_crop, (bg_tlx_random, bg_tly_random), img_crop)
                lable_file.write("{},{},{},{},{} ".format(bg_tlx_random, bg_tly_random,
                                                          bg_tlx_random + img_crop.size[0],
                                                          bg_tly_random + img_crop.size[1],
                                                          idx))

            else:
                if cur_lt_x + img_crop.size[0] >= w-5 or cur_lt_y + img_crop.size[1] >= h-5:
                    if (cur_number + 1) % num_letter_pre_row == 0:
                        cur_lt_y = cur_lt_y + h_letter_capital + np.random.randint(40, 50)
                        cur_lt_x = int(1 / 8 * w) + np.random.randint(-5, 5)
                    continue
                h_letter_capital  = size_random_range[1]
                text = class_list[idx]
                if text in string.punctuation:
                    bg.paste(img_crop, (cur_lt_x, int(cur_lt_y + h_letter_capital / 2 - img_crop.size[1] / 2 )), img_crop)

                    lable_file.write("{},{},{},{},{} ".format(cur_lt_x,
                                                        int(cur_lt_y + h_letter_capital / 2 - img_crop.size[1] / 2),
                                                        cur_lt_x + img_crop.size[0],
                                                        int(cur_lt_y + h_letter_capital / 2 + img_crop.size[1] / 2),
                                                        idx))
                    # bg_draw.rectangle([(cur_lt_x, int(cur_lt_y + h_letter_capital / 2 - img_crop.size[1] / 2)),
                    #                (cur_lt_x + img_crop.size[0], int(cur_lt_y + h_letter_capital / 2 + img_crop.size[1] / 2))])
                else:
                    bg.paste(img_crop, (cur_lt_x, cur_lt_y), img_crop)

                    lable_file.write("{},{},{},{},{} ".format(cur_lt_x,
                                                              cur_lt_y,
                                                              cur_lt_x + img_crop.size[0],
                                                              cur_lt_y + img_crop.size[1],
                                                              idx))
                    # bg_draw.rectangle([(cur_lt_x, cur_lt_y),
                    #                (cur_lt_x + img_crop.size[0], cur_lt_y + img_crop.size[1])])



                if (cur_number+1) % num_letter_pre_row == 0:
                    cur_lt_y = cur_lt_y+h_letter_capital+np.random.randint(10,20)
                    cur_lt_x = int(1 / 8 * w) + np.random.randint(0, 2)
                else:
                    cur_lt_y = cur_lt_y+np.random.randint(-2, 2)
                    cur_lt_x = cur_lt_x + img_crop.size[0] + 5 + np.random.randint(-2,2)
                    if max_y < cur_lt_y:
                        max_y = cur_lt_y
                    if splite > np.random.rand():
                        cur_lt_x = cur_lt_x + img_crop.size[0]



            # bg_draw.rectangle([(bg_tlx_random,bg_tly_random),
            #                    (bg_tlx_random+img_crop.size[0],bg_tly_random+img_crop.size[1])])
        lable_file.write("\n")
        bg.save(os.path.join(save_dir,"generate_PIL_random_{}.jpg".format(index)))
    lable_file.close()
    print("Done")




def generate_new_warped_data(num_img):
    """
    这个是针对荣旭当时的情况,先生成新的铭牌.然后把生成的名牌从新贴回带背景的原图上.

    :param num_img:
    :return:
    """
    h = 288
    w = 736
    warped_card_save_dir = "/mnt/newhome/yexing/workspace/myTools/cv_tools/OCR/fake_warped_data"
    if not os.path.exists(warped_card_save_dir):
        os.mkdir(warped_card_save_dir)

    new_card_save_dir = "/mnt/newhome/yexing/workspace/myTools/cv_tools/OCR/fake_training_data_PIL_random"
    generate_strong_random_new_data_PIL(num_img,h,w,angle_ramdom_range=(0,0),
                                        num_letter_random_range=(30,60),
                                        size_random_range=(55,65),
                                        R_range=(0,0),
                                        G_range=(0,0),
                                        B_range=(0,0),
                                        width_range=(0.5,0.9),
                                        random_width=0.4,
                                        random_blur=0.0,
                                        random_block=0.0,
                                        random_location=False,
                                        splite=0.2,
                                        num_row=3,
                                        num_letter_pre_row=20)
    label_file = os.path.join(new_card_save_dir,"new_train.txt")
    f = open(label_file, "r")
    lines = f.readlines()

    warped_label_file = os.path.join(warped_card_save_dir,"new_train.txt")
    f_warped_label = open(warped_label_file, "w")



    for idx,((box, path, ori_img, _),line) in enumerate(zip(iter_extract_list(num_img,h,w),lines)):

        box_ori = np.array([[0, 0],
                          [w, 0],
                          [w, h],
                          [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(box_ori,box)
        line = line.strip().strip("\n").split(" ")
        img_name = line[0]
        img_path = os.path.join(new_card_save_dir,img_name)
        gen_data = cv2.imread(img_path)

        warped_gen = cv2.warpPerspective(gen_data,M,(ori_img.shape[1],ori_img.shape[0]))
        mask = np.zeros((h,w))
        mask.fill(255)

        warped_mask = cv2.warpPerspective(mask,M,(ori_img.shape[1],ori_img.shape[0]))
        warped_mask= warped_mask.astype(np.bool)
        warped_mask = np.expand_dims(warped_mask,axis=0)
        warped_mask = np.repeat(warped_mask,3,axis=0)
        warped_mask = np.transpose(warped_mask,(1,2,0))

        np.copyto(ori_img, warped_gen,where = warped_mask)
        cv2.imwrite(os.path.join(warped_card_save_dir,"gen_warp_sample_{}.jpg".format(idx)),ori_img)
        f_warped_label.write("gen_warp_sample_{}.jpg ".format(idx))

        for bbox in line[1:]:
            logging.debug("box: {}".format(bbox))

            xmin, ymin, xmax, ymax, id = bbox.split(",")
            xmin, ymin, xmax, ymax, id = int(xmin), int(ymin), int(xmax), int(ymax), int(id)
            points = np.array([[[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]])
            points = points.astype(np.float)
            warped_corner = cv2.perspectiveTransform(points,M)
            logging.debug("warped coordinate: {}".format(warped_corner))
            f_warped_label.write("{},{},{},{},{} ".format(int(min(warped_corner[0,:,0])),
                                                          int(min(warped_corner[0,:,1])),
                                                          int(max(warped_corner[0,:,0])),
                                                          int(max(warped_corner[0,:,1])),
                                                          id))
        f_warped_label.write("\n")
    f_warped_label.close()



logging.basicConfig(level=logging.INFO)


num = sys.argv[1]
generate_new_warped_data(num)

# generate_strong_random_new_data_PIL(10,h = 288, w = 736,
#                                         angle_ramdom_range = (-15,15),
#                                         num_letter_random_range = (50,100),
#                                         size_random_range=(35,60),R_range = (0,0),
#                                         G_range = (0,0), B_range = (0,0),
#                                         random_blur = 0.7,
#                                         random_block = 0.7)












