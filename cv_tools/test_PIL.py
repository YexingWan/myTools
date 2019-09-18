from PIL import Image, ImageFont, ImageDraw ,ImageOps
from OCR.generate_new_training_data import generate_background
import numpy as np


font = ImageFont.truetype('/home/yx-wan/newhome/workspace/myTools/cv_tools/fonts/Kosugi_Maru/KosugiMaru-Regular.ttf',60)

text = "A"
(real_w, real_h),(offset_w,offset_h) = font.font.getsize(text)
(w,h) = font.getsize(text) # the size
ascent, descent = font.getmetrics()

print("real size: {}".format((real_w, real_h)))
print("offset : {}".format((offset_w,offset_h)))
print("final size : {}".format((w,h)))
print("ascent and descent : {}".format((ascent, descent)))

bg = generate_background(200,200)
np.transpose(bg,(2,1,0))
bg = Image.fromarray(bg)

img = Image.new("RGBA", (w,h), (0,0,0,0))
draw = ImageDraw.Draw(img)
draw.text((0,0),text,font = font,fill=(255,0,0,255))
img.save("./{}.png".format(text))

img = img.rotate(30,expand=1)
print("rotated size : {}".format(img.size))
img.save("./{}_rotated.png".format(text))

img_np = np.asarray(img)
layer_A = img_np[:,:,3]
letter_pixel = np.array(np.where(layer_A > 0))
print(letter_pixel)

tly,tlx,bry,brx = min(letter_pixel[0]), min(letter_pixel[1]),max(letter_pixel[0]), max(letter_pixel[1])
img_crop = img.crop((tlx,tly,brx,bry))
print("rotated and crop size : {}".format(img_crop.size))
img_crop.save("./{}_rotated_crop.png".format(text))


bg.paste(img_crop, (0,0),  img_crop)
bg_draw = ImageDraw.Draw(bg)
bg_draw.rectangle([(0,0),img_crop.size])
bg.save("./bg_with_{}_rotated_crop.png".format(text))



