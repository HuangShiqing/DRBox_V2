import cv2
import numpy as np


def resize_img(img):
    img_w = img.shape[1]
    img_h = img.shape[0]

    ratio = img_w / img_h
    net_w, net_h = 416, 416
    if ratio < 1:
        new_h = int(net_h)
        new_w = int(net_h * ratio)
    else:
        new_w = int(net_w)
        new_h = int(net_w / ratio)
    im_sized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    dx = net_w - new_w
    dy = net_h - new_h

    if dx > 0:
        im_sized = np.pad(im_sized, ((0, 0), (int(dx / 2), 0), (0, 0)), mode='constant', constant_values=0)
        im_sized = np.pad(im_sized, ((0, 0), (0, dx - int(dx / 2)), (0, 0)), mode='constant', constant_values=0)
    else:
        im_sized = im_sized[:, -dx:, :]
    if dy > 0:
        im_sized = np.pad(im_sized, ((int(dy / 2), 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        im_sized = np.pad(im_sized, ((0, dy - int(dy / 2)), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        im_sized = im_sized[-dy:, :, :]
    return im_sized


img = cv2.imread('QUANZHOU_Level_19.tif_res_0.71_27_rotate270.tif')
# img = resize_img(img)

x = 148.820997
y = 162.968333
w = 63.035310
h = 13.029849

xmin = int(x - w / 2)
xmax = int(x + w / 2)
ymin = int(y - h / 2)
ymax = int(y + h / 2)

origin_img_sized = img  # [:, :, ::-1]
origin_img_sized = origin_img_sized.copy()
cv2.rectangle(origin_img_sized, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
# for y in range(51):
#     cv2.line(origin_img_sized, (0, 8 * (y + 1)), (416, 8 * (y + 1)), (0, 0, 255), 1)
# for x in range(51):
#     cv2.line(origin_img_sized, (8 * (x + 1), 0), (8 * (x + 1), 416), (0, 0, 255), 1)
cv2.imwrite('out.bmp', origin_img_sized)

cv2.getRotationMatrix2D()