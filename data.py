import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def rotate_box(xmin, xmax, ymin, ymax, angle):
    rot_mat = cv2.getRotationMatrix2D(((xmin + xmax) / 2, (ymin + ymax) / 2), angle, 1)
    a = np.dot(rot_mat, np.array([[xmin], [ymin], [1]]))
    b = np.dot(rot_mat, np.array([[xmax], [ymin], [1]]))
    c = np.dot(rot_mat, np.array([[xmax], [ymax], [1]]))
    d = np.dot(rot_mat, np.array([[xmin], [ymax], [1]]))

    return a[0][0], a[1][0], b[0][0], b[1][0], c[0][0], c[1][0], d[0][0], d[1][0]


def visualization(img_path):
    img = cv2.imread(img_path)
    origin_img_sized = img  # [:, :, ::-1]
    origin_img_sized = origin_img_sized.copy()

    with open(img_path + '.rbox', 'r') as fh:
        for line in fh:
            line = line.strip().split(' ')

            x = float(line[0])
            y = float(line[1])
            w = float(line[2])
            h = float(line[3])
            angle = float(line[5])

            xmin = int(x - w / 2)
            xmax = int(x + w / 2)
            ymin = int(y - h / 2)
            ymax = int(y + h / 2)

            x1, y1, x2, y2, x3, y3, x4, y4 = rotate_box(xmin, xmax, ymin, ymax, angle)
            cv2.line(origin_img_sized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            cv2.line(origin_img_sized, (int(x2), int(y2)), (int(x3), int(y3)), (0, 0, 255), 1)
            cv2.line(origin_img_sized, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), 1)
            cv2.line(origin_img_sized, (int(x4), int(y4)), (int(x1), int(y1)), (0, 0, 255), 1)
    cv2.imwrite('out1.bmp', origin_img_sized)


visualization('QUANZHOU_Level_19.tif_res_0.71_16_rotate270.tif')
