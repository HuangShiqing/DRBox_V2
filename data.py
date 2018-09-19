import cv2
import numpy as np
import matplotlib.pyplot as plt
from varible import *
from tqdm import tqdm


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


def resize_boxes(labels, img_w, net_w):
    new_labels = list()
    for label in labels:
        new_x = float(label[0]) * net_w / img_w
        new_y = float(label[1]) * net_w / img_w
        new_w = float(label[2]) * net_w / img_w
        new_h = float(label[3]) * net_w / img_w
        new_labels.append([new_x, new_y, new_w, new_h, 1, float(label[5])])
    return new_labels


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


def visualization2(img, label):
    img = img.copy()

    for line in label:
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
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)), (0, 0, 255), 1)
        cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), 1)
        cv2.line(img, (int(x4), int(y4)), (int(x1), int(y1)), (0, 0, 255), 1)
    # for y in range(51):
    #     cv2.line(img, (0, 8 * (y + 1)), (416, 8 * (y + 1)), (255, 0, 0), 1)
    # for x in range(51):
    #     cv2.line(img, (8 * (x + 1), 0), (8 * (x + 1), 416), (255, 0, 0), 1)
    cv2.imwrite('out2.bmp', img)


def read_txt():
    chunks = list()
    with open(Gb_txt_path, 'r') as fh:
        for line in tqdm(fh):
            img_name = line.split(' ')[0]
            label_name = line.split(' ')[1].strip()
            boxes = list()
            with open(Gb_label_path + label_name, 'r') as fh2:
                for line2 in fh2:
                    boxes.append(line2.strip().split(' '))

            chunks.append([img_name, boxes])
    return chunks


def random_flip(image, flip):
    if flip == 1:
        return cv2.flip(image, 1)
    return image


def flip_boxes(boxes, flip):
    if flip == 1:
        for box in boxes:
            box[0] = 416 - box[0]
            box[5] = 180 - box[5]
            # box[1] = 416 - box[1]
    return boxes


def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    def _rand_scale(scale):
        scale = np.random.uniform(1, scale)
        return scale if (np.random.randint(2) == 0) else 1. / scale

    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation)
    dexp = _rand_scale(exposure)
    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')
    # change satuation and exposure
    image[:, :, 1] *= dsat
    image[:, :, 2] *= dexp
    # change hue
    image[:, :, 0] += dhue
    image[:, :, 0] -= (image[:, :, 0] > 180) * 180
    image[:, :, 0] += (image[:, :, 0] < 0) * 180
    # avoid overflow when astype('uint8')
    image[...] = np.clip(image[...], 0, 255)
    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)


def get_data(chunk):
    img = cv2.imread(chunk[0])  # Gb_img_path +
    img = resize_img(img)
    boxes = resize_boxes(chunk[1], 300, 416)

    img = random_distort_image(img)

    flip = np.random.randint(2)
    img = random_flip(img, flip)
    boxes = flip_boxes(boxes, flip)

    # visualization2(img, boxes)
    return img, boxes


def rbox_iou(box_x, box_y, box_w, box_h, box_angle, anchor_x, anchor_y, anchor_w, anchor_h, anchor_angle):
    rect1 = ((box_x - box_w / 2, box_y - box_h / 2), (box_x + box_w / 2, box_y + box_h / 2), box_angle)
    rect1_ares = box_w * box_h
    rect2 = ((anchor_x - anchor_w / 2, anchor_y - anchor_h / 2), (anchor_x + anchor_w / 2, anchor_y + anchor_h / 2),
             anchor_angle)
    rect2_ares = anchor_w * anchor_h
    intersect = cv2.rotatedRectangleIntersection(rect1, rect2)
    union = rect1_ares + rect2_ares - intersect
    return float(intersect) / union


def get_y_true(boxes):
    y_true = list()
    # initialize the inputs and the outputs
    base_grid_w, base_grid_h = 13, 13
    cell_size = [8, 16, 32]
    y_true.append(np.zeros((Gb_batch_size, 4 * base_grid_h, 4 * base_grid_w, 3, 6, 6)))  # desired network output 3
    y_true.append(np.zeros((Gb_batch_size, 2 * base_grid_h, 2 * base_grid_w, 3, 6, 6)))  # desired network output 2
    y_true.append(np.zeros((Gb_batch_size, 1 * base_grid_h, 1 * base_grid_w, 3, 6, 6)))  # desired network output 1

    for instance_index in range(Gb_batch_size):
        for box in boxes:
            for i in range(len(Gb_anchors) // 2):
                x = box[0] // cell_size[i // 3] + 1
                y = box[1] // cell_size[i // 3] + 1
                box_w = box[2]
                box_h = box[3]
                box_angle = box[5]
                anchor_w = Gb_anchors[i * 2]
                anchor_h = Gb_anchors[i * 2 + 1]
                for j in range(6):
                    anchor_angle = 30 * j
                    iou = rbox_iou(x, y, box_w, box_h, box_angle, x, y, anchor_w, anchor_h, anchor_angle)  #


if __name__ == '__main__':
    # visualization('QUANZHOU_Level_19.tif_res_0.71_27_rotate270.tif')
    # chunks = read_txt()
    chunks = [['QUANZHOU_Level_19.tif_res_0.71_27_rotate270.tif',
               [['148.820997', '162.968333', '63.035310', '13.029849', '1', '35.000000']]]]
    img, boxes = get_data(chunks[0])
    get_y_true(boxes)
    exit()
