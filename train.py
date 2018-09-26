import tensorflow as tf
import tensorlayer as tl
import numpy as np

from varible import *
from data import data_generator, read_txt
from model import infenence
import time
import os


def yolo_loss(y_pred, y_true):
    batch_size = Gb_batch_size
    anchors = tf.constant(Gb_anchors, dtype='float', shape=[1, 1, 1, 9, 2])

    loss = 0
    sum_loss_xy = 0
    sum_loss_wh = 0
    sum_loss_c = 0
    sum_loss_angle = 0

    cellbase_x = tf.to_float(tf.reshape(tf.tile(tf.range(52), [52]), (1, 52, 52, 1, 1, 1)))
    cellbase_y = tf.transpose(cellbase_x, (0, 2, 1, 3, 4, 5))
    cellbase_grid = tf.tile(tf.concat([cellbase_x, cellbase_y], -1), [batch_size, 1, 1, 6, 3, 1])
    for i in range(3):
        anchor = anchors[..., 3 * i:3 * (i + 1), :]
        object_mask = y_true[i][..., 4:5]
        grid_h = tf.shape(y_pred[i])[1]  # 13
        grid_w = tf.shape(y_pred[i])[2]  # 13
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 1, 2])
        net_out_reshape = tf.reshape(y_pred[i], [-1, grid_h, grid_w, 6, 3, 6])

        adjusted_true_xy = y_true[i][..., :2] * grid_factor - cellbase_grid[:, :grid_h, :grid_w, ...]
        adjusted_out_wh = tf.exp(net_out_reshape[..., 2:4]) * anchor / img_factor

        loss_xy = tf.reduce_sum(
            object_mask * xywh_scale * tf.nn.sigmoid_cross_entropy_with_logits(logits=net_out_reshape[..., :2],
                                                                               labels=adjusted_true_xy)) / batch_size
