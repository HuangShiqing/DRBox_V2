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

    loss = 0
    sum_loss_xy = 0
    sum_loss_wh = 0
    sum_loss_c = 0
    sum_loss_angle = 0

    cellbase_y = tf.to_float(tf.reshape(tf.tile(tf.range(52), [52]), (1, 52, 52, 1, 1, 1)))
    cellbase_x = tf.transpose(cellbase_y, (0, 2, 1, 3, 4, 5))
    cellbase_grid = tf.tile(tf.concat([cellbase_x, cellbase_y], -1), [batch_size, 1, 1, 3, 6, 1])
    for i in range(3):
        object_mask = y_true[i][..., 4:5]
        grid_w = tf.shape(y_pred[i])[1]  # 13
        grid_h = tf.shape(y_pred[i])[2]  # 13
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 1, 2])
        net_out_reshape = tf.reshape(y_pred[i], [-1, grid_w, grid_h, 3, 6, 6])

        adjusted_true_xy = y_true[i][..., :2] * grid_factor - cellbase_grid[:, :grid_w, :grid_h, ...]

        loss_xy = tf.reduce_sum(
            object_mask * xywh_scale * tf.nn.sigmoid_cross_entropy_with_logits(logits=net_out_reshape[..., :2],
                                                                               labels=adjusted_true_xy)) / batch_size
