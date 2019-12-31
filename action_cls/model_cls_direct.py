"""
    Compared with model_baseline, do not use correlation output for skip link
    Compared to model_baseline_fixed, added return values to test whether nsample is set reasonably.
"""

import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/sampling'))
import tf_util
from net_utils import *

def placeholder_inputs(batch_size, num_point, num_frames):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * num_frames, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, num_frames, is_training, bn_decay=None):
    """ Input:
            point_cloud: [batch_size, num_point * num_frames, 3]
        Output:
            net: [batch_size, num_class] """
    end_points = {}
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value // num_frames

    l0_xyz = point_cloud
    l0_time = tf.concat([tf.ones([batch_size, num_point, 1]) * i for i in range(num_frames)], \
            axis=-2)
    l0_points = tf.concat([point_cloud[:, :, 3:], l0_time], axis=-1)

    RADIUS1 = np.linspace(0.5, 0.6, num_frames, dtype='float32')
    RADIUS2 = RADIUS1 * 2
    RADIUS3 = RADIUS1 * 4
    RADIUS4 = RADIUS1 * 8

    l1_xyz, l1_time, l1_points, l1_indices = meteor_direct_module(l0_xyz, l0_time, l0_points, npoint=1024, radius=RADIUS1, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_time, l2_points, l2_indices = meteor_direct_module(l1_xyz, l1_time, l1_points, npoint=512, radius=RADIUS2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_time, l3_points, l3_indices = meteor_direct_module(l2_xyz, l2_time, l2_points, npoint=128, radius=RADIUS3, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 20, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024*2,6))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
