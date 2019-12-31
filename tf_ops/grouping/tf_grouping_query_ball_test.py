
import tensorflow as tf
import numpy as np
import os

from tf_grouping import query_ball_point, query_ball_point_var_rad, group_point

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

np.random.seed(0)
total_points = 16384
npoint = 5
nsample = 1024

xyz1 = np.random.uniform(0, 1, size=(1,total_points,2))
xyz2 = np.random.uniform(0, 1, size=(1,total_points,2))
xyz_seed = np.random.uniform(0, 1, size=(1,npoint,2))

xyz1 = np.concatenate([xyz1, np.zeros([1, total_points, 1])], axis=-1)
xyz2 = np.concatenate([xyz2, np.zeros([1, total_points, 1])], axis=-1)
xyz_seed = np.concatenate([xyz_seed, np.zeros([1, npoint, 1])], axis=-1)

xyz1_tf = tf.constant(xyz1, dtype=tf.float32)
xyz2_tf = tf.constant(xyz2, dtype=tf.float32)
xyz_seed_tf = tf.constant(xyz_seed, dtype=tf.float32)

radius1 = np.array([[[0.1]]])
radius2 = np.array([[[0.3]]])
radius1 = np.tile(radius1, [1, npoint, total_points])
radius2 = np.tile(radius2, [1, npoint, total_points])

radius1 = np.arange(0.1, 0.2, 0.1/5).reshape([1, -1, 1])
radius2 = np.arange(0.2, 0.3, 0.1/5).reshape([1, -1, 1])[:, ::-1, :]
radius1 = np.tile(radius1, [1, 1, total_points])
radius2 = np.tile(radius2, [1, 1, total_points])

radius1_tf = tf.constant(radius1, dtype=tf.float32)
radius2_tf = tf.constant(radius2, dtype=tf.float32)

idx1, pts1_cnt = query_ball_point_var_rad(radius1_tf, nsample, xyz1_tf, xyz_seed_tf)
idx2, pts2_cnt = query_ball_point_var_rad(radius2_tf, nsample, xyz2_tf, xyz_seed_tf)

grouped_xyz1 = group_point(xyz1_tf, idx1)
grouped_xyz2 = group_point(xyz2_tf, idx2)


with tf.Session() as sess:
    idx1_np, grouped_xyz1_np = sess.run([idx1, grouped_xyz1])
    idx2_np, grouped_xyz2_np = sess.run([idx2, grouped_xyz2])
    grouped_xyz1_np = np.reshape(grouped_xyz1_np, [-1, 3])
    grouped_xyz2_np = np.reshape(grouped_xyz2_np, [-1, 3])

import mayavi.mlab as mlab

xyz1 = np.reshape(xyz1, [-1, 3])
xyz2 = np.reshape(xyz2, [-1, 3])
mlab.points3d(grouped_xyz1_np[:, 0], grouped_xyz1_np[:, 1], grouped_xyz1_np[:, 2], color=(1,0,0), scale_factor=0.01)
mlab.points3d(grouped_xyz2_np[:, 0], grouped_xyz2_np[:, 1], grouped_xyz2_np[:, 2], color=(0,0,1), scale_factor=0.01)
mlab.points3d(xyz1[:, 0], xyz1[:, 1], xyz1[:, 2], color=(1,1,1), scale_factor=0.008)
mlab.points3d(xyz2[:, 0], xyz2[:, 1], xyz2[:, 2], color=(1,1,1), scale_factor=0.008)
mlab.view()
input()
