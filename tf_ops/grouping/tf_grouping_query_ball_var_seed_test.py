
import tensorflow as tf
import numpy as np
import os

from tf_grouping import query_ball_point, query_ball_point_var_rad, query_ball_point_var_rad_var_seed, group_point

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

np.random.seed(2)
batch_size = 5
total_points = 16384
npoint = 2
nsample = 4096
nframe = 4

xyz1 = np.random.uniform(0, 1, size=(batch_size, total_points, 2))
for i in range(nframe):
    xyz1[:, (total_points//nframe)*i:(total_points//nframe)*(i+1), 0] = \
            xyz1[:, (total_points//nframe)*i:(total_points//nframe)*(i+1), 0] / nframe + i / nframe
xyz1_time = np.concatenate([np.ones((batch_size, total_points // nframe), dtype='int32') * i for i in range(nframe)], axis=-1)
xyz2_seed = np.random.uniform(0.4, 0.1, size=(batch_size, npoint, 2))

xyz1 = np.concatenate([xyz1, np.zeros([batch_size, total_points, 1])], axis=-1)
xyz2_seed = np.concatenate([xyz2_seed, np.zeros([batch_size, npoint, 1])], axis=-1)

xyz2_seed = np.expand_dims(xyz2_seed, -2)
motion = np.array([0.28, 0.05, 0]).reshape([1, 1, -1])
motion = np.stack([motion * i for i in range(nframe)], axis=-2)
xyz2_seed = xyz2_seed + motion

xyz1_tf = tf.constant(xyz1, dtype=tf.float32)
xyz1_time_tf = tf.constant(xyz1_time, dtype=tf.float32)
xyz2_tf = tf.constant(xyz2_seed, dtype=tf.float32)

radius1 = np.arange(0.06, 0.12, 0.06/npoint).reshape([1, -1, 1])
radius1 = np.tile(radius1, [batch_size, 1, total_points])

radius1_tf = tf.constant(radius1, dtype=tf.float32)

idx1, pts1_cnt = query_ball_point_var_rad_var_seed(radius1_tf, nsample, xyz1_tf, xyz1_time_tf, xyz2_tf)

grouped_xyz1 = group_point(xyz1_tf, idx1)

with tf.Session() as sess:
    idx1_np, grouped_xyz1_np = sess.run([idx1, grouped_xyz1])

import mayavi.mlab as mlab

##### visualization

viz_batch_idx = 1

time = xyz1_time[viz_batch_idx, idx1_np[viz_batch_idx]]
xyz1_time_ = xyz1_time[viz_batch_idx]
grouped_xyz1_np = grouped_xyz1_np[viz_batch_idx]
xyz2_seed = xyz2_seed[viz_batch_idx]

colors = [
          [1, 0, 0], # red
          [1, 1, 0], # yellow
          [0, 1, 0], # green
          [0, 0, 1], # blue
         ]

xyz1_ = xyz1[viz_batch_idx]
xyz1 = np.reshape(xyz1, [-1, 3])

#### show time background
for t in range(nframe):
    sub_group = (xyz1_ - np.array([0, 0, 0.1]))[xyz1_time_ == t]
    mlab.points3d(sub_group[:, 0], sub_group[:, 1], sub_group[:, 2], color=(colors[t][0], colors[t][1], colors[t][2]), scale_factor=0.01)

mlab.points3d(xyz1[:, 0], xyz1[:, 1], xyz1[:, 2], color=(1,1,1), scale_factor=0.005)
for t in range(nframe):
    sub_group = grouped_xyz1_np[time == t]
    mlab.points3d(sub_group[:, 0], sub_group[:, 1], sub_group[:, 2], color=(colors[t][0], colors[t][1], colors[t][2]), scale_factor=0.01)
mlab.points3d(xyz2_seed[:,0,0], xyz2_seed[:,0,1], xyz2_seed[:,0,2], color=(1,0,1), scale_factor=0.01)


motion = motion[0,0,1]
for f in range(nframe):
    for n in range(npoint):
        mlab.quiver3d(xyz2_seed[n,f,0], xyz2_seed[n,f,1], xyz2_seed[n,f,2], motion[0], motion[1], motion[2], line_width=3, scale_factor=1, color=(1,0,1))

mlab.view()
input()



