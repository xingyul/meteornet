import tensorflow as tf
import numpy as np
from tf_grouping import query_ball_point, query_ball_point_var_rad, group_point

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class GroupPointTest(tf.test.TestCase):
  def test(self):
      """
      Return the test is set

      Args:
          self: (todo): write your description
      """
    pass

  def test_grad(self):
      """
      Perform the gradient for the gradient.

      Args:
          self: (todo): write your description
      """
    with tf.device('/gpu:0'):
      points = tf.constant(np.random.random((1,128,16)).astype('float32'))
      print(points)
      xyz1 = tf.constant(np.random.random((1,128,3)).astype('float32'))
      xyz2 = tf.constant(np.random.random((1,8,3)).astype('float32'))
      radius = 0.3
      nsample = 32
      idx, pts_cnt = query_ball_point(radius, nsample, xyz1, xyz2)
      grouped_points = group_point(points, idx)
      print(grouped_points)

    with self.test_session():
      print("---- Going to compute gradient error")
      err = tf.test.compute_gradient_error(points, (1,128,16), grouped_points, (1,8,32,16))
      print(err)
      self.assertLess(err, 1e-4)

if __name__=='__main__':
  tf.test.main()
