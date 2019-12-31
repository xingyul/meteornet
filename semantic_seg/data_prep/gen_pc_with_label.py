


import numpy as np
import cv2
import os
import sys
import tensorflow as tf
import glob
import argparse

import class_mapping
sys.path.append('../../tf_ops/sampling')
from tf_sampling import gather_point, farthest_point_sample

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0,1,2,3', help='GPU to use [default: GPU 0,1,2,3]')
parser.add_argument('--data_root', default='/raid/datasets/synthia', help='Unprocessed data dir [default: /raid/datasets/synthia]')
parser.add_argument('--output_dir', default='../processed_pc', help='Output data dir [default: ../processed_pc]')
parser.add_argument('--camera_name', default='Stereo_Left', help='Left/right camera to use [default: Stereo_Left]')
parser.add_argument('--npoint', type=int, default=32768, help='Number of points in the full scene [default: 32768]')
parser.add_argument('--downsample_rate', type=float, default=4, help='Downsample rate first [default: 4]')
parser.add_argument('--debug', type=int, default=0, help='Debug option [default: 0]')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

NUM_GPUS = len(FLAGS.gpu.split(','))
DATA_ROOT = FLAGS.data_root
OUTPUT_DIR = FLAGS.output_dir
CAMERA_NAME = FLAGS.camera_name
BATCH_SIZE = NUM_GPUS
NPOINT = FLAGS.npoint
DOWNSAMPLE_RATE = FLAGS.downsample_rate
DEBUG = FLAGS.debug

DEBUG = bool(DEBUG)

if not os.path.exists(OUTPUT_DIR):
    os.system('mkdir -p {}'.format(OUTPUT_DIR))
os.system('cp {} {}'.format(__file__, OUTPUT_DIR))

def get_intrinsics(filename):
    f = open(filename, 'r')

    focal_length = f.readline()
    focal_length = float(focal_length)

    f.readline()
    principal_point_x = f.readline()
    f.readline()
    principal_point_y = f.readline()

    principal_point_x = float(principal_point_x)
    principal_point_y = float(principal_point_y)

    return focal_length, principal_point_x, principal_point_y


def get_global_pose(filename):
    f = open(filename, 'r')
    line = f.readline()

    matrix = np.array([float(l) for l in line.split(' ')])
    matrix = np.reshape(matrix, [4, 4]).T
    return matrix

def pc_transform(pc, matrix):
    '''
        pc: nx3
        matrix: 4x4
    '''
    pc_ = np.concatenate([pc, np.ones([pc.shape[0], 1])], axis=-1)

    pc_ = np.dot(pc_, matrix.T)
    pc_ = pc_[:, :3] / np.expand_dims(pc_[:, -1], -1)
    return pc_

def get_pc(depth_filename, rgb_filename, semantic_filename, camera_intrinsic, global_pose):
    depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
    depth = depth[:, :, 0]
    depth = depth.astype('float32') / 100

    rgb = cv2.imread(rgb_filename)[:, :, ::-1]

    semantic = cv2.imread(semantic_filename, cv2.IMREAD_UNCHANGED)
    semantic = semantic[:, :, -1]

    xx, yy = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    x = -(xx - camera_intrinsic[1]) / camera_intrinsic[0] * depth
    y = (yy - camera_intrinsic[2]) / camera_intrinsic[0] * depth
    z = depth

    ##### transform from (x right, y down, z front) to (z front)

    pc = np.stack([x, y, z], axis=-1)
    pc = np.reshape(pc, [-1, 3])

    pc = pc_transform(pc, global_pose)

    rgb = np.reshape(rgb, [-1, 3])
    semantic = np.reshape(semantic, [-1])

    return pc, rgb, semantic

def get_full_scene(data_root, sequence_name, frame_id, camera_name='Stereo_Left'):
    camera_intrinsic = get_intrinsics(os.path.join(data_root, sequence_name, 'CameraParams', 'intrinsics.txt'))

    pcs = []
    rgb_semantics = []
    for view in ['Omni_B', 'Omni_F', 'Omni_L', 'Omni_R']:
        global_pose = get_global_pose(os.path.join(data_root, sequence_name, 'CameraParams', camera_name, view, str(frame_id).zfill(6) + '.txt'))

        global_pose[:3, 3] *= -1

        pc, rgb, semantic = get_pc( \
                os.path.join(data_root, sequence_name, 'Depth', camera_name, view, str(frame_id).zfill(6) + '.png'), \
                os.path.join(data_root, sequence_name, 'RGB', camera_name, view, str(frame_id).zfill(6) + '.png'), \
                os.path.join(data_root, sequence_name, 'GT', 'LABELS', camera_name, view, str(frame_id).zfill(6) + '.png'), \
                camera_intrinsic, global_pose)

        pc = np.reshape(pc, [-1, 3])
        rgb = np.reshape(rgb, [-1, 3])
        semantic = np.reshape(semantic, [-1, 1])
        rgb_semantic = np.concatenate([rgb, semantic], axis=-1).astype('float32')

        pcs.append(pc)
        rgb_semantics.append(rgb_semantic)
    pc = np.concatenate(pcs, axis=-2)
    rgb_semantic = np.concatenate(rgb_semantics, axis=-2)

    center = global_pose[:3, 3]
    chebyshev_dist_to_center = np.max(np.abs(pc - center), axis=-1)

    sample_flag = chebyshev_dist_to_center < 25
    pc = pc[sample_flag]
    rgb_semantic = rgb_semantic[sample_flag]

    sample_flag = (rgb_semantic[:, -1].round().astype('int32') != 1) & \
                  (rgb_semantic[:, -1].round().astype('int32') != 13) & \
                  (rgb_semantic[:, -1].round().astype('int32') != 14)
    pc = pc[sample_flag]
    rgb_semantic = rgb_semantic[sample_flag]

    return pc, rgb_semantic, center

def get_tf_sess_pl(npoint, batch_size, num_gpu):

    pc_placeholder = tf.placeholder(tf.float32, shape=[batch_size, None, 3])
    feature_placeholder = tf.placeholder(tf.float32, shape=[batch_size, None, 4])

    device_batch_size = batch_size // num_gpu
    new_xyz_gpu = []
    new_feature_gpu = []

    for i in range(num_gpu):
        with tf.device('/gpu:%d'%(i)), tf.name_scope('gpu_%d'%(i)) as scope:
            pc_batch = tf.slice(pc_placeholder,
                    [i*device_batch_size,0,0], [device_batch_size,-1,-1])
            feature_batch = tf.slice(feature_placeholder,
                    [i*device_batch_size,0,0], [device_batch_size,-1,-1])

            sample_idx = farthest_point_sample(npoint, pc_batch)
            new_xyz = gather_point(pc_batch, sample_idx)
            new_feature_part_1 = gather_point(feature_batch[:, :, :3], sample_idx)
            new_feature_part_2 = gather_point(feature_batch[:, :, -3:], sample_idx)
            new_feature = tf.concat([new_feature_part_1, tf.expand_dims(new_feature_part_2[:, :, -1], axis=-1)], axis=-1)

            new_xyz_gpu.append(new_xyz)
            new_feature_gpu.append(new_feature)

    new_xyz = tf.concat(new_xyz_gpu, 0)
    new_feature = tf.concat(new_feature_gpu, 0)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    return sess, new_xyz, new_feature, pc_placeholder, feature_placeholder

def get_batch(start_idx, filenames, batch_size, data_root, camera_name):
    pcs = []
    rgb_semantics = []
    centers = []

    cur_batch_size = min(len(filenames), start_idx + batch_size) - start_idx
    for b in range(cur_batch_size):
        sequence_name, frame_id = filenames[start_idx + b]
        pc, rgb_semantic, center = get_full_scene(data_root=data_root, sequence_name=sequence_name, frame_id=frame_id, camera_name=camera_name)

        pc_norm = np.linalg.norm(pc - center, axis=-1)
        sample_idx = np.random.choice(pc_norm.shape[0], int(pc_norm.shape[0] / DOWNSAMPLE_RATE), p=pc_norm/pc_norm.sum(), replace=False)
        pc = pc[sample_idx]
        rgb_semantic = rgb_semantic[sample_idx]

        pcs.append(pc)
        rgb_semantics.append(rgb_semantic)
        centers.append(center)

    if cur_batch_size < batch_size:
        for b in range(batch_size - cur_batch_size):
            pcs.append(pc)
            rgb_semantics.append(rgb_semantic)

    max_npoint_in_batch = max([p.shape[0] for p in pcs])
    for i in range(len(pcs)):
        print(pcs[i].shape, max_npoint_in_batch)
        rest_part_idx = np.random.randint(0, pcs[i].shape[0], size=max_npoint_in_batch - pcs[i].shape[0])
        pcs[i] = np.concatenate([pcs[i], pcs[i][rest_part_idx]], axis=0)
        rgb_semantics[i] = np.concatenate([rgb_semantics[i], rgb_semantics[i][rest_part_idx]], axis=0)
    pc = np.stack(pcs, 0)
    rgb_semantic = np.stack(rgb_semantics, 0)
    center = np.stack(centers, 0)

    return pc, rgb_semantic, center, cur_batch_size


if DEBUG:
    f = open('view.pts', 'w')

train_filename = 'train_raw.txt'
val_filename = 'val_raw.txt'
test_filename = 'test_raw.txt'

filenames = []
for filename in [train_filename, val_filename, test_filename]:
    raw_txt_file = open(filename, 'r')
    l = raw_txt_file.readline()
    while len(l) > 0:
        l = l.split(' ')[0]
        l = l.split('/')
        sequence_name = l[0]
        frame_id = int(l[-1].split('.')[0])

        synthia_filename = os.path.join(DATA_ROOT, sequence_name, 'Depth', CAMERA_NAME, 'Omni_B', str(frame_id).zfill(6) + '.png')

        filenames.append([sequence_name, frame_id])

        if not os.path.exists(synthia_filename):
            print(synthia_filename)

        l = raw_txt_file.readline()

filenames.sort()

file_dirs = glob.glob(os.path.join(DATA_ROOT, 'SYNTHIA-SEQS-*'))
file_dirs = [f for f in file_dirs if os.path.isdir(f)]

filenames = []
for f in file_dirs:
    filenames += glob.glob(os.path.join(f, 'Depth/Stereo_Left/Omni_B/', '*'))
filenames = [[f.split('/')[-5], str(f.split('/')[-1].split('.')[0])] for f in filenames ]
print(len(filenames))


with tf.Graph().as_default():
    with tf.device('/cpu:0'):
        sess, new_xyz, new_feature, pc_placeholder, feature_placeholder = get_tf_sess_pl(npoint=NPOINT, batch_size=BATCH_SIZE, num_gpu=NUM_GPUS)

        for start_idx in range(0, len(filenames), BATCH_SIZE):
            print(filenames[start_idx])
            pc, rgb_semantic, center, cur_batch_size = get_batch(start_idx, filenames, BATCH_SIZE, DATA_ROOT, CAMERA_NAME)

            new_pc, new_rgb_semantic = sess.run([new_xyz, new_feature], feed_dict={pc_placeholder: pc, feature_placeholder: rgb_semantic})
            new_pc = new_pc[:cur_batch_size, :, :3]
            new_rgb = new_rgb_semantic[:cur_batch_size, :, :3]
            new_semantic = new_rgb_semantic[:cur_batch_size, :, 3]

            new_semantic = np.round(new_semantic).astype('int32')
            new_rgb /= 255.

            if not DEBUG:
                for b in range(cur_batch_size):
                    sequence_name, frame_id = filenames[start_idx + b]
                    np.savez_compressed(os.path.join(OUTPUT_DIR, sequence_name + '-' + str(frame_id).zfill(6) + '.npz'), \
                            pc=new_pc[b],
                            rgb=new_rgb[b],
                            semantic=new_semantic[b],
                            center=center[b])

            if DEBUG:
                for frame in range(2):
                    for i in range(new_pc[frame].shape[0]):
                        p = new_pc[frame][i]
                        color =  2 * new_rgb[frame][i] - 1
                        semantic_color = class_mapping.index_to_color[new_semantic[frame][i]] / 255.
                        # semantic_color = 2 * new_semantic[frame][i] / 255. - 1
                        ##### write color
                        # f.write('{} {} {} {} {} {}\n'.format(p[0], p[1], p[2], color[0], color[1], color[2]))
                        f.write('{} {} {} {} {} {}\n'.format(p[0], p[1], p[2], semantic_color[0], semantic_color[1], semantic_color[2]))
                        # f.write('{} {} {} {} {} {}\n'.format(p[0], p[1], p[2], 2 * frame - 1, 2 * frame -1, -1))
                exit()


