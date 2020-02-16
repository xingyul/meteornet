

import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import cv2
import random
import copy
import glob
import multiprocessing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..', '..'))
sys.path.append(os.path.join(BASE_DIR, '..', 'utils'))

from dict_restore import DictRestore
from saver_restore import SaverRestore
import synthia_pairwise_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_concat_upsa', help='Model name [default: model_concat_upsa]')
parser.add_argument('--data', default='data_preprocessing/data_processed_maxcut_35_20k_2k_8192', help='Dataset directory [default: /data_preprocessing/data_processed_maxcut_35_20k_2k_8192]')
parser.add_argument('--model_path', default='log_train/model.ckpt', help='model checkpoint file path [default: log_train/model.ckpt]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--output_dir', default='../semantic_seg/init_flow', help='Output directory [default: ../semantic_seg/init_flow]')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

NUM_GPUS = len(FLAGS.gpu.split(','))
BATCH_SIZE = FLAGS.batch_size
assert(BATCH_SIZE % NUM_GPUS == 0)
DEVICE_BATCH_SIZE = BATCH_SIZE // NUM_GPUS

NUM_POINT = FLAGS.num_point
DATA = FLAGS.data
GPU_INDEX = FLAGS.gpu

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
MODEL_PATH = FLAGS.model_path
OUTPUT_DIR = FLAGS.output_dir
if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
os.system('cp %s %s' % (MODEL_FILE, OUTPUT_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, OUTPUT_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(OUTPUT_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

TEST_DATASET = synthia_pairwise_dataset.SceneflowDataset(DATA, npoints=NUM_POINT, train=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    with tf.Graph().as_default():
        pointclouds_pl, labels_pl, masks_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        print("--- Get model ---")
        # Get model
        MODEL.get_model(pointclouds_pl, is_training=is_training_pl, bn_decay=None, reuse=False)

        pred_gpu = []
        for i in range(NUM_GPUS):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                with tf.device('/gpu:%d'%(i)), tf.name_scope('gpu_%d'%(i)) as scope:
                    # Evenly split input data to each GPU
                    pc_batch = tf.slice(pointclouds_pl,
                        [i*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                    label_batch = tf.slice(labels_pl,
                        [i*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                    mask_batch = tf.slice(masks_pl,
                        [i*DEVICE_BATCH_SIZE,0], [DEVICE_BATCH_SIZE,-1])

                    pred_batch, end_points = MODEL.get_model(pc_batch,
                        is_training=is_training_pl, bn_decay=None, reuse=True)

                    pred_gpu.append(pred_batch)

        pred = tf.concat(pred_gpu, 0)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver.restore(sess, MODEL_PATH)
        log_string("Model restored.")

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'masks_pl': masks_pl,
               'is_training_pl': is_training_pl,
               'pred': pred}

        eval_one_epoch(sess, ops)

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT*2, 6))
    batch_label = np.zeros((bsize, NUM_POINT, 3))
    batch_mask = np.zeros((bsize, NUM_POINT))
    batch_filenames = []
    for i in range(bsize):
        pc1, color1, filename1, pc2, color2, filename2 = dataset[idxs[i+start_idx]]
        batch_data[i,:NUM_POINT,:3] = pc1
        # batch_data[i,:NUM_POINT,3:] = color1
        batch_data[i,NUM_POINT:,:3] = pc2
        # batch_data[i,NUM_POINT:,3:] = color2
        batch_filenames.append([filename1, filename2])
    return batch_data, batch_label, batch_mask, batch_filenames

def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    test_idxs = np.arange(len(TEST_DATASET))
    num_batches = (len(TEST_DATASET)+BATCH_SIZE-1) // BATCH_SIZE

    log_string(str(datetime.now()))
    log_string('---- EVALUATION ----')

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT*2, 3))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    batch_mask = np.zeros((BATCH_SIZE, NUM_POINT))
    batch_filenames = []
    # for batch_idx in range(num_batches):
    for batch_idx in range(num_batches):
        if batch_idx %20==0:
            log_string('%03d/%03d'%(batch_idx, num_batches))

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx

        cur_batch_data, cur_batch_label, cur_batch_mask, cur_batch_filenames = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)
        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_label = cur_batch_label
            batch_mask = cur_batch_mask
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_label[0:cur_batch_size] = cur_batch_label
            batch_mask[0:cur_batch_size] = cur_batch_mask
        batch_filenames = cur_batch_filenames

        base = '-'.join(os.path.basename(batch_filenames[0][0]).split('.npz')[0].split('-')[:-1])
        start_id = os.path.basename(batch_filenames[0][0]).split('.npz')[0].split('-')[-1]
        end_id = os.path.basename(batch_filenames[0][1]).split('.npz')[0].split('-')[-1]
        save_file_basename = os.path.join(OUTPUT_DIR, base + '-' + start_id + '-' + end_id + '.npz')
        if os.path.exists(save_file_basename):
            continue

        # ---------------------------------------------------------------------
        # ---- INFERENCE BELOW ----
        pred_val_sum = np.zeros((BATCH_SIZE, NUM_POINT, 3))
        SHUFFLE_TIMES = 10
        RECURRENT_TIMES = 0
        for shuffle_cnt in range(SHUFFLE_TIMES):
            shuffle_idx = np.arange(NUM_POINT)
            np.random.shuffle(shuffle_idx)
            batch_data_new = np.copy(batch_data)
            batch_data_new[:,:NUM_POINT,:] = batch_data[:,shuffle_idx,:]
            batch_data_new[:,NUM_POINT:,:] = batch_data[:,NUM_POINT+shuffle_idx,:]
            feed_dict = {ops['pointclouds_pl']: batch_data_new,
                         ops['labels_pl']: batch_label[:,shuffle_idx,:],
                         ops['masks_pl']: batch_mask[:,shuffle_idx],
                         ops['is_training_pl']: is_training}
            pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
            for recurrent_cnt in range(RECURRENT_TIMES):
                batch_data_new[:,0:NUM_POINT,0:3] += pred_val
                batch_label_new = np.copy(batch_label)
                batch_label_new[:,:,:] = batch_label - pred_val
                feed_dict = {ops['pointclouds_pl']: batch_data_new,
                             ops['labels_pl']: batch_label_new[:,shuffle_idx,:],
                             ops['masks_pl']: batch_mask[:,shuffle_idx],
                             ops['is_training_pl']: is_training}
                pred_val_new = sess.run(ops['pred'], feed_dict=feed_dict)
                pred_val += pred_val_new
            pred_val_sum[:,shuffle_idx,:] += pred_val
        # ---- INFERENCE ABOVE ----
        # ---------------------------------------------------------------------
        pred_val = pred_val_sum / float(SHUFFLE_TIMES)

        for i in range(cur_batch_size):
            base = '-'.join(os.path.basename(batch_filenames[i][0]).split('.npz')[0].split('-')[:-1])
            start_id = os.path.basename(batch_filenames[i][0]).split('.npz')[0].split('-')[-1]
            end_id = os.path.basename(batch_filenames[i][1]).split('.npz')[0].split('-')[-1]
            save_file_basename = os.path.join(OUTPUT_DIR, base + '-' + start_id + '-' + end_id + '.npz')
            np.savez_compressed(save_file_basename, flow=pred_val[i])

        DEBUG = False
        if DEBUG:
            f = open('view.pts', 'w')
            debug_idx = 0
            for i in range(NUM_POINT):
                f.write('{} {} {} {} {} {}\n'.format( \
                        batch_data[debug_idx, i, 0], \
                        batch_data[debug_idx, i, 1], \
                        batch_data[debug_idx, i, 2], \
                        1, -1, -1))
            for i in range(NUM_POINT):
                f.write('{} {} {} {} {} {}\n'.format( \
                        batch_data[debug_idx, i + NUM_POINT, 0], \
                        batch_data[debug_idx, i + NUM_POINT, 1], \
                        batch_data[debug_idx, i + NUM_POINT, 2], \
                        -1, 1, -1))
            for i in range(NUM_POINT):
                f.write('{} {} {} {} {} {}\n'.format( \
                        batch_data[debug_idx, i, 0] + pred_val[debug_idx, i, 0], \
                        batch_data[debug_idx, i, 1] + pred_val[debug_idx, i, 1], \
                        batch_data[debug_idx, i, 2] + pred_val[debug_idx, i, 2], \
                        -1, -1, 1))
            exit()


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    evaluate()
    LOG_FOUT.close()
