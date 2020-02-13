import argparse
import math
from datetime import datetime
#import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models')) # model
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../data'))

import synthia_dataset_direct
import class_mapping

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data', default='processed_pc', help='Dataset dir [default: model_basic]')
parser.add_argument('--model', default='model_basic', help='Model name [default: model_basic]')
parser.add_argument('--model_path', default=None, help='Model path to restore [default: None]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--num_frame', type=int, default=1, help='Number of frames [default: 1]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--command_file', default=None, help='Name of command file [default: None]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--visu', type=bool, default=False, help='Whether to dump visualization results.')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
DATA = FLAGS.data
NUM_POINT = FLAGS.num_point
NUM_FRAME = FLAGS.num_frame
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
COMMAND_FILE = FLAGS.command_file

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
MODEL_PATH = FLAGS.model_path
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure
os.system('cp %s %s' % (COMMAND_FILE, LOG_DIR)) # bkp of command file
os.system('cp %s %s' % ('synthia_dataset_direct.py', LOG_DIR)) # bkp of command file
os.system('cp ../utils/net_utils.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

NUM_CLASSES = 12

TRAINVAL_DATASET = synthia_dataset_direct.SegDataset(DATA, filelist_name='data_prep/trainval_raw.txt', npoints=NUM_POINT, num_frames=NUM_FRAME, train=True)
TEST_DATASET = synthia_dataset_direct.SegDataset(DATA, filelist_name='data_prep/test_raw.txt', npoints=NUM_POINT, num_frames=NUM_FRAME, train=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, labelweights_pl, masks_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FRAME)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, NUM_FRAME, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, masks_pl, end_points, labelweights_pl)
            tf.summary.scalar('loss', loss)


            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        #sess.run(init, {is_training_pl: True})

        if (MODEL_PATH is not None) and (MODEL_PATH != 'None'):
            saver = tf.train.Saver()
            saver.restore(sess, MODEL_PATH)
            log_string('Model restored.')

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'labelweights_pl': labelweights_pl,
               'masks_pl': masks_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            log_string('learning_rate: {}'.format(sess.run(learning_rate)))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)

            # Save the variables to disk.
            if (epoch+1) % 1 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model-{}.ckpt".format(epoch)))
                log_string("Model saved in file: %s" % save_path)

            log_string('---- EPOCH %03d TEST ----'%(epoch))
            eval_one_epoch(sess, ops, test_writer, dataset=TEST_DATASET, epoch_cnt=epoch)



def get_batch(dataset, idxs, start_idx, end_idx, half=0):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT * NUM_FRAME, 3 + 3))
    batch_label = np.zeros((bsize, NUM_POINT * NUM_FRAME), dtype='int32')
    batch_mask = np.zeros((bsize, NUM_POINT * NUM_FRAME), dtype=np.bool)

    for i in range(bsize):
        pc, rgb, label, labelweights, loss_mask, valid_pred_idx_in_full = dataset.get(idxs[i+start_idx], half)

        batch_data[i, :, :3] = pc
        batch_data[i, :, 3:] = rgb
        batch_label[i] = label
        batch_mask[i] = loss_mask

    batch_labelweights = labelweights[batch_label]

    return batch_data, batch_label, batch_labelweights, batch_mask

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAINVAL_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAINVAL_DATASET) // BATCH_SIZE

    log_string(str(datetime.now()))

    loss_sum = 0
    for batch_idx in range(num_batches):
        for half in [0, 1]:
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            batch_data, batch_label, batch_labelweights, batch_mask = \
                    get_batch(TRAINVAL_DATASET, train_idxs, start_idx, end_idx, half)

            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['labels_pl']: batch_label,
                         ops['labelweights_pl']: batch_labelweights,
                         ops['masks_pl']: batch_mask,
                         ops['is_training_pl']: is_training,}

            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            loss_sum += loss_val

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10 / 2))
            loss_sum = 0

def eval_one_epoch(sess, ops, test_writer, dataset, epoch_cnt):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    test_idxs = np.arange(0, len(dataset))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (len(dataset)+BATCH_SIZE-1) // BATCH_SIZE

    loss_sum = 0
    total_correct = 0
    total_seen = 0
    total_pred_label_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_class = [0 for _ in range(NUM_CLASSES)]


    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(epoch_cnt))

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT*NUM_FRAME, 3 + 3))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT*NUM_FRAME))
    batch_mask = np.zeros((BATCH_SIZE, NUM_POINT*NUM_FRAME))
    batch_labelweights = np.zeros((BATCH_SIZE, NUM_POINT*NUM_FRAME))
    for batch_idx in range(num_batches):
        if batch_idx %20==0:
            log_string('%03d/%03d'%(batch_idx, num_batches))

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(dataset), (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        for half in [0, 1]:
            cur_batch_data, cur_batch_label, cur_batch_labelweights, cur_batch_mask = \
                    get_batch(dataset, test_idxs, start_idx, end_idx, half)
            if cur_batch_size == BATCH_SIZE:
                batch_data = cur_batch_data
                batch_label = cur_batch_label
                batch_mask = cur_batch_mask
                batch_labelweights = cur_batch_labelweights
            else:
                batch_data[0:(cur_batch_size)] = cur_batch_data
                batch_label[0:(cur_batch_size)] = cur_batch_label
                batch_mask[0:(cur_batch_size)] = cur_batch_mask
                batch_labelweights[0:(cur_batch_size)] = cur_batch_labelweights

            # ---------------------------------------------------------------------
            # ---- INFERENCE BELOW ----
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['labels_pl']: batch_label,
                         ops['labelweights_pl']: batch_labelweights,
                         ops['masks_pl']: batch_mask,
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            test_writer.add_summary(summary, step)
            # ---- INFERENCE ABOVE ----
            # ---------------------------------------------------------------------

            pred_val = np.argmax(pred_val, 2) # BxN
            cur_pred_val = pred_val[0:cur_batch_size]
            correct = np.sum((cur_pred_val == cur_batch_label) * cur_batch_mask) # evaluate only on 20 categories but not unknown
            total_correct += correct
            total_seen += np.sum(cur_batch_mask)
            if cur_batch_size == BATCH_SIZE:
                loss_sum += loss_val
            for l in range(NUM_CLASSES):
                total_pred_label_class[l] += np.sum(((cur_pred_val==l) | (cur_batch_label==l)) & cur_batch_mask)
                total_correct_class[l] += np.sum((cur_pred_val==l) & (cur_batch_label==l) & cur_batch_mask)
                total_class[l] += np.sum((cur_batch_label==l) & cur_batch_mask)

    log_string('eval mean loss: %f' % (loss_sum / float(len(dataset)/BATCH_SIZE)))

    ACCs = []
    for i in range(NUM_CLASSES):
        acc = total_correct_class[i] / float(total_class[i])
        if total_class[i] == 0:
            acc = 0
        log_string('eval acc of %s:\t %f'%(class_mapping.index_to_class[class_mapping.label_to_index[i]], acc))
        ACCs.append(acc)
    log_string('eval accuracy: %f'% (np.mean(np.array(ACCs))))

    IoUs = []
    for i in range(NUM_CLASSES):
        iou = total_correct_class[i] / float(total_pred_label_class[i])
        if total_pred_label_class[i] == 0:
            iou = 0
        log_string('eval mIoU of %s:\t %f'%(class_mapping.index_to_class[class_mapping.label_to_index[i]], iou))
        IoUs.append(iou)
    log_string('eval mIoU:\t %f'%(np.mean(np.array(IoUs))))

    return loss_sum/float(len(dataset)/BATCH_SIZE)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
