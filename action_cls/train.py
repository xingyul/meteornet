'''
    Single-GPU training.
'''
import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, '..', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))

import msr_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name [default: pointnet2_cls_ssg]')
parser.add_argument('--model_path', default='', help='Model checkpint path [default: ]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--data', default='', help='Data path [default: ]')
parser.add_argument('--num_frames', type=int, default=1, help='Number of frames [default: 1]')
parser.add_argument('--skip_frames', type=int, default=1, help='Skip frames [default: 1]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--command_file', default=None, help='Command file name [default: None]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_FRAME = FLAGS.num_frames
DATA = FLAGS.data
SKIP_FRAME = FLAGS.skip_frames
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
COMMAND_FILE = FLAGS.command_file

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
os.system('cp %s %s' % (COMMAND_FILE, LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


NUM_CLASSES = 20

TRAIN_DATASET = msr_dataset.Dataset(root=DATA, num_points=NUM_POINT, num_frames=NUM_FRAME, skip_frames=SKIP_FRAME, train=True)
TEST_DATASET = msr_dataset.Dataset(root=DATA, num_points=NUM_POINT, num_frames=NUM_FRAME, skip_frames=SKIP_FRAME, train=False)


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
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
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
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FRAME)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, NUM_FRAME, is_training_pl, bn_decay=bn_decay)
            MODEL.get_loss(pred, labels_pl, end_points)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses:
                tf.summary.scalar(l.op.name, l)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)

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

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'accuracy': accuracy,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def get_batch(dataset, start_idx, end_idx):
    bsize = end_idx-start_idx
    # assert(NUM_FRAME==1)
    batch_data = np.zeros((bsize, NUM_POINT * NUM_FRAME, 3))
    batch_label = np.zeros((bsize,), dtype=np.int32)
    batch_seqid = [0]*bsize
    for i in range(bsize):
        data, label, seq_id = dataset[i+start_idx]
        batch_data[i] = np.reshape(data, [-1, 3])
        batch_label[i] = label
        batch_seqid[i] = seq_id
    return batch_data, batch_label, batch_seqid

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string(str(datetime.now()))

    losses, accuracies = [], []
    num_batch = len(TRAIN_DATASET) // BATCH_SIZE
    for batch_idx in range(num_batch):
        batch_data, batch_label, _ = get_batch(TRAIN_DATASET, batch_idx*BATCH_SIZE, (batch_idx+1)*BATCH_SIZE)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val, acc_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred'], ops['accuracy']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        losses.append(loss_val)
        accuracies.append(acc_val)

        if (batch_idx+1)%50 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
    log_string('mean loss: %f' % (np.mean(losses)))
    log_string('accuracy: %f' % (np.mean(accuracies)))

    TRAIN_DATASET.shuffle()


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT*NUM_FRAME,3))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    num_batches = (len(TEST_DATASET)-1) // BATCH_SIZE +1

    total_correct = 0
    total_count = 0
    loss_sum = 0
    per_class_count = [0 for _ in range(NUM_CLASSES)]
    per_class_correct = [0 for _ in range(NUM_CLASSES)]
    per_seq_vote = {}
    per_seq_label = {}

    for batch_idx in range(num_batches):
        batch_data, batch_label, batch_seqid = get_batch(TEST_DATASET, batch_idx*BATCH_SIZE, min((batch_idx+1)*BATCH_SIZE, len(TEST_DATASET)))
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)

        def softmax(arr):
            arr = arr - np.amax(arr)
            exp_arr = np.exp(arr)
            return exp_arr / np.sum(exp_arr)

        for i in range(0, bsize):
            seq_id = batch_seqid[i]
            pred_prob = softmax(pred_val[i,:])
            if seq_id in per_seq_vote:
                per_seq_vote[seq_id] += pred_prob
            else:
                per_seq_vote[seq_id] = pred_prob
                per_seq_label[seq_id] = batch_label[i]

        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_count += bsize
        loss_sum += loss_val

        for i in range(0, bsize):
            l = batch_label[i]
            per_class_count[l] += 1
            per_class_correct[l] += (pred_val[i]==l)

    per_class_acc = [c/float(s) for c,s in zip(per_class_correct, per_class_count)]

    per_seq_pred = {k:np.argmax(v) for k,v in per_seq_vote.items()}
    per_seq_correct = [per_seq_pred[k]==per_seq_label[k] for k in per_seq_pred]
    per_seq_acc = np.mean(per_seq_correct)
    per_seq_per_class_count = [0 for _ in range(NUM_CLASSES)]
    per_seq_per_class_correct = [0 for _ in range(NUM_CLASSES)]
    for k in per_seq_pred:
        l = per_seq_label[k]
        per_seq_per_class_count[l] += 1
        per_seq_per_class_correct[l] += (per_seq_pred[k]==l)
    per_seq_per_class_acc = [c/float(s) for c,s in zip(per_seq_per_class_correct, per_seq_per_class_count)]

    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval per sample accuracy: %f'% (total_correct / float(total_count)))
    log_string('eval per sample per class acc: ' + str(per_class_acc))
    log_string('eval per sequence accuracy: %f'% (per_seq_acc))
    log_string('eval per sequence per class acc: ' + str(per_seq_per_class_acc))

    EPOCH_CNT += 1



if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
