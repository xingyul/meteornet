'''
    Provider for duck dataset from xingyu liu
'''

import os
import os.path
import json
import numpy as np
import sys
import pickle
import copy
import psutil

from pyquaternion import Quaternion
import class_mapping

class SegDataset():
    def __init__(self, root='processed_pc', \
            filelist_name='data_prep/train_raw.txt', \
            labelweight_filename = 'data_prep/labelweights.npz', \
            npoints = 16384, num_frames=1, train=True):
        self.npoints = npoints
        self.train = train
        self.root = root
        self.num_frames = num_frames

        self.labelweights = np.load(labelweight_filename)['labelweights']

        filenames = []
        raw_txt_file = open(filelist_name, 'r')
        l = raw_txt_file.readline()
        while len(l) > 0:
            l = l.split(' ')[0]
            l = l.split('/')
            sequence_name = l[0]
            frame_id = int(l[-1].split('.')[0])

            filenames.append([sequence_name, frame_id])
            l = raw_txt_file.readline()

        filenames.sort()
        self.filenames = filenames

        ##### debug
        # self.filenames = [f for f in self.filenames if 'SYNTHIA-SEQS-01-DAWN' in f[0]]

        self.cache = {}
        self.cache_mem_usage = 0.95

    def read_data(self, sequence_name, frame_id):
        if sequence_name in self.cache:
            if frame_id in self.cache[sequence_name]:
                pc, rgb, semantic, center = self.cache[sequence_name][frame_id]
                return pc, rgb, semantic, center

        fn = os.path.join(self.root, sequence_name + '-' + str(frame_id).zfill(6) + '.npz')
        if os.path.exists(fn):
            data = np.load(fn)

            pc = data['pc']
            rgb = data['rgb']
            semantic = data['semantic']
            center = data['center']

            semantic = semantic.astype('uint8')
        else:
            pc, rgb, semantic, center = None, None, None, None

        mem = psutil.virtual_memory()
        if (mem.used / mem.total) < self.cache_mem_usage:
            if sequence_name not in self.cache:
                self.cache[sequence_name] = {}
            self.cache[sequence_name][frame_id] = (pc, rgb, semantic, center)
        return pc, rgb, semantic, center

    def read_training_data_point(self, index):
        sequence_name, frame_id = self.filenames[index]

        pcs = []
        rgbs = []
        semantics = []
        center_0 = None

        most_recent_success = -1
        for diff in range(0, self.num_frames):
            pc, rgb, semantic, center = self.read_data(sequence_name, frame_id - diff)
            if pc is None:
                pc, rgb, semantic, center = self.read_data(sequence_name, most_recent_success)
            else:
                most_recent_success = frame_id - diff

            if diff == 0:
                center_0 = center

            pcs.append(pc)
            rgbs.append(rgb)
            semantics.append(semantic)

        pc = np.stack(pcs, axis=0)
        rgb = np.stack(rgbs, axis=0)
        semantic = np.stack(semantics, axis=0)

        return pc, rgb, semantic, center_0

    def half_crop_w_context(self, half, context, pc, rgb, semantic, center):
        num_frames = pc.shape[0]
        all_idx = np.arange(pc.shape[1])
        sample_indicies_half_w_context = []
        if half == 0:
            for f in range(num_frames):
                sample_idx_half_w_context = all_idx[pc[f, :, 2] > (center[2] - context)]
                sample_indicies_half_w_context.append(sample_idx_half_w_context)
        else:
            for f in range(num_frames):
                sample_idx_half_w_context = all_idx[pc[f, :, 2] < (center[2] + context)]
                sample_indicies_half_w_context.append(sample_idx_half_w_context)

        pc_half_w_context = [pc[f, s] for f, s in enumerate(sample_indicies_half_w_context)]
        rgb_half_w_context = [rgb[f, s] for f, s in enumerate(sample_indicies_half_w_context)]
        semantic_half_w_context = [semantic[f, s] for f, s in enumerate(sample_indicies_half_w_context)]
        if half == 0:
            loss_masks = [p[:, 2] > center[2] for p in pc_half_w_context]
        else:
            loss_masks = [p[:, 2] < center[2] for p in pc_half_w_context]
        valid_pred_idx_in_full = sample_indicies_half_w_context

        return pc_half_w_context, rgb_half_w_context, semantic_half_w_context, loss_masks, valid_pred_idx_in_full

    def augment(self, pc, center):
        flip = np.random.uniform(0, 1) > 0.5
        if flip:
            pc = (pc - center)
            pc[:, 0] *= -1
            pc += center

        scale = np.random.uniform(0.8, 1.2)
        pc = (pc - center) * scale + center

        rot_axis = np.array([0, 1, 0])
        rot_angle = np.random.uniform(np.pi * 2)
        q = Quaternion(axis=rot_axis, angle=rot_angle)
        R = q.rotation_matrix

        pc = np.dot(pc - center, R) + center
        return pc

    def mask_and_label_conversion(self, semantic, loss_mask):
        labels = []
        loss_masks = []
        for i, s in enumerate(semantic):
            sem = s.astype('int32')
            label = class_mapping.index_to_label_vec_func(sem)
            loss_mask_ = (label != 12) * loss_mask[i]
            label[label == 12] = 0

            labels.append(label)
            loss_masks.append(loss_mask_)
        return labels, loss_masks

    def choice_to_num_points(self, pc, rgb, label, loss_mask, valid_pred_idx_in_full):

        # shuffle idx to change point order (change FPS behavior)
        for f in range(self.num_frames):
            idx = np.arange(pc[f].shape[0])
            choice_num = self.npoints
            if pc[f].shape[0] > choice_num:
                shuffle_idx = np.random.choice(idx, choice_num, replace=False)
            else:
                shuffle_idx = np.concatenate([np.random.choice(idx, choice_num -  idx.shape[0]), \
                        np.arange(idx.shape[0])])
            pc[f] = pc[f][shuffle_idx]
            rgb[f] = rgb[f][shuffle_idx]
            label[f] = label[f][shuffle_idx]
            loss_mask[f] = loss_mask[f][shuffle_idx]
            valid_pred_idx_in_full[f] = valid_pred_idx_in_full[f][shuffle_idx]

        pc = np.concatenate(pc, axis=0)
        rgb = np.concatenate(rgb, axis=0)
        label = np.concatenate(label, axis=0)
        loss_mask = np.concatenate(loss_mask, axis=0)
        valid_pred_idx_in_full = np.concatenate(valid_pred_idx_in_full, axis=0)

        return pc, rgb, label, loss_mask, valid_pred_idx_in_full

    def get(self, index, half=0, context=1.):

        pc, rgb, semantic, center = self.read_training_data_point(index)
        pc, rgb, semantic, loss_mask, valid_pred_idx_in_full = \
                self.half_crop_w_context(half, context, pc, rgb, semantic, center)

        label, loss_mask = self.mask_and_label_conversion(semantic, loss_mask)

        pc, rgb, label, loss_mask, valid_pred_idx_in_full = \
                self.choice_to_num_points(pc, rgb, label, loss_mask, valid_pred_idx_in_full)

        if self.train:
            pc = self.augment(pc, center)

        if self.train:
            labelweights = 1/np.log(1.2 + self.labelweights)
            # labelweights = 1 / self.labelweights
            labelweights = labelweights / labelweights.min()
        else:
            labelweights = np.ones_like(self.labelweights)

        return pc, rgb, label, labelweights, loss_mask, valid_pred_idx_in_full

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    import mayavi.mlab as mlab
    import class_mapping
    NUM_POINT = 8192
    num_frames = 2
    d = SegDataset(root='processed_pc', npoints=NUM_POINT, train=True, num_frames=num_frames)
    print(len(d))
    import time
    tic = time.time()
    point_size = 0.2
    for idx in range(200, len(d)):
        for half in [0, 1]:

            batch_data = np.zeros((NUM_POINT * num_frames, 3 + 3))
            batch_label = np.zeros((NUM_POINT * num_frames), dtype='int32')
            batch_mask = np.zeros((NUM_POINT * num_frames), dtype=np.bool)

            pc, rgb, label, labelweights, loss_mask, valid_pred_idx_in_full = d.get(idx, half)

            batch_data[:, :3] = pc
            batch_data[:, 3:] = rgb
            batch_label = label
            batch_mask = loss_mask

            batch_labelweights = labelweights[batch_label]

            batch_data = batch_data[:NUM_POINT]
            batch_label = batch_label[:NUM_POINT]
            batch_mask = batch_mask[:NUM_POINT]
            batch_labelweights = batch_labelweights[:NUM_POINT]

            mlab.figure(bgcolor=(1,1,1))

            pc_valid = batch_data[:, :3][batch_mask]
            rgb_valid = batch_data[:, 3:][batch_mask]
            label_valid = batch_label[batch_mask]
            for i in range(12):
                pc_sem = pc_valid[label_valid == i]
                color = class_mapping.index_to_color[class_mapping.label_to_index[i]]

                mlab.points3d(pc_sem[:,0], pc_sem[:,1], pc_sem[:,2], scale_factor=point_size, color=(color[0]/255,color[1]/255,color[2]/255))

            pc_non_valid = batch_data[:, :3][np.logical_not(batch_mask)]
            mlab.points3d(pc_non_valid[:,0], pc_non_valid[:,1], pc_non_valid[:,2], scale_factor=point_size, color=(0, 0, 0))

            f = open('view.pts', 'w')
            for i in range(batch_data.shape[0]):
                p = batch_data[i, :3]
                color = 2 * batch_data[i, 3:] - 1
                ##### write color
                f.write('{} {} {} {} {} {}\n'.format(p[0], p[1], p[2], color[0], color[1], color[2]))


            input()

    print(time.time() - tic)


