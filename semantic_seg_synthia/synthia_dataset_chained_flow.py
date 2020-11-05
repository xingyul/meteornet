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
            chained_flow_root='chained_flow',
            filelist_name='data_prep/train_raw.txt', \
            labelweight_filename = 'data_prep/labelweights.npz', \
            npoints = 16384, num_frames=1, train=True):
        """
        Initialize the dataset.

        Args:
            self: (todo): write your description
            root: (str): write your description
            chained_flow_root: (bool): write your description
            filelist_name: (str): write your description
            labelweight_filename: (str): write your description
            npoints: (int): write your description
            num_frames: (int): write your description
            train: (todo): write your description
        """
        self.npoints = npoints
        self.train = train
        self.root = root
        self.chained_flow_root = chained_flow_root
        self.num_frames = num_frames
        self.num_max_nonkey = num_frames - 1

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
        """
        Read a single frame from a single frame.

        Args:
            self: (todo): write your description
            sequence_name: (str): write your description
            frame_id: (str): write your description
        """
        if sequence_name in self.cache:
            if frame_id in self.cache[sequence_name]:
                pc, rgb, semantic, chained_flowed, center = self.cache[sequence_name][frame_id]
                return pc, rgb, semantic, chained_flowed, center

        fn = os.path.join(self.root, sequence_name + '-' + str(frame_id).zfill(6) + '.npz')
        if os.path.exists(fn):
            data = np.load(fn)

            pc = data['pc']
            rgb = data['rgb']
            semantic = data['semantic']
            center = data['center']
            chained_flow = []

            ##### read flow
            basename_split = os.path.basename(fn).split('.npz')[0].split('-')
            for f in range(-self.num_max_nonkey, self.num_max_nonkey+1):
                if f != 0:
                    new_basename = '-'.join(basename_split + [str(int(basename_split[-1]) + f).zfill(6)]) + '.npz'
                    chained_flow_fn = os.path.join(self.chained_flow_root, new_basename)
                    if os.path.exists(chained_flow_fn):
                        chained_flow_data = np.load(chained_flow_fn)['chained_flow']
                    else:
                        chained_flow_data = None
                else:
                    chained_flow_data = pc
                chained_flow.append(chained_flow_data)
            for i in range(self.num_max_nonkey+1, self.num_max_nonkey*2 + 1):
                if chained_flow[i] is None:
                    chained_flow[i] = chained_flow[i-1]
                else:
                    chained_flow[i] = chained_flow[i-1] + chained_flow[i]
            for i in range(self.num_max_nonkey-1, -1, -1):
                if chained_flow[i] is None:
                    chained_flow[i] = chained_flow[i+1]
                else:
                    chained_flow[i] = chained_flow[i+1] + chained_flow[i]
            chained_flowed = np.stack(chained_flow, axis=-2)

            semantic = semantic.astype('uint8')
        else:
            pc, rgb, semantic, chained_flowed, center = None, None, None, None, None

        mem = psutil.virtual_memory()
        if (mem.used / mem.total) < self.cache_mem_usage:
            if sequence_name not in self.cache:
                self.cache[sequence_name] = {}
            self.cache[sequence_name][frame_id] = (pc, rgb, semantic, chained_flowed, center)
        return pc, rgb, semantic, chained_flowed, center

    def read_training_data_point(self, index):
        """
        Reads ---------- index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        sequence_name, frame_id = self.filenames[index]

        pcs = []
        rgbs = []
        semantics = []
        chained_floweds_raw = []
        center_0 = None
        exist_frame_id = []

        most_recent_success = -1
        for diff in range(0, self.num_frames):
            ##### combination of (sequence_name, frame_id) is guaranteed to exist, therefore diff=0 will not return none
            pc, rgb, semantic, chained_flowed, center = self.read_data(sequence_name, frame_id - diff)
            if pc is None:
                pc, rgb, semantic, chained_flowed, center = self.read_data(sequence_name, most_recent_success)
            else:
                most_recent_success = frame_id - diff
            exist_frame_id.append(most_recent_success)

            if diff == 0:
                center_0 = center

            pcs.append(pc)
            rgbs.append(rgb)
            semantics.append(semantic)
            chained_floweds_raw.append(chained_flowed)
        exist_frame_id.reverse()

        ##### resolve the cases for repeated frames, at the start of the sequence in the dataset
        chained_floweds_list = []
        for f_dest in range(self.num_frames):
            chained_floweds = []
            for f_src in range(self.num_frames):
                f_diff = exist_frame_id[f_dest] - exist_frame_id[f_src]
                chained_floweds.append(chained_floweds_raw[f_dest][:, f_diff + self.num_max_nonkey])
            chained_floweds = np.stack(chained_floweds, axis=-2)
            chained_floweds_list.append(chained_floweds)

        pc = np.stack(pcs, axis=0)
        rgb = np.stack(rgbs, axis=0)
        semantic = np.stack(semantics, axis=0)
        chained_flowed = np.stack(chained_floweds_list, axis=0)

        return pc, rgb, semantic, chained_flowed, center_0

    def half_crop_w_context(self, half, context, pc, rgb, semantic, chained_flowed, center):
        """
        Create a crop context of a crop.

        Args:
            self: (todo): write your description
            half: (todo): write your description
            context: (todo): write your description
            pc: (todo): write your description
            rgb: (todo): write your description
            semantic: (todo): write your description
            chained_flowed: (bool): write your description
            center: (float): write your description
        """
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
        chained_flowed_half_w_context = [chained_flowed[f, s] for f, s in enumerate(sample_indicies_half_w_context)]
        if half == 0:
            loss_masks = [p[:, 2] > center[2] for p in pc_half_w_context]
        else:
            loss_masks = [p[:, 2] < center[2] for p in pc_half_w_context]
        valid_pred_idx_in_full = sample_indicies_half_w_context

        return pc_half_w_context, rgb_half_w_context, semantic_half_w_context, chained_flowed_half_w_context, \
                loss_masks, valid_pred_idx_in_full

    def augment(self, pc, chained_flowed, center):
        """
        Generate a rotation matrix.

        Args:
            self: (todo): write your description
            pc: (todo): write your description
            chained_flowed: (bool): write your description
            center: (float): write your description
        """
        flip = np.random.uniform(0, 1) > 0.5
        if flip:
            pc = (pc - center)
            pc[:, 0] *= -1
            pc += center

            chained_flowed = (chained_flowed - center)
            chained_flowed[:, :, 0] *= -1
            chained_flowed += center

        scale = np.random.uniform(0.8, 1.2)
        pc = (pc - center) * scale + center
        chained_flowed = (chained_flowed - center) * scale + center

        rot_axis = np.array([0, 1, 0])
        rot_angle = np.random.uniform(np.pi * 2)
        q = Quaternion(axis=rot_axis, angle=rot_angle)
        R = q.rotation_matrix

        pc = np.dot(pc - center, R) + center
        chained_flowed = np.dot(chained_flowed - center, R) + center
        return pc, chained_flowed

    def mask_and_label_conversion(self, semantic, loss_mask):
        """
        Generate mask mask and mask.

        Args:
            self: (todo): write your description
            semantic: (todo): write your description
            loss_mask: (todo): write your description
        """
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

    def choice_to_num_points(self, pc, rgb, label, chained_flowed, loss_mask, valid_pred_idx_in_full):
        """
        Generate_to_to_flowed.

        Args:
            self: (todo): write your description
            pc: (todo): write your description
            rgb: (todo): write your description
            label: (str): write your description
            chained_flowed: (todo): write your description
            loss_mask: (todo): write your description
            valid_pred_idx_in_full: (str): write your description
        """

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
            chained_flowed[f] = chained_flowed[f][shuffle_idx]
            label[f] = label[f][shuffle_idx]
            loss_mask[f] = loss_mask[f][shuffle_idx]
            valid_pred_idx_in_full[f] = valid_pred_idx_in_full[f][shuffle_idx]

        pc = np.concatenate(pc, axis=0)
        rgb = np.concatenate(rgb, axis=0)
        label = np.concatenate(label, axis=0)
        chained_flowed = np.concatenate(chained_flowed, axis=0)
        loss_mask = np.concatenate(loss_mask, axis=0)
        valid_pred_idx_in_full = np.concatenate(valid_pred_idx_in_full, axis=0)

        return pc, rgb, label, chained_flowed, loss_mask, valid_pred_idx_in_full

    def get(self, index, half=0, context=1.):
        """
        Get the best match.

        Args:
            self: (todo): write your description
            index: (int): write your description
            half: (int): write your description
            context: (dict): write your description
        """

        pc, rgb, semantic, chained_flowed, center = self.read_training_data_point(index)
        pc, rgb, semantic, chained_flowed, loss_mask, valid_pred_idx_in_full = \
                self.half_crop_w_context(half, context, pc, rgb, semantic, chained_flowed, center)

        label, loss_mask = self.mask_and_label_conversion(semantic, loss_mask)

        pc, rgb, label, chained_flowed, loss_mask, valid_pred_idx_in_full = \
                self.choice_to_num_points(pc, rgb, label, chained_flowed, loss_mask, valid_pred_idx_in_full)

        if self.train:
            pc, chained_flowed = self.augment(pc, chained_flowed, center)

        if self.train:
            labelweights = 1/np.log(1.2 + self.labelweights)
            # labelweights = 1 / self.labelweights
            labelweights = labelweights / labelweights.min()
        else:
            labelweights = np.ones_like(self.labelweights)

        return pc, rgb, label, chained_flowed, labelweights, loss_mask, valid_pred_idx_in_full

    def __len__(self):
        """
        Returns the number of filenames.

        Args:
            self: (todo): write your description
        """
        return len(self.filenames)


if __name__ == '__main__':
    import mayavi.mlab as mlab
    import class_mapping
    NUM_POINT = 8192
    num_frames = 3
    d = SegDataset(root='processed_pc', chained_flow_root='chained_flow', npoints=NUM_POINT, train=True, num_frames=num_frames)
    print(len(d))
    import time
    tic = time.time()
    point_size = 0.2
    for idx in range(200, len(d)):
        for half in [0, 1]:
            print(d.filenames[idx])

            batch_data = np.zeros((NUM_POINT * num_frames, 3 + 3))
            batch_chained_flowed = np.zeros((NUM_POINT * num_frames, 3))
            batch_label = np.zeros((NUM_POINT * num_frames), dtype='int32')
            batch_mask = np.zeros((NUM_POINT * num_frames), dtype=np.bool)

            pc, rgb, label, chained_flowed, labelweights, loss_mask, valid_pred_idx_in_full = d.get(idx, half)

            batch_data[:, :3] = pc
            batch_data[:, 3:] = rgb
            batch_chained_flowed = chained_flowed
            batch_label = label
            batch_mask = loss_mask

            print(batch_data[0*NUM_POINT:1*NUM_POINT, :3] - batch_chained_flowed[0*NUM_POINT:1*NUM_POINT, 0])
            print(batch_data[1*NUM_POINT:2*NUM_POINT, :3] - batch_chained_flowed[1*NUM_POINT:2*NUM_POINT, 1])
            print(batch_data[2*NUM_POINT:3*NUM_POINT, :3] - batch_chained_flowed[2*NUM_POINT:3*NUM_POINT, 2])

            batch_labelweights = labelweights[batch_label]

            ##### select only the first frame, for viz
            batch_data = batch_data[:NUM_POINT]
            batch_label = batch_label[:NUM_POINT]
            batch_chained_flowed = batch_chained_flowed[:NUM_POINT]
            batch_mask = batch_mask[:NUM_POINT]
            batch_labelweights = batch_labelweights[:NUM_POINT]

            ##### mlab viz, with semantic
            mlab.figure(bgcolor=(1,1,1))
            pc_valid = batch_data[:, :3][batch_mask]
            rgb_valid = batch_data[:, 3:][batch_mask]
            label_valid = batch_label[batch_mask]
            chained_flowed_valid = batch_chained_flowed[batch_mask]
            for i in range(12):
                pc_sem = pc_valid[label_valid == i]
                color = class_mapping.index_to_color[class_mapping.label_to_index[i]]

                mlab.points3d(pc_sem[:,0], pc_sem[:,1], pc_sem[:,2], scale_factor=point_size, color=(color[0]/255,color[1]/255,color[2]/255))

            pc_non_valid = batch_data[:, :3][np.logical_not(batch_mask)]
            mlab.points3d(pc_non_valid[:,0], pc_non_valid[:,1], pc_non_valid[:,2], scale_factor=point_size, color=(0, 0, 0))

            color = np.array([[1,0,0], [1,1,0], [0,1,0], [0,1,1], [0,0,1]])
            fwrite = open('view.pts', 'w')
            for i in range(batch_data.shape[0]):
                # p = batch_data[i, :3]
                for f in range(0, num_frames):
                    p = batch_chained_flowed[i,f]
                    fwrite.write('{} {} {} {} {} {}\n'.format(p[0], p[1], p[2], color[f,0], color[f,1], color[f,2]))

            input()

    print(time.time() - tic)


