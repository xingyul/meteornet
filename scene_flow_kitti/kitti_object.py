''' Helper class and functions for loading KITTI objects

Modified by Xingyu Liu
Original by Charles R. Qi (https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/kitti_object.py, commit ec03a2e)

Date: Dec 2019
'''


import os
import sys
import numpy as np
import cv2
import mayavi.mlab as mlab
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'mayavi'))

import kitti_utils as utils

class_average_size = {  'Person_sitting':   [0.80202703, 0.59490991, 1.27495495], \
                        'Van':              [5.07836651, 1.90207962, 2.20659231], \
                        'Cyclist':          [1.7635464 , 0.5967732 , 1.73720344], \
                        'Truck':            [10.10907678,  2.58509141,  3.25170932], \
                        'Misc':             [3.57559096, 1.5138335 , 1.90713258], \
                        'DontCare':         [-1., -1., -1.], \
                        'Car':              [3.88395449, 1.62858987, 1.52608343], \
                        'Tram':             [16.09426614,  2.54373777,  3.52892368], \
                        'Pedestrian':       [0.84228438, 0.66018944, 1.76070649]}
class_of_interest = ['Car', 'Pedestrian', 'Cyclist']

class kitti_object(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')

    def __len__(self):
        """
        Returns the number of samples in this dataset.

        Args:
            self: (todo): write your description
        """
        return self.num_samples

    def get_image(self, idx):
        """
        Get an image.

        Args:
            self: (int): write your description
            idx: (int): write your description
        """
        assert(idx<self.num_samples)
        img_filename = os.path.join(self.image_dir, '%06d.png'%(idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        """
        Get the cidar file corresponding to - idx.

        Args:
            self: (todo): write your description
            idx: (int): write your description
        """
        assert(idx<self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin'%(idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        """
        Get calibration calibration.

        Args:
            self: (todo): write your description
            idx: (int): write your description
        """
        assert(idx<self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        """
        Get the label label.

        Args:
            self: (todo): write your description
            idx: (int): write your description
        """
        assert(idx<self.num_samples and self.split=='training')
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return utils.read_label(label_filename)

    def get_depth_map(self, idx):
        """
        Get the depth map for the given index.

        Args:
            self: (todo): write your description
            idx: (int): write your description
        """
        pass

    def get_top_down(self, idx):
        """
        Get the number of the topology.

        Args:
            self: (todo): write your description
            idx: (str): write your description
        """
        pass

class kitti_raw(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir

    def get_lidar(self, raw_date, raw_sequence_id, raw_frame_id):
        """
        Get lidar lidar sequence.

        Args:
            self: (todo): write your description
            raw_date: (todo): write your description
            raw_sequence_id: (str): write your description
            raw_frame_id: (str): write your description
        """
        raw_frame_id_filled = str(raw_frame_id).zfill(10)
        lidar_filename = os.path.join(self.root_dir, raw_date, raw_sequence_id, 'velodyne_points', 'data', raw_frame_id_filled + '.bin')

        return utils.load_velo_scan(lidar_filename)

    def get_pose(self, raw_date, raw_sequence_id, raw_frame_id):
        """
        Get a raw text file.

        Args:
            self: (todo): write your description
            raw_date: (todo): write your description
            raw_sequence_id: (str): write your description
            raw_frame_id: (str): write your description
        """
        zero_filled = str(0).zfill(10)
        gps_ref_filename = os.path.join(self.root_dir, raw_date, raw_sequence_id, 'oxts', 'data', zero_filled + '.txt')
        oxts_ref = utils.read_gps_from_txt(gps_ref_filename)

        raw_frame_id_filled = str(raw_frame_id).zfill(10)
        gps_filename = os.path.join(self.root_dir, raw_date, raw_sequence_id, 'oxts', 'data', raw_frame_id_filled + '.txt')
        oxts = utils.read_gps_from_txt(gps_filename)

        pose = get_pose_from_gps(oxts, oxts_ref)

        return pose

def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo

def get_pose_from_gps(oxts, oxts_ref):
    """
    Concatencode from spherical coordinates

    Args:
        oxts: (str): write your description
        oxts_ref: (str): write your description
    """
    scale = np.cos(oxts_ref[0] * np.pi / 180.0);

    def latlonToMercator(lat, lon, scale):
        """
        Convert from lat / longitude

        Args:
            lat: (array): write your description
            lon: (todo): write your description
            scale: (float): write your description
        """
        er = 6378137
        mx = scale * lon * np.pi * er / 180
        my = scale * er * np.log( np.tan((90+lat) * np.pi / 360) )
        return mx, my

    mx, my = latlonToMercator(oxts[0], oxts[1], scale)
    t = np.array([[mx, my, oxts[2]]])
    rx = oxts[3] # roll
    ry = oxts[4] # pitch
    rz = oxts[5] # heading
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R  = np.dot(Rz, np.dot(Ry, Rx))
    Rt = np.concatenate([R, t.T], axis=1)
    pose = np.concatenate([Rt, np.array([[0, 0, 0, 1]])], axis=0)

    return pose

def get_average_object_size(kitti_det_dir, split='training'):
    """
    Get the average size of the kitti.

    Args:
        kitti_det_dir: (str): write your description
        split: (str): write your description
    """
    dataset_det = kitti_object(kitti_det_dir, split=split)
    sizes = {}
    for i in range(dataset_det.num_samples):
        labels = dataset_det.get_label_objects(i)
        for l in labels:
            if l.type in sizes:
                sizes[l.type].append([l.l, l.w, l.h])
            else:
                sizes[l.type] = [[l.l, l.w, l.h]]
    average_sizes = {}
    for k in sizes.keys():
        average_sizes[k] = np.mean(np.array(sizes[k]), axis=0)
    return average_sizes

def regress_target_transform(regress_target, foreground_mask, search_range, delta, nbin_theta):
    '''
        regress_target: N x [hwl, xyz, delta]
        reg_target_trans: N x [hwl, resx, resy, z, resdelta]
        cls_target_trans: N x [binx, biny, bindelta]
    '''
    reg_target_trans = np.zeros([regress_target.shape[0], 7], dtype='float32')
    cls_target_trans = np.zeros([regress_target.shape[0], 3], dtype='int32')

    hwl_xyz_theta = regress_target
    reg_target_trans[:, np.array([0,1,2,5])] = hwl_xyz_theta[:, np.array([0,1,2,5])]

    xy = hwl_xyz_theta[:, 3:5]
    binx_biny = ((xy + search_range) / delta).astype('int32')
    binx_biny[np.abs(xy) > search_range] = int((2 * search_range - 1e-7) / delta)
    binx_biny[np.abs(xy) < -search_range] = 0
    resx_resy = (xy + search_range - binx_biny * delta - delta / 2) / delta

    reg_target_trans[:, np.array([3,4])] = resx_resy
    cls_target_trans[:, np.array([0,1])] = binx_biny

    theta = hwl_xyz_theta[:, -1]
    theta[theta < 0] += np.pi * 8
    theta = theta - (theta / (np.pi * 2)).astype('int32') * (np.pi * 2)
    delta_theta = np.pi * 2 / nbin_theta
    bintheta = (theta / delta_theta).astype('int32')
    restheta = (theta - delta_theta * bintheta - delta_theta / 2) / delta_theta

    reg_target_trans[:, -1] = restheta
    cls_target_trans[:, -1] = bintheta

    return reg_target_trans, cls_target_trans

def regress_target_inv_transform(reg_target_trans, cls_target_trans, search_range, delta, nbin_theta):
    '''
        reg_target_trans: N x [hwl, resx, resy, z, resdelta]
        cls_target_trans: N x [binx, biny, bindelta]
        regress_target: N x [hwl, xyz, delta]
    '''
    regress_target = np.zeros([reg_target_trans.shape[0], 7], dtype='float32')
    regress_target[:, :3] = reg_target_trans[:, :3]
    regress_target[:, 5] = reg_target_trans[:, 5]

    regress_target[:, 3:5] = cls_target_trans[:, :2] * delta + delta / 2 + reg_target_trans[:, 3:5] * delta - search_range

    delta_theta = np.pi * 2 / nbin_theta
    regress_target[:, -1] = cls_target_trans[:, -1] * delta_theta + delta_theta / 2 + reg_target_trans[:, -1] * delta_theta

    return regress_target

def visualize_scene(point_cloud, foreground_mask, foreground_ignore_mask, \
        reg_target_trans, cls_target_trans, search_range, delta, nbin_theta, class_of_interest, pred_foreground=None, nonkey_point_cloud=None):
    """
    Visualize a single scene.

    Args:
        point_cloud: (todo): write your description
        foreground_mask: (str): write your description
        foreground_ignore_mask: (bool): write your description
        reg_target_trans: (todo): write your description
        cls_target_trans: (todo): write your description
        search_range: (todo): write your description
        delta: (float): write your description
        nbin_theta: (todo): write your description
        class_of_interest: (todo): write your description
        pred_foreground: (todo): write your description
        nonkey_point_cloud: (todo): write your description
    """

    regress_target = regress_target_inv_transform(reg_target_trans, \
            cls_target_trans, search_range, delta, nbin_theta)

    if pred_foreground is not None:
        hwl_xyz_theta = regress_target[pred_foreground]
        center = point_cloud[pred_foreground] + hwl_xyz_theta[:, 3:6]
    else:
        hwl_xyz_theta = regress_target[foreground_mask, :]
        center = point_cloud[foreground_mask] + hwl_xyz_theta[:, 3:6]

    foreground = point_cloud[foreground_mask]
    ignore = point_cloud[foreground_ignore_mask]
    background = point_cloud[np.logical_not(foreground_mask) & np.logical_not(foreground_ignore_mask)]

    mlab.figure(bgcolor=(0, 0, 0))
    mlab.clf()
    axes=np.array([
        [2.,0.,0.,0.],
        [0.,2.,0.,0.],
        [0.,0.,2.,0.],
        ],dtype=np.float64)

    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None)
    mlab.points3d(foreground[:,0], foreground[:,1], foreground[:,2], color=(1, 0, 0), scale_factor=0.1)
    mlab.points3d(ignore[:,0], ignore[:,1], ignore[:,2], color=(0, 0, 1), scale_factor=0.1)
    mlab.points3d(background[:,0], background[:,1], background[:,2], color=(1, 1, 0), scale_factor=0.1)

    if nonkey_point_cloud is not None:
        mlab.points3d(nonkey_point_cloud[:,0], nonkey_point_cloud[:,1], nonkey_point_cloud[:,2], color=(0, 1, 1), scale_factor=0.1)

    for _ in range(30):
        if hwl_xyz_theta.shape[0] > 0:
            i = np.random.randint(hwl_xyz_theta.shape[0])
            corners = utils.compute_label_box_corners(hwl_xyz_theta[i][:3] + class_average_size[class_of_interest], center[i], hwl_xyz_theta[i][-1])
            utils.draw_box3d_mlab(mlab, corners, (0, 1, 0))

    mlab.view()
    input()

if __name__ == '__main__':
    average_sizes = get_average_object_size('/raid/datasets/kitti/3d_det', split='training')
    print(average_sizes['Car'])
    print(average_sizes['Cyclist'])
    print(average_sizes['Pedestrian'])
    print(average_sizes)

