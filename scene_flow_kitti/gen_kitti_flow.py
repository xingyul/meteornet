#!/usr/bin/python
''' Script for generating kitti scene flow data

Author: Xingyu Liu
Date: Dec 2019
'''

import numpy as np
import os
import cv2
import pickle
import sys
import scipy.optimize


import kitti_object
from camera_params import *


def transform(transform_matrix, points):
    '''
        transform_matrix is 4x4
    '''
    points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
    trans_points = np.dot(points, transform_matrix.T)
    trans_points = trans_points[:, :3] / np.reshape(trans_points[:, 3], [-1,1])
    return trans_points


def project(project_matrix, points):
    '''
        project_matrix is 3x4
    '''
    points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
    project_points = np.dot(points, project_matrix.T)
    project_points = project_points[:, :2] / np.reshape(project_points[:, 2], [-1,1])
    return project_points


def depth_pop_up_v2(project_matrix, project_points, depth):
    '''
        project_matrix is 3x4
        project_points is Nx2
        depth is N
        output is Nx3
    '''
    points = np.zeros([project_points.shape[0], 3])
    P11 = project_matrix[0,0]; P13 = project_matrix[0,2]; P14 = project_matrix[0,3];
    P22 = project_matrix[1,1]; P23 = project_matrix[1,2]; P24 = project_matrix[1,3];
    P34 = project_matrix[2,3];

    u = project_points[:, 0]
    v = project_points[:, 1]
    z = depth

    points[:, 0] = (u*(z + P34) - P13*z - P14) / P11
    points[:, 1] = (v*(z + P34) - P23*z - P24) / P22
    points[:, 2] = z
    return points


def pop_up_single_depth_map(project_matrix, depth_map, flow=None):
    """
    Pop_up_up_depth_map_up_depth ).

    Args:
        project_matrix: (todo): write your description
        depth_map: (todo): write your description
        flow: (todo): write your description
    """
    depth_map_valid = depth_map > 1e-8
    valid_depth_map = depth_map[depth_map_valid]
    xx, yy = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))

    if flow is not None:
        flow_u = flow[:, :, 2]
        flow_v = flow[:, :, 1]
        flow_u = (flow_u - 32768.) / 64.
        flow_v = (flow_v - 32768.) / 64.
        xx = xx + flow_u
        yy = yy + flow_v

    xx_depth_map_valid = xx[depth_map_valid]
    yy_depth_map_valid = yy[depth_map_valid]
    projected_points = np.stack([xx_depth_map_valid, yy_depth_map_valid], axis=1)

    poped_up_points = depth_pop_up_v2(project_matrix, projected_points, valid_depth_map)

    return poped_up_points

def find_nearest(seed, candidate_value, candidate_valid, search_range=2):
    """
    Find the nearest neighbor in a list of candidates in - place.

    Args:
        seed: (int): write your description
        candidate_value: (bool): write your description
        candidate_valid: (bool): write your description
        search_range: (str): write your description
    """

    xx, yy = np.meshgrid(np.arange(candidate_value.shape[1]), np.arange(candidate_value.shape[0]))
    return_val = np.zeros([seed.shape[0], candidate_value.shape[-1]]) * np.float('nan')

    for i, s in enumerate(seed):
        start_x = max(int(s[0])-search_range, 0)
        end_x = min(int(s[0])+search_range, candidate_value.shape[1])
        start_y = max(int(s[1])-search_range, 0)
        end_y = min(int(s[1])+search_range, candidate_value.shape[0])

        valid_in_search_range = candidate_valid[start_y:end_y, start_x:end_x]
        value_in_search_range = candidate_value[start_y:end_y, start_x:end_x]
        pos_x_in_search_range = xx[start_y:end_y, start_x:end_x] - int(s[0])
        pos_y_in_search_range = yy[start_y:end_y, start_x:end_x] - int(s[1])
        dist_in_search_range = np.square(pos_x_in_search_range) + np.square(pos_y_in_search_range)

        dist_in_search_range_flatten = np.reshape(dist_in_search_range, [-1])
        valid_in_search_range_flatten = np.reshape(valid_in_search_range, [-1])
        value_in_search_range_flatten = np.reshape(value_in_search_range, [-1, value_in_search_range.shape[-1]])

        dist_argsort = dist_in_search_range_flatten.argsort()
        for d in dist_argsort:
            if valid_in_search_range_flatten[d]:
                return_val[i] = value_in_search_range_flatten[d]

    return return_val


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_sceneflow_dir', default='sceneflow', help='kitti scene flow dir [default: sceneflow]')
    parser.add_argument('--kitti_raw_dir', default='raw', help='kitti raw dir [default: raw]')
    parser.add_argument('--output_dir', default='kitti_flow', help='output dir [default: kitti_flow]')
    FLAGS = parser.parse_args()

    kitti_sceneflow_dir = FLAGS.kitti_sceneflow_dir
    kitti_raw_dir = FLAGS.kitti_raw_dir
    output_dir = FLAGS.output_dir
    mapping_file = os.path.join(kitti_sceneflow_dir, 'devkit/mapping/train_mapping.txt')

    if not os.path.exists(output_dir):
        os.system('mkdir -p {}'.format(output_dir))

    mapping_f = open(mapping_file, 'r')
    l = mapping_f.readline()
    sf_id = 0

    while len(l) > 0:

        if len(l) == 1:
            l = mapping_f.readline()
            sf_id += 1
            continue

        raw_date, raw_sequence_id, raw_frame_id = l.split(' ')
        print(raw_date, raw_sequence_id, raw_frame_id)
        raw_frame_id = int(raw_frame_id)

        index_string = str(sf_id).zfill(6)

        disp0 = cv2.imread(os.path.join(kitti_sceneflow_dir, 'training/disp_occ_0', index_string + '_10.png'), cv2.IMREAD_UNCHANGED) / 256.0
        flow0 = cv2.imread(os.path.join(kitti_sceneflow_dir, 'training/flow_occ', index_string + '_10.png'), cv2.IMREAD_UNCHANGED)
        disp1 = cv2.imread(os.path.join(kitti_sceneflow_dir, 'training/disp_occ_1', index_string + '_10.png'), cv2.IMREAD_UNCHANGED) / 256.0

        transform_mat = P_rect_00[raw_date]
        poped_up_points_disp0 = pop_up_single_depth_map(P_rect_00[raw_date], baseline[raw_date] * focal_length[raw_date] / disp0, None)
        poped_up_points_disp1 = pop_up_single_depth_map(P_rect_00[raw_date], baseline[raw_date] * focal_length[raw_date] / disp1, flow0)
        flow = poped_up_points_disp1 - poped_up_points_disp0
        flow_map = np.reshape(flow, [disp0.shape[0], disp0.shape[1], -1])


        dataset_raw = kitti_object.kitti_raw(kitti_raw_dir)
        points1 = dataset_raw.get_lidar(raw_date, raw_sequence_id, raw_frame_id)[:, :3]
        points2 = dataset_raw.get_lidar(raw_date, raw_sequence_id, raw_frame_id + 1)[:, :3]
        points3 = dataset_raw.get_lidar(raw_date, raw_sequence_id, raw_frame_id + 2)[:, :3]
        points4 = dataset_raw.get_lidar(raw_date, raw_sequence_id, raw_frame_id + 3)[:, :3]

        points1 = points1[points1[:, 0] > 0]
        points2 = points2[points2[:, 0] > 0]
        points3 = points3[points3[:, 0] > 0]
        points4 = points4[points4[:, 0] > 0]

        project_pos1 = project(P_velo_to_img[raw_date], points1)
        fov_inds = (np.round(project_pos1[:,1]) < disp0.shape[0]) & (np.round(project_pos1[:,1]) >= 0) & \
                    (np.round(project_pos1[:,0]) < disp0.shape[1]) & (np.round(project_pos1[:,0]) >= 0)
        project_pos1 = project_pos1[fov_inds]
        points1 = points1[fov_inds]

        project_pos2 = project(P_velo_to_img[raw_date], points2)
        fov_inds = (np.round(project_pos2[:,1]) < disp0.shape[0]) & (np.round(project_pos2[:,1]) >= 0) & \
                    (np.round(project_pos2[:,0]) < disp0.shape[1]) & (np.round(project_pos2[:,0]) >= 0)
        project_pos2 = project_pos2[fov_inds]
        points2 = points2[fov_inds]

        project_pos3 = project(P_velo_to_img[raw_date], points3)
        fov_inds = (np.round(project_pos3[:,1]) < disp0.shape[0]) & (np.round(project_pos3[:,1]) >= 0) & \
                    (np.round(project_pos3[:,0]) < disp0.shape[1]) & (np.round(project_pos3[:,0]) >= 0)
        project_pos3 = project_pos3[fov_inds]
        points3 = points3[fov_inds]

        project_pos4 = project(P_velo_to_img[raw_date], points4)
        fov_inds = (np.round(project_pos4[:,1]) < disp0.shape[0]) & (np.round(project_pos4[:,1]) >= 0) & \
                    (np.round(project_pos4[:,0]) < disp0.shape[1]) & (np.round(project_pos4[:,0]) >= 0)
        project_pos4 = project_pos4[fov_inds]
        points4 = points4[fov_inds]


        project_pos1_flow = find_nearest(seed=project_pos1, candidate_value=flow_map, candidate_valid=disp0 > 0)


        points1_cam_0_coord = transform(Tr_velo_to_cam[raw_date], points1)
        points1_rect = transform(R_cam_to_rect[raw_date], points1_cam_0_coord)

        points2_cam_0_coord = transform(Tr_velo_to_cam[raw_date], points2)
        points2_rect = transform(R_cam_to_rect[raw_date], points2_cam_0_coord)

        points3_cam_0_coord = transform(Tr_velo_to_cam[raw_date], points3)
        points3_rect = transform(R_cam_to_rect[raw_date], points3_cam_0_coord)

        points4_cam_0_coord = transform(Tr_velo_to_cam[raw_date], points4)
        points4_rect = transform(R_cam_to_rect[raw_date], points4_cam_0_coord)

        np.savez_compressed(os.path.join(output_dir, index_string + '.npz'), \
                points1=points1_rect, \
                points2=points2_rect, \
                points3=points3_rect, \
                points4=points4_rect, \
                flow=project_pos1_flow, \
                mask=np.logical_not(np.isnan(project_pos1_flow[:, 0])))

        l = mapping_f.readline()
        sf_id += 1





