""" Helper methods for loading and parsing KITTI data.

Modified by Xingyu Liu
Original by Charles R. Qi (https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/kitti_util.py, commit ec03a2e)

Date: Dec 2019
"""

from __future__ import print_function

import numpy as np
import cv2
import os
import copy
from pyquaternion import Quaternion

class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line):
        """
        Initialize a yaw file

        Args:
            self: (todo): write your description
            label_file_line: (str): write your description
        """
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0] # 'Car', 'Pedestrian', ...
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])

        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
        self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        self.centered_xyz = None
        self.yaw_sin = None
        self.yaw_cos = None
        self.corners = None
        self.orientation = None
        self.theta = None

    def print_object(self):
        """
        Print the object

        Args:
            self: (todo): write your description
        """
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
            (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
            (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
            (self.h, self.w, self.l))
        if self.centered_xyz is not None:
            print('3d bbox centered location, xyz: (%f, %f, %f)' % \
                (self.centered_xyz[0], self.centered_xyz[1], self.centered_xyz[2]))
            print('3d bbox orientation: (%f, %f, %f), yaw sin: %f, yaw cos: %f' % \
                (self.orientation[0], self.orientation[1], self.orientation[2], \
                self.yaw_sin, self.yaw_cos))


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, calib_filepath, from_video=False):
        """
        Initialize calibration

        Args:
            self: (todo): write your description
            calib_filepath: (str): write your description
            from_video: (int): write your description
        """
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3,4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3,4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rigid transform from Velodyne coord to reference camera coord
        self.I2V = calibs['Tr_imu_to_velo']
        self.I2V = np.reshape(self.I2V, [3,4])
        self.V2I = inverse_rigid_trans(self.I2V)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0,[3,3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0,2]
        self.c_v = self.P[1,2]
        self.f_u = self.P[0,0]
        self.f_v = self.P[1,1]
        self.b_x = self.P[0,3]/(-self.f_u) # relative
        self.b_y = self.P[1,3]/(-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3,4))
        Tr_velo_to_cam[0:3,0:3] = np.reshape(velo2cam['R'], [3,3])
        Tr_velo_to_cam[:,3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Output: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_imu(self, pts_3d_velo):
        """
        Convert a 2d to the velocity velocity

        Args:
            self: (todo): write your description
            pts_3d_velo: (array): write your description
        """
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2I))

    def project_imu_to_velo(self, pts_3d_velo):
        """
        Return the velocity of the velocity in the velocity velocity

        Args:
            self: (todo): write your description
            pts_3d_velo: (todo): write your description
        """
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.I2V))

    def project_velo_to_ref(self, pts_3d_velo):
        """
        Convert a reference velocity velocity coordinates in - plane velocity

        Args:
            self: (todo): write your description
            pts_3d_velo: (todo): write your description
        """
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        """
        Convert a 3d to the velocity velocity

        Args:
            self: (todo): write your description
            pts_3d_ref: (todo): write your description
        """
        pts_3d_ref = self.cart2hom(pts_3d_ref) # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        """
        Projects the velocity to rectangular velocity

        Args:
            self: (todo): write your description
            pts_3d_velo: (array): write your description
        """
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
        pts_3d_rect = np.zeros((n,3))
        pts_3d_rect[:,0] = x
        pts_3d_rect[:,1] = y
        pts_3d_rect[:,2] = uv_depth[:,2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        """
        Convert the velocity velocity velocity coordinates to velocity coordinates

        Args:
            self: (todo): write your description
            uv_depth: (int): write your description
        """
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

def read_label(label_filename):
    """
    Read label_filename label file.

    Args:
        label_filename: (str): write your description
    """
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects

def load_image(img_filename):
    """
    Load an image from file.

    Args:
        img_filename: (str): write your description
    """
    return cv2.imread(img_filename)

def load_velo_scan(velo_filename):
    """
    Load velocities from a veloc.

    Args:
        velo_filename: (str): write your description
    """
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n,1))))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]


def compute_box_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    #print corners_3d.shape
    corners_3d[0,:] = corners_3d[0,:] + obj.t[0]
    corners_3d[1,:] = corners_3d[1,:] + obj.t[1]
    corners_3d[2,:] = corners_3d[2,:] + obj.t[2]
    #print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2,:]<0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    #print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def compute_orientation_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    '''

    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l],[0,0],[0,0]])

    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0,:] = orientation_3d[0,:] + obj.t[0]
    orientation_3d[1,:] = orientation_3d[1,:] + obj.t[1]
    orientation_3d[2,:] = orientation_3d[2,:] + obj.t[2]

    # vector behind image plane?
    if np.any(orientation_3d[2,:]<0.1):
      orientation_2d = None
      return orientation_2d, np.transpose(orientation_3d)

    # project orientation into the image plane
    orientation_2d = project_to_image(np.transpose(orientation_3d), P)
    return orientation_2d, np.transpose(orientation_3d)

def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       # use LINE_AA for opencv3
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
    return image


def read_corr_from_txt(filename):
    """
    Reads the correlation file.

    Args:
        filename: (str): write your description
    """
    f = open(filename, 'r')
    det_to_raw_corr = []
    l = f.readline()
    while len(l) > 0:
        _, date, sequence, raw_idx = l.rstrip().split(' ')
        det_to_raw_corr.append([date, sequence, raw_idx])
        l = f.readline()
    f.close()
    return det_to_raw_corr

def project_homo(matrix, points):
    '''
        matrix: 4x4
        points: Nx3
    '''
    points_ = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
    points_ = np.dot(matrix, points_.T).T
    points_ = points_[:, :3] / np.expand_dims(points_[:, -1], axis=1)
    return points_

def read_gps_from_txt(filename):
    """
    Read gps file with gps file.

    Args:
        filename: (str): write your description
    """
    f = open(filename, 'r')
    l = f.readline()
    l = l.rstrip()
    l = [float(i) for i in l.split(' ')]
    f.close()
    return l

def check_inside_bbox_deprecated(points, corners, roof_side_toler=0., all_toler=0.):
    '''
        points: [N, 3]
        box: [3, 8]
        roof_side_toler: a number

        return: [N] bool
    '''
    N = points.shape[0]
    points = np.expand_dims(points, 1)
    corners = np.expand_dims(corners.T, 0)

    orientation = corners[0, 1] - corners[0, 2]
    normal_orien = np.array([-orientation[1], orientation[0], 0])

    corners = copy.deepcopy(corners)
    corners[0, np.array([0, 1, 4, 5])] += (roof_side_toler + all_toler) * np.expand_dims(orientation, axis=0)
    corners[0, np.array([2, 3, 6, 7])] -= (roof_side_toler + all_toler) * np.expand_dims(orientation, axis=0)
    corners[0, np.array([0, 3, 4, 7])] += (roof_side_toler + all_toler) * np.expand_dims(normal_orien, axis=0)
    corners[0, np.array([1, 2, 5, 6])] -= (roof_side_toler + all_toler) * np.expand_dims(normal_orien, axis=0)
    corners[0, np.array([4, 5, 6, 7]), 2] += (roof_side_toler + all_toler)
    corners[0, np.array([0, 1, 2, 3]), 2] -= all_toler
    d = np.ones([N, 3])

    plane_x_1 = corners[:, np.array([0,1,2]), :]
    try:
        dx_1 = np.linalg.solve(plane_x_1 - points, d)
    except:
        dx_1 = np.linalg.solve(plane_x_1 - points + np.random.uniform(-1e-8, 1e-8), d)

    plane_x_2 = corners[:, np.array([4,5,6]), :]
    try:
        dx_2 = np.linalg.solve(plane_x_2 - points, d)
    except:
        dx_2 = np.linalg.solve(plane_x_2 - points + np.random.uniform(-1e-8, 1e-8), d)

    dot_x = np.sum(dx_1 * dx_2, axis=1)

    plane_y_1 = corners[:, np.array([0,3,4]), :]
    try:
        dy_1 = np.linalg.solve(plane_y_1 - points, d)
    except:
        dy_1 = np.linalg.solve(plane_y_1 - points + np.random.uniform(-1e-8, 1e-8), d)

    plane_y_2 = corners[:, np.array([1,2,5]), :]
    try:
        dy_2 = np.linalg.solve(plane_y_2 - points, d)
    except:
        dy_2 = np.linalg.solve(plane_y_2 - points + np.random.uniform(-1e-8, 1e-8), d)

    dot_y = np.sum(dy_1 * dy_2, axis=1)

    plane_z_1 = corners[:, np.array([0,1,4]), :]
    try:
        dz_1 = np.linalg.solve(plane_z_1 - points, d)
    except:
        dz_1 = np.linalg.solve(plane_z_1 - points + np.random.uniform(-1e-8, 1e-8), d)

    plane_z_2 = corners[:, np.array([2,3,6]), :]
    try:
        dz_2 = np.linalg.solve(plane_z_2 - points, d)
    except:
        dz_2 = np.linalg.solve(plane_z_2 - points + np.random.uniform(-1e-8, 1e-8), d)

    dot_z = np.sum(dz_1 * dz_2, axis=1)

    return (dot_x < 0) * (dot_y < 0) * (dot_z < 0)

def check_inside_bbox_velo_coord(points, corners, roof_side_toler=0., all_toler=0.):
    '''
	points: [N, 3]
	corners: [3, 8]
	roof_side_toler: a number

	return: [N] bool
    '''
    N = points.shape[0]
    corners = copy.deepcopy(corners)
    corners = corners.T

    orientation = corners[1] - corners[2]
    orientation = orientation / np.linalg.norm(orientation)
    normal_orien = np.array([-orientation[1], orientation[0], 0])

    corners[np.array([0, 1, 4, 5])] += (roof_side_toler + all_toler) * np.expand_dims(orientation, axis=0)
    corners[np.array([2, 3, 6, 7])] -= (roof_side_toler + all_toler) * np.expand_dims(orientation, axis=0)
    corners[np.array([0, 3, 4, 7])] += (roof_side_toler + all_toler) * np.expand_dims(normal_orien, axis=0)
    corners[np.array([1, 2, 5, 6])] -= (roof_side_toler + all_toler) * np.expand_dims(normal_orien, axis=0)
    corners[np.array([4, 5, 6, 7]), 2] += (roof_side_toler + all_toler)
    corners[np.array([0, 1, 2, 3]), 2] -= all_toler

    points_vec = points - corners[0]
    pos_vec = corners[np.array([1,3,4])] - corners[np.array([0,0,0])]
    mask1 = np.all(np.dot(points_vec, pos_vec.T) > 0, axis=-1)

    points_vec = points - corners[6]
    pos_vec = corners[np.array([2,5,7])] - corners[np.array([6,6,6])]
    mask2 = np.all(np.dot(points_vec, pos_vec.T) > 0, axis=-1)

    return mask1 & mask2

def check_inside_bbox(points, corners, roof_side_toler=0., all_toler=0.):
    '''
	points: [N, 3]
	corners: [8, 3]
	roof_side_toler: a number

	return: [N] bool
    '''
    N = points.shape[0]
    corners = copy.deepcopy(corners)

    orientation = corners[1] - corners[0]
    orientation = orientation / np.linalg.norm(orientation)
    normal_orien = np.array([-orientation[2], 0, orientation[0]])

    corners[np.array([1, 2, 5, 6])] += (roof_side_toler + all_toler) * np.expand_dims(orientation, axis=0)
    corners[np.array([0, 3, 4, 7])] -= (roof_side_toler + all_toler) * np.expand_dims(orientation, axis=0)
    corners[np.array([0, 1, 4, 5])] += (roof_side_toler + all_toler) * np.expand_dims(normal_orien, axis=0)
    corners[np.array([2, 3, 6, 7])] -= (roof_side_toler + all_toler) * np.expand_dims(normal_orien, axis=0)
    corners[np.array([4, 5, 6, 7]), 1] -= (roof_side_toler + all_toler)
    corners[np.array([0, 1, 2, 3]), 1] += all_toler

    points_vec = points - corners[0]
    pos_vec = corners[np.array([1,3,4])] - corners[np.array([0,0,0])]
    mask1 = np.all(np.dot(points_vec, pos_vec.T) > 0, axis=-1)

    points_vec = points - corners[6]
    pos_vec = corners[np.array([2,5,7])] - corners[np.array([6,6,6])]
    mask2 = np.all(np.dot(points_vec, pos_vec.T) > 0, axis=-1)

    return mask1 & mask2


def compute_label_box_corners(lwh, center_xyz, theta):
    '''
        coordinate is left x, down y, front z
    '''
    l, w, h = lwh

    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]

    R = roty(theta)
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))

    corners_3d += np.expand_dims(center_xyz, -1)
    return corners_3d

def draw_box3d_mlab(mlab, corners, color, line_width=2.0):
    """
    Draws a box3d box.

    Args:
        mlab: (todo): write your description
        corners: (todo): write your description
        color: (todo): write your description
        line_width: (float): write your description
    """
    mlab.plot3d([corners[0,0], corners[0,1]], [corners[1,0], corners[1,1]], [corners[2,0], corners[2,1]], color=color, tube_radius=None, line_width=line_width)
    mlab.plot3d([corners[0,1], corners[0,2]], [corners[1,1], corners[1,2]], [corners[2,1], corners[2,2]], color=color, tube_radius=None, line_width=line_width)
    mlab.plot3d([corners[0,2], corners[0,3]], [corners[1,2], corners[1,3]], [corners[2,2], corners[2,3]], color=color, tube_radius=None, line_width=line_width)
    mlab.plot3d([corners[0,3], corners[0,0]], [corners[1,3], corners[1,0]], [corners[2,3], corners[2,0]], color=color, tube_radius=None, line_width=line_width)

    mlab.plot3d([corners[0,4], corners[0,5]], [corners[1,4], corners[1,5]], [corners[2,4], corners[2,5]], color=color, tube_radius=None, line_width=line_width)
    mlab.plot3d([corners[0,5], corners[0,6]], [corners[1,5], corners[1,6]], [corners[2,5], corners[2,6]], color=color, tube_radius=None, line_width=line_width)
    mlab.plot3d([corners[0,6], corners[0,7]], [corners[1,6], corners[1,7]], [corners[2,6], corners[2,7]], color=color, tube_radius=None, line_width=line_width)
    mlab.plot3d([corners[0,7], corners[0,4]], [corners[1,7], corners[1,4]], [corners[2,7], corners[2,4]], color=color, tube_radius=None, line_width=line_width)

    mlab.plot3d([corners[0,0], corners[0,4]], [corners[1,0], corners[1,4]], [corners[2,0], corners[2,4]], color=color, tube_radius=None, line_width=line_width)
    mlab.plot3d([corners[0,1], corners[0,5]], [corners[1,1], corners[1,5]], [corners[2,1], corners[2,5]], color=color, tube_radius=None, line_width=line_width)
    mlab.plot3d([corners[0,2], corners[0,6]], [corners[1,2], corners[1,6]], [corners[2,2], corners[2,6]], color=color, tube_radius=None, line_width=line_width)
    mlab.plot3d([corners[0,3], corners[0,7]], [corners[1,3], corners[1,7]], [corners[2,3], corners[2,7]], color=color, tube_radius=None, line_width=line_width)

if __name__ == '__main__':
    points = np.random.uniform(-1.5, 1.5, [10, 3])
    corners = np.array([ \
        [1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1]], dtype='float32')
    print(points)
    mask = check_inside_bbox_deprecated(points, corners, roof_side_toler=0., all_toler=0.)
    print(mask)
    mask = check_inside_bbox(points, corners, roof_side_toler=0., all_toler=0.)
    print(mask)


