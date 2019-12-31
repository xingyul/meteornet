

import struct
import numpy as np
import os
import cv2
import multiprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='Depth', help='Depth directory for input [default: Depth]')
parser.add_argument('--output_dir', default='processed_data', help='Output processed data directory [default: processed_data]')
parser.add_argument('--num_cpu', type=int, default=8, help='Number of CPUs to use in parallel [default: 8]')
FLAGS = parser.parse_args()

input_dir = FLAGS.input_dir
output_dir = FLAGS.output_dir
num_cpu = FLAGS.num_cpu

debug = False

def read_bin(filename):
    f = open(filename, 'rb')

    num_frames = f.read(4)
    num_frames = struct.unpack("<L", num_frames)[0]

    width = f.read(4)
    width = struct.unpack("<L", width)[0]

    height = f.read(4)
    height = struct.unpack("<L", height)[0]

    depth = f.read()
    depth = struct.unpack('{}I'.format(num_frames*height*width), depth)
    depth = np.array(depth)
    depth = np.reshape(depth, [num_frames, height, width])
    return depth

def process_one_file(filename):

    depth = read_bin(filename)

    focal = 280

    xx, yy = np.meshgrid(np.arange(depth.shape[2]), np.arange(depth.shape[1]))

    point_clouds = []
    for d in range(depth.shape[0]):
        depth_map = depth[d]

        depth_min = depth_map[depth_map > 0].min()
        depth_map = depth_map

        x = xx[depth_map > 0]
        y = yy[depth_map > 0]
        z = depth_map[depth_map > 0]
        x = (x - depth_map.shape[1] / 2) / focal * z
        y = (y - depth_map.shape[0] / 2) / focal * z

        points = np.stack([x, y, z], axis=-1)

        point_clouds.append(points)

        if debug:
            f = open('view.pts', 'w')

            for i in range(x.shape[0]):
                f.write('{} {} {} {} {} {}\n'.format(x[i], y[i], z[i], 1, -1, -1))
            exit()

            cv2.imshow('f', ((depth_map - depth_min) / (depth_map.max() - depth_min) * 255).astype('uint8'))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    np.savez_compressed(os.path.join(output_dir, os.path.basename(filename).split('.')[0] + '.npz'), point_clouds=point_clouds)

if not os.path.exists(output_dir):
    os.system('mkdir -p {}'.format(output_dir))

files = os.listdir(input_dir)

pool = multiprocessing.Pool(num_cpu)

for input_file in files:
    print(input_file)
    # process_one_file(os.path.join(input_dir, input_file))
    pool.apply_async(process_one_file, (os.path.join(input_dir, input_file), ))

pool.close()
pool.join()


