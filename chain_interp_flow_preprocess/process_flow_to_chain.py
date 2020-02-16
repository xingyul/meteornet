

import numpy as np
import chain_flow
import os
import glob
import multiprocessing
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--original_pc_dir', default='../semantic_seg_synthia/processed_pc', help='Original point cloud directory [default: ../semantic_seg_synthia/processed_pc]')
parser.add_argument('--pairwise_flow_dir', default='../semantic_seg_synthia/init_flow', help='Initial scene flow directory [default: ../semantic_seg_synthia/init_flow]')
parser.add_argument('--chained_flowed_dir', default='../semantic_seg_synthia/chained_flow', help='Chained scene flow directory, output [default: ../semantic_seg_synthia/chained_flow]')
parser.add_argument('--maximum_frame_diff', type=int, default=3, help='Maximum frame difference [default: 3]')
parser.add_argument('--num_threads', type=int, default=4, help='Number of threads [default: 3]')
parser.add_argument('--num_nn', type=int, default=2, help='Number of Nearest Neighbors in Interpolation [default: 2]')
FLAGS = parser.parse_args()


original_pc_dir = FLAGS.original_pc_dir
pairwise_flow_dir = FLAGS.pairwise_flow_dir
chained_flowed_dir = FLAGS.chained_flowed_dir
maximum_frame_diff = FLAGS.maximum_frame_diff
num_threads = FLAGS.num_threads
num_nn = FLAGS.num_nn

if not os.path.exists(chained_flowed_dir):
    os.system('mkdir -p {}'.format(chained_flowed_dir))

def process_one_chain(k, pc_filenames, init_flow_filenames):
    '''
    input:
        k: number the nearest neighbors to choose
        pc_filenames: a list of original point cloud npz filenames
        init_flow_filenames: a list of initial pairwise flow npz filenames, should be same length as pc_filenames
    output:
        chained_flowed: a list of end points relative to original pc, should be same length as pc_filenames
    '''
    original_pc = []
    for i in pc_filenames:
        data = np.load(i)
        original_pc.append(data['pc'])

    pred = []
    for i in init_flow_filenames:
        data = np.load(i)
        pred.append(data['flow'])

    if len(pred) == 0:
        return

    chained_flowed = []
    chained_flowed.append(original_pc[0] + pred[0])

    for i in range(len(pred)-1):
        relay_flow = chain_flow.chain_flow(chained_flowed[i], original_pc[i+1], pred[i+1], k=k)
        chained_flowed.append(chained_flowed[i] + relay_flow)

    base = '-'.join(os.path.basename(pc_filenames[0]).split('-')[:-1])
    base_idx = os.path.basename(pc_filenames[0]).split('.npz')[0].split('-')[-1]

    for i in range(len(chained_flowed)):
        end_idx = pc_filenames[i+1].split('.npz')[0].split('-')[-1]
        save_filename = base + '-' + str(base_idx).zfill(6) + '-' +  end_idx.zfill(6) + '.npz'
        save_filename = os.path.join(chained_flowed_dir, save_filename)
        np.savez_compressed(save_filename, chained_flow=chained_flowed[i] - original_pc[0])

npz_files = os.listdir(original_pc_dir)
npz_files = [n for n in npz_files if n.endswith('.npz')]
npz_files.sort()

pool = multiprocessing.Pool(num_threads)

for idx, n in enumerate(npz_files):
    base = '-'.join(n.split('.npz')[0].split('-')[:-1])
    idx = int(n.split('.npz')[0].split('-')[-1])

    pc_filenames = [os.path.join(original_pc_dir, \
            base + '-' + str(idx+d).zfill(6) + '.npz') for d in range(maximum_frame_diff)]
    init_flow_filenames = [os.path.join(pairwise_flow_dir, \
            base + '-' + str(idx+d).zfill(6) + '-' + str(idx+d+1).zfill(6) + '.npz') for d in range(maximum_frame_diff)]
    pc_filenames = [p for p in pc_filenames if os.path.exists(p)]
    init_flow_filenames = [i for i in init_flow_filenames if os.path.exists(i)]

    if len(init_flow_filenames) > 0:
        print([os.path.basename(i) for i in init_flow_filenames])
        # process_one_chain(k=num_nn, pc_filenames=pc_filenames, init_flow_filenames=init_flow_filenames)
        pool.apply_async(process_one_chain, (num_nn, pc_filenames, init_flow_filenames))
        pass

    pc_filenames = [os.path.join(original_pc_dir, \
            base + '-' + str(idx+d).zfill(6) + '.npz') for d in reversed(range(-maximum_frame_diff+1, 1))]
    init_flow_filenames = [os.path.join(pairwise_flow_dir, \
            base + '-' + str(idx+d).zfill(6) + '-' + str(idx+d-1).zfill(6) + '.npz') for d in reversed(range(-maximum_frame_diff+1, 1))]
    pc_filenames = [p for p in pc_filenames if os.path.exists(p)]
    init_flow_filenames = [i for i in init_flow_filenames if os.path.exists(i)]

    if len(init_flow_filenames) > 0:
        print([os.path.basename(i) for i in init_flow_filenames])
        # process_one_chain(k=num_nn, pc_filenames=pc_filenames, init_flow_filenames=init_flow_filenames)
        pool.apply_async(process_one_chain, (num_nn, pc_filenames, init_flow_filenames))
        pass

pool.close()
pool.join()

exit()


