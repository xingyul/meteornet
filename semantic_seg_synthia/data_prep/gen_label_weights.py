




import numpy as np
import os
import glob
import argparse

import class_mapping

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='../processed_data', help='Data dir [default: .]')
FLAGS = parser.parse_args()

data_root = FLAGS.data_root

npz_files = glob.glob(os.path.join(data_root, '*npz'))

labelweights = np.zeros(12, dtype='int64')

for j, n in enumerate(npz_files):
    print(j)
    data = np.load(os.path.join(data_root, n))

    semantic = data['semantic']
    label = class_mapping.index_to_label_vec_func(semantic)

    mask = label != 12

    for i in range(12):
        labelweights[i] += np.sum((label == i) & mask)

labelweights = labelweights.astype(np.float128)
labelweights = labelweights/np.sum(labelweights)
np.savez_compressed(os.path.join('.', 'labelweights.npz'), labelweights=labelweights)


