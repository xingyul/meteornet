'''
    Provider for duck dataset from xingyu liu
'''

import os
import os.path
import json
import numpy as np
import sys
import pickle
import glob


class SceneflowDataset():
    def __init__(self, root='../semantic_seg/processed_data', npoints=2048, train=True):
        self.npoints = npoints
        self.train = train
        self.root = root
        self.datapath = glob.glob(os.path.join(self.root, '*.npz'))
        self.datapath.sort()

        self.filelist = []
        for d in self.datapath:
            base = os.path.join(os.path.dirname(d), '-'.join(os.path.basename(d).split('.npz')[0].split('-')[:-1]))
            idx = int(os.path.basename(d).split('.npz')[0].split('-')[-1])
            next_d = base + '-' + str(int(idx) + 1).zfill(6) + '.npz'
            prev_d = base + '-' + str(int(idx) - 1).zfill(6) + '.npz'
            if next_d in self.datapath:
                self.filelist.append([d, next_d])
            if prev_d in self.datapath:
                self.filelist.append([d, prev_d])

    def __getitem__(self, index):
        fn1, fn2 = self.filelist[index]
        with open(fn1, 'rb') as fp:
            data = np.load(fp)

            center1 = data['center']
            rgb1 = data['rgb']
            semantic1 = data['semantic']
            pc1 = data['pc']

        with open(fn2, 'rb') as fp:
            data = np.load(fp)

            center2 = data['center']
            rgb2 = data['rgb']
            semantic2 = data['semantic']
            pc2 = data['pc']

        return pc1, rgb1, fn1, pc2, rgb2, fn2

    def __len__(self):
        return len(self.filelist)


if __name__ == '__main__':
    import mayavi.mlab as mlab
    d = SceneflowDataset(root='../semantic_seg/processed_data', npoints=2048)
    print(len(d))
    import time
    tic = time.time()
    point_size = 0.2
    for i in range(200, 500):
        pc1, rgb1, pc2, rgb2 = d[i]
        print(pc1.shape, rgb1.shape, pc2.shape, rgb2.shape)
        print(rgb1.max())
        print(rgb1.min())
        exit()

        mlab.figure(bgcolor=(1,1,1))
        mlab.points3d(pc1[:,0], pc1[:,1], pc1[:,2], scale_factor=point_size, color=(1,0,0))
        mlab.points3d(pc2[:,0], pc2[:,1], pc2[:,2], scale_factor=point_size, color=(0,1,0))
        input()

        exit()



