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

class Dataset():
    def __init__(self, \
            root='/scr1/mengyuan/ICCV-data/MSR_processed', \
            num_points = 8192, \
            num_frames=2, skip_frames=1, \
            train=True):
        self.num_points = num_points
        self.num_frames = num_frames
        self.skip_frames = skip_frames # sample frames i, i+skip_frames, i+2*skip_frames, ...
        self.train = train
        self.root = root
        self.datapath = os.listdir(self.root)
        if train:
            self.datapath = [d for d in self.datapath if int(d.split('_')[1].split('s')[1]) <= 5]
        else:
            self.datapath = [d for d in self.datapath if int(d.split('_')[1].split('s')[1]) > 5]
        self.datapath = [d.split('.')[0] for d in self.datapath]
        self.data = []
        self.label = []
        self.index_map = []
        self.load_data()
        self.shuffle()

    def load_data(self): # takes about 5G memory to load
        for i,file in enumerate(self.datapath):
            result = np.load(os.path.join(self.root, file+'.npz'))
            self.data.append(result['point_clouds'])
            self.label.append(int(file.split('_')[0][1:])-1)
            nframes = result['point_clouds'].shape[0]
            for t in range(0, nframes-self.skip_frames*(self.num_frames-1), self.skip_frames):
                self.index_map.append((i,t))

    def shuffle(self):
        self.indices = np.arange(len(self.index_map))
        if self.train:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        id, t = self.index_map[self.indices[idx]]
        points = [self.data[id][t+i*self.skip_frames] for i in range(self.num_frames)]
        for i,p in enumerate(points):
            if p.shape[0] > self.num_points:
                index = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                index = np.random.choice(p.shape[0], size=residue, replace=False)
                index = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [index], axis=0)
            points[i] = p[index, :]
        points = np.array(points)

        if self.train:
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            points = points * scales

        points = points / 300

        return points, self.label[id], id

    def __len__(self):
        return len(self.index_map)


if __name__ == '__main__':
    d = Dataset(num_points=8192)
    print(len(d))
    import time
    tic = time.time()
    point_size = 0.2
    for i in range(100):
        points = d[i]
        print(points.shape)
    print(time.time() - tic)


