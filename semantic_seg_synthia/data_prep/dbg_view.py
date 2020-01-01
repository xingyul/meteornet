




import numpy as np
import os
import mayavi.mlab as mlab

import class_mapping

f = open('view.pts', 'w')

DATA_DIR = '/datasets/synthia/data'

sequence_name = 'SYNTHIA-SEQS-04-SUMMER'
frame_id = 0

data_file = np.load(os.path.join(DATA_DIR, sequence_name + '-' + str(frame_id).zfill(6) + '.npz'))

pc = data_file['pc']
rgb = data_file['rgb']
semantic = data_file['semantic']
center = data_file['center']


ver_lane = False
if ver_lane: # verify lane points rgb are local maximums
    lane_rgb = rgb[semantic == 12]
    lane_pc = pc[semantic == 12]
    non_lane_rgb = rgb[semantic != 12]
    non_lane_pc = pc[semantic != 12]

    for i, p in enumerate(lane_pc):
        dist = np.linalg.norm(non_lane_pc - p, axis=-1)
        top_non_lane_idx = np.argsort(dist)[:5]
        print(lane_rgb[i])
        print(non_lane_rgb[top_non_lane_idx])
        print('')

print(pc.shape)

label = class_mapping.index_to_label_vec_func(semantic)

point_size = 0.2
mlab.figure(bgcolor=(1,1,1))
for i in range(12):
    pc_sem = pc[label == i]
    color = class_mapping.index_to_color[class_mapping.label_to_index[i]]

    mlab.points3d(pc_sem[:,0], pc_sem[:,1], pc_sem[:,2], scale_factor=point_size, color=(color[0]/255,color[1]/255,color[2]/255))
input()

f.write('{} {} {} {} {} {}\n'.format(center[0], center[1], center[2], 1, 1, -1))

for i in range(pc.shape[0]):
    p = pc[i]
    color = 2 * rgb[i] - 1
    semantic_color = class_mapping.index_to_color[semantic[i]] / 255.
    ##### write color
    f.write('{} {} {} {} {} {}\n'.format(p[0], p[1], p[2], color[0], color[1], color[2]))
    # f.write('{} {} {} {} {} {}\n'.format(p[0], p[1], p[2], semantic_color[0], semantic_color[1], semantic_color[2]))
    # f.write('{} {} {} {} {} {}\n'.format(p[0], p[1], p[2], 2 * frame - 1, 2 * frame -1, -1))


