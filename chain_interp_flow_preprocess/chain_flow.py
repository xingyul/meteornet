

import numpy as np

def min_k(dist, k):
    '''
        dist: N1 x N2 array
        return:
            top_k_dist: N1xk, k min dists among last (N2) array
            top_k_indices: N1xk, k argmin among last (N2) array
    '''
    top_k_indices = []
    top_k_dists = []
    for i in range(k):
        base_idx = np.arange(dist.shape[0])
        argmin_idx = dist.argmin(axis=-1)
        top_k_indices.append(argmin_idx)
        top_k_dists.append(dist[base_idx, argmin_idx])
        dist[base_idx, argmin_idx] = np.float('inf')
    top_k_indices = np.stack(top_k_indices, axis=0).T
    top_k_dists = np.stack(top_k_dists, axis=0).T
    return top_k_dists, top_k_indices

def chain_flow(pc_start_flowed, pc_end, flow_at_end, k):
    '''
        pc_start_flowed: N1 x 3
        pc_end: N2 x 3
        flow_at_end: N1 x 3
    '''
    dist = np.sum(np.square(pc_start_flowed), axis=-1, keepdims=True) + \
            np.sum(np.square(pc_end.T), axis=0, keepdims=True) - 2 * np.dot(pc_start_flowed, pc_end.T)

    top_k_dists, top_k_indices = min_k(dist, k)

    top_k_dists = np.expand_dims(top_k_dists, axis=-1)
    top_k_dists_inv = 1 / top_k_dists

    return np.sum(top_k_dists_inv * flow_at_end[top_k_indices], axis=1) / np.sum(top_k_dists_inv, axis=1)


