import pickle
import numpy as np

with open('/home/SENSETIME/zengwang/codes/expose/data/MANO_SMPLX_vertex_ids.pkl', 'rb') as f:
    hand_data = pickle.load(f)

face = np.load('/home/SENSETIME/zengwang/codes/expose/data/SMPL-X__FLAME_vertex_ids.npy')


left_hand = hand_data['left_hand']
right_hand = hand_data['right_hand']

N = 10475
body_flag = np.ones(N)
body_flag[left_hand] = 0
body_flag[right_hand] = 0
body_flag[face] = 0
body_vertex = np.nonzero(body_flag)[0]
np.save('../data/body_vertex_ids.npy', body_vertex)
t = 0
