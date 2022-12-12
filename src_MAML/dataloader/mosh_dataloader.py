
'''
    file:   mosh_dataloader.py

    author: zhangxiong(1025679612@qq.com)
    date:   2018_05_09
    purpose:  load COCO 2017 keypoint dataset
'''

import sys
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio  
import os 
import glob
import numpy as np
import random
import cv2
import json
import h5py
import torch
import tqdm
import dataloader.meta.metadata as md
import pickle as pk
import math

sys.path.append('./src')
from util import calc_aabb, cut_image, flip_image, draw_lsp_14kp__bone, rectangle_intersect, get_rectangle_intersect_ratio, reflect_pose
from config import args
from timer import Clock

class mosh_dataloader(Dataset):
    def __init__(self, data_set_path, use_flip = True, flip_prob = 0.3):
        self.data_folder = data_set_path
        self.use_flip = use_flip
        self.flip_prob = flip_prob
        self.thetas = []
        self.betas = []

        self._load_data_set()
    

    def _load_data_set(self):
        metadata = md.load_h36m_metadata()
        clk = Clock()
        print('start loading mosh data.')
        
        for subject_id in list([1, 5, 6, 7, 8, 9, 11]):
            for action_id in range(2, 17):
                for subaction_id in [1, 2]:
                    for camera_id in [1, 2, 3, 4]:

                        file_prefix = metadata.sequence_mappings['S'+str(subject_id)][(str(action_id), str(subaction_id))]
                        raw_file_name = file_prefix + '.pkl'
                        file_name = file_prefix + '_cam' + str(camera_id - 1) + '_aligned.pkl'
                        file_path = 'S%d/' % subject_id + file_name
                        raw_file_path = 'S%d/' % subject_id + raw_file_name
                        
                        file_path = os.path.join(self.data_folder, file_path)
                        raw_file_path = os.path.join(self.data_folder, raw_file_path)

                        if not os.path.exists(file_path):
                            print((file_path, subject_id, action_id, subaction_id, camera_id), 'not exist')
                            continue

                        else:
                            with open(raw_file_path, 'rb') as f:
                                a = pk._Unpickler(f)
                                a.encoding = 'latin1'
                                raw_p = a.load()
                            with open(file_path, 'rb') as f:
                                a = pk._Unpickler(f)
                                a.encoding = 'latin1'
                                p = a.load()
                            assert math.ceil((raw_p['poses'].shape[0] // 4) / 5) == p['new_poses'].shape[0], (raw_p['poses'].shape, p['new_poses'].shape)

                            for k in range(len(p['new_poses'])):

                                thetas = p['new_poses'][k]
                                global_orient = thetas[:3]
                                R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
                                R_root = cv2.Rodrigues(np.array(global_orient))[0]
                                new_R_root = R_root.dot(R_mod)
                                thetas[:3] = cv2.Rodrigues(new_R_root)[0].reshape(3)
                                self.thetas.append(thetas.tolist())
                                self.betas.append(p['betas'].tolist())
                                
                                

        # anno_file_path = os.path.join(self.data_folder, 'mosh_annot.h5')
        # with h5py.File(anno_file_path) as fp:
        #     self.shapes = np.array(fp['shape'])
        #     self.poses = np.array(fp['pose'])
        print('finished load mosh data, total {} samples'.format(len(self.thetas)))
        clk.stop()

    def __len__(self):
        return len(self.thetas)

    def __getitem__(self, index):
        trival, pose, shape = np.zeros(3), self.thetas[index], self.betas[index]
        
        if self.use_flip and random.uniform(0, 1) <= self.flip_prob:#left-right reflect the pose
            pose = reflect_pose(pose)

        return {
            'theta': torch.tensor(np.concatenate((trival, pose, shape), axis = 0)).float()
        }

if __name__ == '__main__':
    # print(random.rand(1))
    mosh = mosh_dataloader('/data/share/3D/MeshTransformer/datasets/mosh_new')
    l = len(mosh)
    import time
    for _ in range(l):
        r = mosh.__getitem__(_)
        print(r)