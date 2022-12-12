import os
import numpy as np
import torch

from dataloader.utils.comm import get_world_size
from dataloader.datasets.human_mesh_tsv import (MeshTSVDataset, MeshTSVYamlDataset)




def build_dataset(yaml_file, is_train=True, scale_factor=1):
    print("YAML Infor: ")
    print(yaml_file)
    # if not op.isfile(yaml_file):
    #     yaml_file = op.join(args.data_dir, yaml_file)
    #     # code.interact(local=locals())
    #     assert op.isfile(yaml_file)
    return MeshTSVYamlDataset(yaml_file, is_train, False, scale_factor)


def hum36m_dataloader(yaml_file, is_train=True, scale_factor=1):

    dataset = build_dataset(yaml_file, is_train=is_train, scale_factor=scale_factor)
    
    
    # data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=True)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, pin_memory=True)
    
    return dataset


train_yaml_path = '/data/share/3D/MeshTransformer/datasets/human3.6m/train.smpl.p1.yaml'
datasets = hum36m_dataloader(train_yaml_path)
root_path = '/data/share/3D/MeshTransformer/datasets/human3.6m/tasks'


J24_NAME = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder',
'L_Elbow','L_Wrist','Neck','Top_of_Head','Pelvis','Thorax','Spine','Jaw','Head','Nose','L_Eye','R_Eye','L_Ear','R_Ear')
J24_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]


class_list = []
for data in datasets:
    label = data[0]
    img = data[1]
    anno = data[2]
    
    # img/S1_Discussion_1
    class_name = label.split('.')[0].split('/')[1].split('_')[1]
    if class_name not in class_list:
        class_list.append(class_name)
        class_dir_path = os.path.join(root_path, class_name + f'_{class_list.index(class_name)}')
        if not (os.path.exists(class_dir_path)):
            os.mkdir(class_dir_path)      
        
    pose = anno['pose']
    shape = anno['betas']
    data_3d_j3d = anno['joints_3d']
    has_3d_joints = anno['has_3d_joints']
    data_3d_j2d = anno['joints_2d'][J24_TO_J14,:]
    has_2d_joints = anno['has_2d_joints']
    has_smpl = anno['has_smpl']
    
    gt_3d_pelvis = data_3d_j3d[J24_NAME.index('Pelvis'),:3]
    data_3d_j3d = data_3d_j3d[J24_TO_J14,:]
    data_3d_j3d[:,:3] = data_3d_j3d[:,:3] - gt_3d_pelvis[None, :]
    
    class_num = class_list.index(class_name)
    
    name1, name2 = label.split('/')[1].split('.')[0], label.split('/')[1].split('.')[1]
    save_path = os.path.join(class_dir_path, f'{name1}_{name2}.pth')
    torch.save({'pose': pose,
                'shape': shape,
                'data_3d_j3d': data_3d_j3d,
                'has_3d_joints': has_3d_joints,
                'data_3d_j2d': data_3d_j2d,
                'has_2d_joints': has_2d_joints,
                'has_smpl': has_smpl,
                'img': img,
                'class_num': class_num},
               save_path)

print(data)