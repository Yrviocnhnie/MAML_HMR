import torch
import numpy as np
import os
import sys

# from torch.utils.data import Dataset, DataLoader

# class human36m_tasks(Dataset):
    
def load_tasks(is_train = True, n_way = 4, k_spt = 2, k_query = 4, batch_size = 4, 
                root_dir = '/data/share/3D/MeshTransformer/datasets/human3.6m/tasks'):
    # self.root_dir = root_dir
    # self.n_way = n_way
    # self.k_spt = k_spt  # number of support data
    # self.k_query = k_query # number of query data
    # self.root_dir = root_dir
    
    # classes_list = os.listdir(self.root_dir)
    # # 5 way 1 shot: 5 * 1
    # self.setsz = self.k_spt*self.n_way
    # self.querysz = self.k_query*self.n_way
    
    root_dir = root_dir
    n_way = n_way
    k_spt = k_spt  # number of support data
    k_query = k_query # number of query data
    root_dir = root_dir
    
    # len:17 15 for training & 2 for testing
    classes_list = os.listdir(root_dir)
    if is_train:
        classes_list = classes_list[:13]
    else:
        classes_list = classes_list[13:]
    # 3 way 1 shot: 3 * 1
    setsz = k_spt*n_way
    querysz = k_query*n_way
    
    spt_data = []
    qry_data = []
    for random_time in range(5):
        
        # sqt data
        spt_poses, spt_shapes, spt_has_smpls, spt_imgs = [], [], [], []
        spt_data_3d_j3ds, spt_has_3d_joints, spt_data_3d_j2ds, spt_has_2d_joints = [], [], [], []
        spt_thetas = []
        
        # qry data
        qry_poses, qry_shapes, qry_has_smpls, qry_imgs = [], [], [], []
        qry_data_3d_j3ds, qry_has_3d_joints, qry_data_3d_j2ds, qry_has_2d_joints = [], [], [], []
        qry_thetas = []
        
        for _ in range(batch_size):
            # spt & qry data for this batch
            spt_pose, spt_shape, spt_has_smpl, spt_img = [], [], [], []
            spt_data_3d_j3d, spt_has_3d_joint, spt_data_3d_j2d, spt_has_2d_joint = [], [], [], []
            spt_theta = []
            
            qry_pose, qry_shape, qry_has_smpl, qry_img = [], [], [], []
            qry_data_3d_j3d, qry_has_3d_joint, qry_data_3d_j2d, qry_has_2d_joint = [], [], [], []
            qry_theta = []
            
            # select one class to form one batch of data
            selected_classes = np.random.choice(classes_list, n_way, replace = False)
            for selected_class in selected_classes:
                class_path = os.path.join(root_dir, selected_class)
                files_list = [f for f in os.listdir(class_path)]
                
                selected_files = np.random.choice(files_list, k_spt+k_query, replace = False)
                
                for i, selected_file in enumerate(selected_files):
                    # load the selected class and selected file data
                    data = torch.load(os.path.join(class_path, selected_file))
                    theta_files = np.random.choice(files_list, 3, replace = False)
                    thetas = []
                    for j in theta_files:
                        data = torch.load(os.path.join(class_path, j))
                        theta = np.concatenate([np.zeros(3), data['pose'].numpy(), data['shape'].numpy()])
                        thetas.append(theta)

                    if data['shape'].numpy().sum() == 0:
                        has_smpl = 0
                    else:
                        has_smpl = data['has_smpl']
                    if i < k_spt:
                        spt_pose.append(data['pose'].numpy())
                        spt_shape.append(data['shape'].numpy())
                        spt_data_3d_j3d.append(data['data_3d_j3d'].numpy())
                        spt_has_3d_joint.append(data['has_3d_joints'])
                        spt_data_3d_j2d.append(data['data_3d_j2d'].numpy())
                        spt_has_2d_joint.append(data['has_2d_joints'])
                        spt_has_smpl.append(has_smpl)
                        spt_img.append(data['img'].numpy())
                        spt_theta.append(np.array(thetas))
                    else:
                        qry_pose.append(data['pose'].numpy())
                        qry_shape.append(data['shape'].numpy())
                        qry_data_3d_j3d.append(data['data_3d_j3d'].numpy())
                        qry_has_3d_joint.append(data['has_3d_joints'])
                        qry_data_3d_j2d.append(data['data_3d_j2d'].numpy())
                        qry_has_2d_joint.append(data['has_2d_joints'])
                        qry_has_smpl.append(has_smpl)
                        qry_img.append(data['img'].numpy())
                        qry_theta.append(np.array(thetas))
            
            
            perm = np.random.permutation(n_way*k_spt)  
            spt_poses.append(np.array(spt_pose)[perm])
            spt_shapes.append(np.array(spt_shape)[perm])
            spt_data_3d_j3ds.append(np.array(spt_data_3d_j3d)[perm])
            spt_has_3d_joints.append(np.array(spt_has_3d_joint)[perm])
            spt_data_3d_j2ds.append(np.array(spt_data_3d_j2d)[perm])                
            spt_has_2d_joints.append(np.array(spt_has_2d_joint)[perm])
            spt_has_smpls.append(np.array(spt_has_smpl)[perm])
            spt_imgs.append(np.array(spt_img)[perm])
            spt_thetas.append(np.array(spt_theta)[perm])
            
            perm = np.random.permutation(n_way*k_query)  
            qry_poses.append(np.array(qry_pose)[perm])
            qry_shapes.append(np.array(qry_shape)[perm])
            qry_data_3d_j3ds.append(np.array(qry_data_3d_j3d)[perm])
            qry_has_3d_joints.append(np.array(qry_has_3d_joint)[perm])
            qry_data_3d_j2ds.append(np.array(qry_data_3d_j2d)[perm])                
            qry_has_2d_joints.append(np.array(qry_has_2d_joint)[perm])
            qry_has_smpls.append(np.array(qry_has_smpl)[perm])
            qry_imgs.append(np.array(qry_img)[perm])
            qry_thetas.append(np.array(qry_theta)[perm])
        
        spt_poses = np.array(spt_poses)
        spt_shapes = np.array(spt_shapes)
        spt_data_3d_j3ds = np.array(spt_data_3d_j3ds)
        spt_has_3d_joints = np.array(spt_has_3d_joints)
        spt_data_3d_j2ds = np.array(spt_data_3d_j2ds)
        spt_has_2d_joints = np.array(spt_has_2d_joints)
        spt_has_smpls = np.array(spt_has_smpls)
        spt_imgs = np.array(spt_imgs)
        spt_thetas = np.array(spt_thetas)
        
        
        qry_poses = np.array(qry_poses)
        qry_shapes = np.array(qry_shapes)
        qry_data_3d_j3ds = np.array(qry_data_3d_j3ds)
        qry_has_3d_joints = np.array(qry_has_3d_joints)
        qry_data_3d_j2ds = np.array(qry_data_3d_j2ds)
        qry_has_2d_joints = np.array(qry_has_2d_joints)
        qry_has_smpls = np.array(qry_has_smpls)
        qry_imgs = np.array(qry_imgs)
        qry_thetas = np.array(qry_thetas)
        
        # 0, 1, 2, 3
        # 4, 5, 6, 7
        spt_data.append([spt_poses, spt_shapes, spt_data_3d_j3ds, spt_has_3d_joints,
                            spt_data_3d_j2ds, spt_has_2d_joints, spt_has_smpls, spt_imgs, spt_thetas])
        qry_data.append([qry_poses, qry_shapes, qry_data_3d_j3ds, qry_has_3d_joints,
                            qry_data_3d_j2ds, qry_has_2d_joints, qry_has_smpls, qry_imgs, qry_thetas])
        
    return spt_data, qry_data
            
    

if __name__ == '__main__':
    # print(random.rand(1))
    spt_data, qry_data = load_tasks(batch_size = 8)
        