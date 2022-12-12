import os
import torch
import numpy as np
import config
from config import args
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from collections import OrderedDict

from util import align_by_pelvis, batch_rodrigues, copy_state_dict
from model_meta import HMRNetBase
from Discriminator import Discriminator
from meta_weight_update import gradient_update_parameters
from dataloader.human36m_Taskloader import load_tasks




device = torch.device('cuda', 0)

def _fetch_data(data, batch_idx):
    poses, shapes = torch.from_numpy(data[0]).to(device, dtype=torch.float32), torch.from_numpy(data[1]).to(device, dtype=torch.float32)
    data_3d_j3ds, has_3d_joints = torch.from_numpy(data[2]).to(device, dtype=torch.float32), torch.from_numpy(data[3]).to(device)
    data_3d_j2ds, has_2d_joints = torch.from_numpy(data[4]).to(device, dtype=torch.float32), torch.from_numpy(data[5]).to(device)
    has_smpls, imgs = torch.from_numpy(data[6]).to(device), torch.from_numpy(data[7]).to(device)
    thetas = torch.from_numpy(data[8]).to(device, dtype=torch.float32)
    
    
    return poses[batch_idx], shapes[batch_idx], data_3d_j3ds[batch_idx], has_3d_joints[batch_idx], data_3d_j2ds[batch_idx], has_2d_joints[batch_idx], has_smpls[batch_idx], imgs[batch_idx], thetas[batch_idx]

def _accumulate_thetas(generator_outputs):
    thetas = []
    for (theta, verts, j2d, j3d, Rs) in generator_outputs:
        thetas.append(theta)
    return torch.cat(thetas, 0)


class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__() 
        self.update_step = 2
        self.update_step_test = 2
        self._build_model()
    
        
    def _build_model(self):
        print('start building modle.')

        '''
            load pretrain model
        '''
        generator = HMRNetBase()
        model_path = config.pre_trained_model['generator']
        if os.path.exists(model_path):
            copy_state_dict(generator.state_dict(), torch.load(model_path), prefix = 'module.')
        else:
            print('model {} not exist!'.format(model_path))

        discriminator = Discriminator()
        model_path = config.pre_trained_model['discriminator']
        if os.path.exists(model_path):
            copy_state_dict(discriminator.state_dict(), torch.load(model_path), prefix = 'module.')
        else:
            print('model {} not exist!'.format(model_path))

        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        
        self.generator.train()
        self.discriminator.train()
        
        self.e_base_lr, self.d_base_lr = args.e_lr*50, args.d_lr*50
        # optimizer for the generater
        self.e_opt = torch.optim.Adam(self.generator.parameters(), lr = args.e_lr, weight_decay = args.e_wd)
        self.e_sche = torch.optim.lr_scheduler.StepLR(self.e_opt, step_size = 500, gamma = 0.9)
        # optimizer for the discriminator
        self.d_opt = torch.optim.Adam(self.discriminator.parameters(), lr = args.d_lr, weight_decay = args.d_wd)
        self.d_sche = torch.optim.lr_scheduler.StepLR(self.d_opt, step_size = 500, gamma = 0.9)

        print('finished build model.')
        
    # def forward(self, spt_poses, spt_shapes, spt_data_3d_j3ds, spt_has_3d_joints,
    #                  spt_data_3d_j2ds, spt_has_2d_joints, spt_has_smpls, spt_imgs,
    #                   qry_poses, qry_shapes, qry_data_3d_j3ds, qry_has_3d_joints,
    #                  qry_data_3d_j2ds, qry_has_2d_joints, qry_has_smpls, qry_imgs):
    def forward(self, spt_data, qry_data):
        
        # spt data shape: [batch size, n_way*k_spt, ...]
        # qry data shape: [batch size, n_way*k_qry, ...]
        
        batch_size = spt_data[0].shape[0]
        e_loss_list_qry = [0 for _ in range(self.update_step + 1)]
        d_loss_list_qry = [0 for _ in range(self.update_step + 1)]
        
        for i in range(batch_size):
            
            gen_clone = HMRNetBase().to(device)
            disc_clone = Discriminator().to(device)
            gen_clone.train()
            disc_clone.train()
            
            gen_clone.load_state_dict(deepcopy(self.generator.state_dict()))
            disc_clone.load_state_dict(deepcopy(self.discriminator.state_dict()))
            e_opt_clone = torch.optim.Adam(gen_clone.parameters(), lr = self.e_base_lr, weight_decay = args.e_wd)
            d_opt_clone = torch.optim.Adam(disc_clone.parameters(), lr = self.d_base_lr, weight_decay = args.d_wd)
            
            
            # Update step 0
            # calculate the updated and making a deep copy of it
            spt_imgs = torch.from_numpy(spt_data[7][i]).to(device)
            generator_outputs = self.generator(spt_imgs, params = None)
            loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, \
                e_disc_loss, d_disc_loss, d_disc_real, d_disc_predict = self._calc_loss(generator_outputs, spt_data, i, params = None)
                
            e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss 
            d_loss = d_disc_loss
            
            # e_fast_weights = gradient_update_parameters(self.generator, gen_clone, e_loss, step_size=self.e_base_lr)
            # d_fast_weights = gradient_update_parameters(self.discriminator, d_loss, step_size=self.d_base_lr)
            e_opt_clone.zero_grad()
            e_loss.backward()
            e_opt_clone.step()
            
            d_opt_clone.zero_grad()
            d_loss.backward()
            d_opt_clone.step()
            
            # Test on the query set: using the weights before upadte 
            with torch.no_grad():
                qry_imgs = torch.from_numpy(qry_data[7][i]).to(device)
                generator_outputs = self.generator(qry_imgs, params = OrderedDict(self.generator.meta_named_parameters()))
                loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, \
                   e_disc_loss, d_disc_loss, d_disc_real, d_disc_predict = self._calc_loss(generator_outputs, qry_data, i, params = OrderedDict(self.discriminator.meta_named_parameters()))
                   
                e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss 
                d_loss = d_disc_loss

                e_loss_list_qry[0] += e_loss
                d_loss_list_qry[0] += d_loss
                
            # Test on the query set: using the upadte weighted
            with torch.no_grad():
                qry_imgs = torch.from_numpy(qry_data[7][i]).to(device)
                generator_outputs = self.generator(qry_imgs, params = OrderedDict(gen_clone.meta_named_parameters()))
                loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, \
                    e_disc_loss, d_disc_loss, d_disc_real, d_disc_predict = self._calc_loss(generator_outputs, qry_data, i, params = OrderedDict(disc_clone.meta_named_parameters()))
                e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss 
                d_loss = d_disc_loss
                
                e_loss_list_qry[1] += e_loss
                d_loss_list_qry[1] += d_loss
                
            for k in range(1, self.update_step):
                # Uptate step k: using updaated weights from step 0
                spt_imgs = torch.from_numpy(spt_data[7][i]).to(device)
                generator_outputs = self.generator(spt_imgs, params = OrderedDict(gen_clone.meta_named_parameters()))
                loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, \
                    e_disc_loss, d_disc_loss, d_disc_real, d_disc_predict = self._calc_loss(generator_outputs, spt_data, i, params = OrderedDict(disc_clone.meta_named_parameters()))
                    
                e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss 
                d_loss = d_disc_loss
                
                
                # e_fast_weights = gradient_update_parameters(self.generator, e_loss, params = e_fast_weights, step_size=self.e_base_lr)
                # d_fast_weights = gradient_update_parameters(self.discriminator, d_loss, params = d_fast_weights, step_size=self.d_base_lr)
                e_opt_clone.zero_grad()
                e_loss.backward()
                e_opt_clone.step()
                
                d_opt_clone.zero_grad()
                d_loss.backward()
                d_opt_clone.step()
                
                    
                # Test on the query set: using updated weights from step k
                qry_imgs = torch.from_numpy(qry_data[7][i]).to(device)
                generator_outputs = self.generator(qry_imgs, params = OrderedDict(gen_clone.meta_named_parameters()))
                loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, \
                    e_disc_loss, d_disc_loss, d_disc_real, d_disc_predict = self._calc_loss(generator_outputs, qry_data, i, params = OrderedDict(disc_clone.meta_named_parameters()))
                e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss 
                d_loss = d_disc_loss
                
                e_loss_list_qry[k+1] += e_loss
                d_loss_list_qry[k+1] += d_loss
                
                del gen_clone
                del disc_clone
                
                del e_opt_clone
                del d_opt_clone
                
        
            
            
        e_loss_qry = e_loss_list_qry[-1] / batch_size
        d_loss_qry = d_loss_list_qry[-1] / batch_size
        
        self.e_opt.zero_grad()
        e_loss_qry.backward()
        self.e_opt.step()
        
        self.d_opt.zero_grad()
        d_loss_qry.backward()
        self.d_opt.step()
        
        e_loss = np.array([loss.item() for loss in e_loss_list_qry]) / batch_size
        d_loss = np.array([loss.item() for loss in d_loss_list_qry]) / batch_size
        
        
        return e_loss, d_loss
    
    
    def _calc_loss(self, generator_outputs, data_3d, batch_idx, params, finetune = False, new_dis = None):
        poses, shapes, data_3d_j3ds, has_3d_joints, data_3d_j2ds, has_2d_joints, has_smpls, imgs, thetas = _fetch_data(data_3d, batch_idx)
        
        theta = torch.cat((torch.zeros((poses.shape[0], 3)).to(device), poses, shapes), axis = 1).float().to(device)
        data_3d_theta, w_3d, w_smpl = theta, has_3d_joints.float().to(device), has_smpls.float().to(device)
        
        total_predict_thetas = _accumulate_thetas(generator_outputs)
        (predict_theta, predict_verts, predict_j2d, predict_j3d, predict_Rs) = generator_outputs[-1]
        
        data_3d_j2d = data_3d_j2ds
        data_3d_j3d = data_3d_j3ds[:, :, :-1].clone()
        
        real_2d, real_3d = data_3d_j2d.to(device), data_3d_j3d.to(device)
        
        loss_kp_2d = self.batch_kp_2d_l1_loss(real_2d, predict_j2d[:,:14,:]) *  args.e_loss_weight
        loss_kp_3d = self.batch_kp_3d_l2_loss(real_3d, predict_j3d[:,:14,:], w_3d) * args.e_3d_loss_weight * args.e_3d_kp_ratio
        
        real_shape, predict_shape = data_3d_theta[:, 75:], predict_theta[:, 75:]
        loss_shape = self.batch_shape_l2_loss(real_shape, predict_shape, w_smpl) * args.e_3d_loss_weight * args.e_shape_ratio
        
        real_pose, predict_pose = data_3d_theta[:, 3:75], predict_theta[:, 3:75]
        loss_pose = self.batch_pose_l2_loss(real_pose.contiguous(), predict_pose.contiguous(), w_smpl) * args.e_3d_loss_weight * args.e_pose_ratio
        
        if finetune:
            encoder_disc_value = new_dis(total_predict_thetas, params = params)
            e_disc_loss = self.batch_encoder_disc_l2_loss(encoder_disc_value) * args.d_loss_weight * args.d_disc_ratio

            real_thetas = thetas.to(device)
            thetas = []
            for theta in real_thetas:
                thetas.append(theta)    
            real_thetas = torch.cat(thetas, 0)
            fake_thetas = total_predict_thetas.detach()
            fake_disc_value, real_disc_value = new_dis(fake_thetas, params = params), new_dis(real_thetas, params = params)
            d_disc_real, d_disc_fake, d_disc_loss = self.batch_adv_disc_l2_loss(real_disc_value, fake_disc_value)
            d_disc_real, d_disc_fake, d_disc_loss = d_disc_real  * args.d_loss_weight * args.d_disc_ratio, d_disc_fake  * args.d_loss_weight * args.d_disc_ratio, d_disc_loss * args.d_loss_weight * args.d_disc_ratio
        else:
            encoder_disc_value = self.discriminator(total_predict_thetas, params = params)
            e_disc_loss = self.batch_encoder_disc_l2_loss(encoder_disc_value) * args.d_loss_weight * args.d_disc_ratio

            real_thetas = thetas.to(device)
            thetas = []
            for theta in real_thetas:
                thetas.append(theta)    
            real_thetas = torch.cat(thetas, 0)
            fake_thetas = total_predict_thetas.detach()
            fake_disc_value, real_disc_value = self.discriminator(fake_thetas, params = params), self.discriminator(real_thetas, params = params)
            d_disc_real, d_disc_fake, d_disc_loss = self.batch_adv_disc_l2_loss(real_disc_value, fake_disc_value)
            d_disc_real, d_disc_fake, d_disc_loss = d_disc_real  * args.d_loss_weight * args.d_disc_ratio, d_disc_fake  * args.d_loss_weight * args.d_disc_ratio, d_disc_loss * args.d_loss_weight * args.d_disc_ratio

        return loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, e_disc_loss, d_disc_loss, d_disc_real, d_disc_fake
    # Calc L1 error
    def batch_kp_2d_l1_loss(self, real_2d_kp, predict_2d_kp):
        kp_gt = real_2d_kp.view(-1, 3) # (224, 3)
        kp_pred = predict_2d_kp.contiguous().view(-1, 2) # (224, 2)
        vis = kp_gt[:, 2]
        k = torch.sum(vis) * 2.0 + 1e-8
        dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
        return torch.matmul(dif_abs, vis) * 1.0 / k
    
    # calc mse * 0.5
    def batch_kp_3d_l2_loss(self, real_3d_kp, fake_3d_kp, w_3d):
        shape = real_3d_kp.shape
        k = torch.sum(w_3d) * shape[1] * 3.0 * 2.0 + 1e-8

        #first align it
        # real_3d_kp, fake_3d_kp = align_by_pelvis(real_3d_kp), align_by_pelvis(fake_3d_kp)
        kp_gt = real_3d_kp
        kp_pred = fake_3d_kp
        kp_dif = (kp_gt - kp_pred) ** 2
        return torch.matmul(kp_dif.sum(1).sum(1), w_3d) * 1.0 / k
    
    # calc mse * 0.5
    def batch_shape_l2_loss(self, real_shape, fake_shape, w_shape):
        k = torch.sum(w_shape) * 10.0 * 2.0 + 1e-8
        shape_dif = (real_shape - fake_shape) ** 2
        return  torch.matmul(shape_dif.sum(1), w_shape) * 1.0 / k

    # calc mse * 0.5
    def batch_pose_l2_loss(self, real_pose, fake_pose, w_pose):
        k = torch.sum(w_pose) * 207.0 * 2.0 + 1e-8
        real_rs, fake_rs = batch_rodrigues(real_pose.view(-1, 3)).view(-1, 24, 9)[:,1:,:], batch_rodrigues(fake_pose.view(-1, 3)).view(-1, 24, 9)[:,1:,:]
        dif_rs = ((real_rs - fake_rs) ** 2).view(-1, 207)
        return torch.matmul(dif_rs.sum(1), w_pose) * 1.0 / k
    
    def batch_encoder_disc_l2_loss(self, disc_value):
        k = disc_value.shape[0]
        return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k
    
    def batch_adv_disc_l2_loss(self, real_disc_value, fake_disc_value):
        ka = real_disc_value.shape[0]
        kb = fake_disc_value.shape[0]
        lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
        return la, lb, la + lb
    
    def save_model(self, result):
        exclude_key = 'module.smpl'
        def exclude_smpl(model_dict):
            result = OrderedDict()
            for (k, v) in model_dict.items():
                if exclude_key in k:
                    continue
                result[k] = v
            return result

        parent_folder = '/home/richie/MAML_HMR/src/trained_model_0'
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)

        title = result['epoch']
        generator_save_path = os.path.join(parent_folder, str(title) + '_generator.pkl')
        torch.save(exclude_smpl(self.generator.state_dict()), generator_save_path)
        disc_save_path = os.path.join(parent_folder, str(title) + '_discriminator.pkl')
        torch.save(exclude_smpl(self.discriminator.state_dict()), disc_save_path)
        with open(os.path.join(parent_folder, str(title) + '.txt'), 'w') as fp:
            fp.write(str(result))
            
    # def finetuning(self, spt_data, qry_data):
        
    #     batch_size = spt_data[0].shape[0]
    #     query_size = spt_data[0].shape[1]
    #     e_loss_list_qry = [0 for _ in range(self.update_step + 1)]
    #     d_loss_list_qry = [0 for _ in range(self.update_step + 1)]
    #     for i in range(batch_size):
            
    #         new_gen = deepcopy(self.generator)
    #         new_dis = deepcopy(self.discriminator)
    #         # Update step 0
    #         # calculate the updated and making a deep copy of it
    #         spt_imgs = torch.from_numpy(spt_data[7][i]).to(device)
    #         generator_outputs = new_gen(spt_imgs, params = None)
    #         loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, \
    #             e_disc_loss, d_disc_loss, d_disc_real, d_disc_predict = self._calc_loss(generator_outputs, spt_data, i, params = None, finetune = True, new_dis = new_dis)
                
    #         e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss 
    #         d_loss = d_disc_loss
            
    #         e_fast_weights = gradient_update_parameters(new_gen, e_loss, step_size=self.e_base_lr)
    #         d_fast_weights = gradient_update_parameters(new_dis, d_loss, step_size=self.d_base_lr)
            
    #         # Test on the query set: using the weights before upadte 
    #         with torch.no_grad():
    #             qry_imgs = torch.from_numpy(qry_data[7][i]).to(device)
    #             generator_outputs = new_gen(qry_imgs, params = OrderedDict(new_gen.meta_named_parameters()))
    #             dis_params = OrderedDict(new_dis.meta_named_parameters())
    #             loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, \
    #                 e_disc_loss, d_disc_loss, d_disc_real, d_disc_predict = self._calc_loss(generator_outputs, qry_data, i, params = dis_params, finetune = True, new_dis = new_dis)
                    
    #             e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss 
    #             d_loss = d_disc_loss

    #             e_loss_list_qry[0] += e_loss / query_size
    #             d_loss_list_qry[0] += d_loss / query_size
                
    #         # Test on the query set: using the upadte weighted
    #         with torch.no_grad():
    #             qry_imgs = torch.from_numpy(qry_data[7][i]).to(device)
    #             generator_outputs = new_gen(qry_imgs, params = e_fast_weights)
    #             loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, \
    #                 e_disc_loss, d_disc_loss, d_disc_real, d_disc_predict = self._calc_loss(generator_outputs, qry_data, i, params = d_fast_weights, finetune = True, new_dis = new_dis)
    #             e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss 
    #             d_loss = d_disc_loss
                
    #             e_loss_list_qry[1] += e_loss / query_size
    #             d_loss_list_qry[1] += d_loss / query_size
                
    #         for k in range(1, self.update_step):
    #             # Uptate step k: using updaated weights from step 0
    #             spt_imgs = torch.from_numpy(spt_data[7][i]).to(device)
    #             generator_outputs = new_gen(spt_imgs, params = e_fast_weights)
    #             loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, \
    #                 e_disc_loss, d_disc_loss, d_disc_real, d_disc_predict = self._calc_loss(generator_outputs, spt_data, i, params = d_fast_weights, finetune = True, new_dis = new_dis)
                    
    #             e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss 
    #             d_loss = d_disc_loss
                
    #             e_fast_weights = gradient_update_parameters(new_gen, e_loss, step_size=self.e_base_lr)
    #             d_fast_weights = gradient_update_parameters(new_dis, d_loss, step_size=self.d_base_lr)
                    
    #             # Test on the query set: using updated weights from step k
    #             qry_imgs = torch.from_numpy(qry_data[7][i]).to(device)
    #             generator_outputs = new_gen(qry_imgs, params = e_fast_weights)
    #             loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, \
    #                 e_disc_loss, d_disc_loss, d_disc_real, d_disc_predict = self._calc_loss(generator_outputs, qry_data, i, params = d_fast_weights, finetune = True, new_dis = new_dis)
    #             e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss 
    #             d_loss = d_disc_loss
                
    #             e_loss_list_qry[k+1] += e_loss / query_size
    #             d_loss_list_qry[k+1] += d_loss / query_size
            
    #         del new_gen
    #         del new_dis
        
    #     e_loss = np.array([loss.item() for loss in e_loss_list_qry]) / batch_size
    #     d_loss = np.array([loss.item() for loss in d_loss_list_qry]) / batch_size
        
    
    #     return e_loss, d_loss
  
