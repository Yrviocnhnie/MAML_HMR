

'''
    file:   trainer.py

    date:   2018_05_07
    author: zhangxiong(1025679612@qq.com)
'''

import sys
from model_meta import HMRNetBase
from Discriminator import Discriminator
from config import args
import config
import torch
import torch.nn as nn
import datetime
# from torch.utils.tensorboard import SummaryWriter

from util import align_by_pelvis, batch_rodrigues
from timer import Clock
import time
import datetime
from collections import OrderedDict
import os
import numpy as np

from dataloader.human36m_Taskloader import load_tasks
from MetaLearner_1 import MetaLearner

# from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda', 1)

class HMRTrainer(object):
    def __init__(self):
        self.pix_format = 'NCHW'
        self.normalize = True
        self.flip_prob = 0.5
        self.use_flip = False
        self.w_smpl = torch.ones((config.args.eval_batch_size)).float().to(device)

        self.meta = MetaLearner().to(device)
        start = time.time()
        print("Start Loading Datassets: ")
        self.indexes = {'train': 0, 'test': 0}
        self.datasets_caches = {'train': load_tasks(is_train=True, batch_size = config.args.batch_3d_size),
                                'test': load_tasks(is_train=False, batch_size = config.args.batch_3d_size)}
        
        end = time.time()
        print(f'Finish Loading using {end - start} s\n')
        

    def next(self, mode = 'train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        
        if self.indexes[mode] >= len(self.datasets_caches[mode][0]):
            self.indexes[mode] = 0
            if mode == 'train':
                self.datasets_caches[mode] = load_tasks(is_train=True, batch_size = config.args.batch_3d_size)
            else:
                self.datasets_caches[mode] = load_tasks(is_train=False, batch_size = config.args.batch_3d_size)
        
        spt_data = self.datasets_caches[mode][0][self.indexes[mode]]
        qry_data = self.datasets_caches[mode][1][self.indexes[mode]]
        self.indexes[mode] += 1

        return spt_data, qry_data
    
        

    def train(self):
        
        '''# Create the graph to view
        graph_folder = args.save_graph_path
        os.makedirs(graph_folder, exist_ok=True)
        writer = SummaryWriter(graph_folder, flush_secs=15)
        '''
        torch.backends.cudnn.benchmark = True
        
            
        # self.generator.train()
        # self.discriminator.train()
        task_num = config.args.batch_3d_size
        epochs = 60000
        for epoch in range(epochs):
            spt_data, qry_data = self.next('train')
            e_loss, d_loss = self.meta(spt_data, qry_data)
            
            if epoch % 10 == 0:
                round_e_loss = [ '%.2f' % loss for loss in e_loss ]
                round_d_loss = [ '%.2f' % loss for loss in d_loss ]
                
                iter_msg = OrderedDict(
                    [
                        ('time',datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')),
                        ('epoch',epoch+26500),
                        ('e_loss', round_e_loss),
                        ('d_loss',round_d_loss),
                    ]
                )
                
                # print(iter_msg)
                # print(round_e_loss)
                # print(round_d_loss)
                
                with open('/home/richie/MAML_HMR/src/output_1.txt', 'a+') as f:
                    f.write('epoch: '+ str(epoch+26500))
                    f.write(', e_loss: '+ str(round_e_loss))
                    f.write(', d_loss: '+ str(round_d_loss))
                    f.write('\n')
                
            if epoch % 1000 == 0:
                self.meta.save_model(iter_msg)
                e_losses, d_losses = [], []
                losses_kp_2d, losses_kp_3d , losses_shape, losses_pose, e_disc_losses = [], [], [], [], []

                
                # spt_data, qry_data = self.next('test')
                for _ in range(100):
                    spt_data, qry_data = self.next('test')
                    e_loss, d_loss, loss_kp_2d, loss_kp_3d, loss_shape, loss_pose, e_disc_loss = self.meta.finetuning(spt_data, qry_data)
                    e_losses.append(e_loss)
                    d_losses.append(d_loss)
                    losses_kp_2d.append(loss_kp_2d)
                    losses_kp_3d.append(loss_kp_3d)
                    losses_shape.append(loss_shape)
                    losses_pose.append(loss_pose)
                    e_disc_losses.append(e_disc_loss)
                
                
                e_losses = np.array(e_losses).mean(axis=0).astype(np.float16)
                d_losses = np.array(d_losses).mean(axis=0).astype(np.float16)
                losses_kp_2d = np.array(losses_kp_2d).mean(axis=0).astype(np.float16)
                losses_kp_3d = np.array(losses_kp_3d).mean(axis=0).astype(np.float16)
                losses_shape = np.array(losses_shape).mean(axis=0).astype(np.float16)
                losses_pose = np.array(losses_pose).mean(axis=0).astype(np.float16)
                e_disc_losses = np.array(e_disc_losses).mean(axis=0).astype(np.float16)
                
                
                with open('/home/richie/MAML_HMR/src/output_1.txt', 'a+') as f:
                    f.write('epoch: '+ str(epoch+26500))
                    f.write(', Testing Generator Loss:: '+ str(round_e_loss))
                    f.write(', Testing 2d kp Loss:: '+ str(round_e_loss))
                    f.write(', Testing 3d kp Loss: '+ str(round_d_loss))
                    f.write(', Testing shape Loss:: '+ str(round_e_loss))
                    f.write(', Testing pose Loss: '+ str(round_d_loss))
                    f.write(', Testing e disc Loss:: '+ str(round_e_loss))
                    f.write(', Testing Discriminator Loss: '+ str(round_d_loss))
                    f.write('\n')
                
            
                    
            
def main():
    trainer = HMRTrainer()
    trainer.train()

if __name__ == '__main__':
    main()
