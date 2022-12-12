import os.path as op
import torch
import logging
import code
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


def hum36m_dataloader(batch_size, train_epochs, yaml_file, is_train=True, scale_factor=1):

    dataset = build_dataset(yaml_file, is_train=is_train, scale_factor=scale_factor)
    logger = logging.getLogger(__name__)
    if is_train==True:
        shuffle = True
        images_per_gpu = batch_size
        images_per_batch = images_per_gpu * get_world_size()
        # iters_per_batch = len(dataset) // images_per_batch
        num_iters = train_epochs
        # logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    # else:
        # shuffle = False
        # images_per_gpu = args.per_gpu_eval_batch_size
        # num_iters = None
        # start_iter = 0

    # sampler = make_data_sampler(dataset, shuffle, is_distributed)
    # batch_sampler = make_batch_data_sampler(
    #     sampler, images_per_gpu, num_iters, start_iter
    # )
    
    # data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=True)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, pin_memory=True)
    
    return dataset
