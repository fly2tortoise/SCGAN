from __future__ import absolute_import, division, print_function

import sys

import cfg
import archs
import datasets
from network import train, validate, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger, count_parameters_in_MB
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.flop_benchmark import print_FLOPs

import torch
import os
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # set visible GPU ids
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        
    # set TensorFlow environment for evaluation (calculate IS and FID)
    _init_inception([args.eval_batch_size,args.img_size,args.img_size,3])
    inception_path = check_or_download_inception('./tmp/imagenet/')
    create_inception_graph(inception_path)

    # the first GPU in visible GPUs is dedicated for evaluation (running Inception model)
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for id in range(len(str_ids)):
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 1:
      args.gpu_ids = args.gpu_ids[1:]
    else:
      args.gpu_ids = args.gpu_ids
    
    # select the searched GANs architectures
    SCGAN =   [[1, 1, 1, 3, 2, 2, 1], [1, 1, 3, 1, 3, 1, 1], [1, 1, 1, 0, 1, 2, 4]]
    SCGAN1 =   [[1, 1, 1, 3, 1, 2, 2], [1, 1, 4, 2, 5, 0, 1], [1, 1, 1, 5, 2, 1, 3]]
    genotype_G = SCGAN
    # genotype D
    # genotype_D = np.load(os.path.join(genotypes_root, 'latest_D.npy'))

    # import network from genotype
    basemodel_gen = eval('archs.' + args.arch + '.Generator')(args, genotype_G)
    gen_net = torch.nn.DataParallel(basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])
    basemodel_dis = eval('archs.' + args.arch + '.Discriminator')(args)
    dis_net = torch.nn.DataParallel(basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
            
    gen_net.apply(weights_init)
    dis_net.apply(weights_init)
    
    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train
    print(len(train_loader))
    
    # epoch number for dis_net
    args.max_epoch_D = args.max_epoch_G * args.n_critic
    if args.max_iter_G:
        args.max_epoch_D = np.ceil(args.max_iter_G * args.n_critic / len(train_loader))
    max_iter_D = args.max_epoch_D * len(train_loader)
    
    # set optimizer
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, max_iter_D)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, max_iter_D)

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    elif args.dataset.lower() == 'celeba':
        fid_stat = 'fid_stat/fid_stats_celebA_train.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')

    assert os.path.exists(fid_stat)

    # initial
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0
    best_fid = 1e4
    best_is = 0

    # set writer
    if args.checkpoint:
        # resuming
        print(f'=> resuming from {args.checkpoint}')
        print(os.path.join('exps', args.checkpoint))
        assert os.path.exists(os.path.join('exps', args.checkpoint))
        checkpoint_file = os.path.join('exps', args.checkpoint, 'Model', 'checkpoint_best.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(avg_gen_net)
        del avg_gen_net

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('exps', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }
    
    # for visualization
    if args.draw_arch:
        from utils.genotype import draw_graph_G, draw_graph_D
        draw_graph_G(genotype_G, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], 'latest_G'))
        # draw_graph_D(genotype_D, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], 'latest_D'))

    # model size
    logger.info('Param size of G = %fMB', count_parameters_in_MB(gen_net))
    logger.info('Param size of D = %fMB', count_parameters_in_MB(dis_net))
    logger.info(genotype_G)
    print_FLOPs(basemodel_gen, (1, args.latent_dim), logger)
    print_FLOPs(basemodel_dis, (1, 3, args.img_size, args.img_size), logger)
    
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))
    
    improvement_count = 6
    icounter = improvement_count
    logger.info(f'Upper bound: {args.bu} and Lower Bound: {args.bl}.')
    logger.info(f'Best FID score: {best_fid}. Best IS score: {best_is}.')

    # train loop
    # print()
    # sys.exit(args.max_epoch_D)
    is_tty = sys.stdout.isatty()  # 后台运行时不出现进度条
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch_D)), desc='total progress',disable=not is_tty):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer,
              gen_avg_param, train_loader, epoch, writer_dict, lr_schedulers)

        # if epoch % args.val_freq == 0 or epoch == int(args.max_epoch_D)-1:
        if epoch == 0 or (epoch % args.val_freq == 0 and epoch >= 100):
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param)
            inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
            logger.info(f'Inception score mean: {inception_score}, Inception score std: {std}, '
                        f'FID score: {fid_score} || @ epoch {epoch}.')
            load_params(gen_net, backup_param)
            if fid_score < best_fid:
                best_fid = fid_score
                best_is = inception_score
                is_best = True
                icounter = improvement_count
            else:
                is_best = False
                icounter = icounter - 1
        else:
            is_best = False

        # If there is no improvement for 30 epoches then load the best model
        if icounter == 0:
            print(f'Upper bound changed from {args.bu} to {args.bu*2}.')
            args.bu = args.bu * 2
            if args.bu > 32:
                args.bu = 32
            logger.info(f'Upper bound: {args.bu} and Lower Bound: {args.bl}.')
            icounter = improvement_count
            logger.info(f'Best FID score: {best_fid}, Best IS score: {best_is}. || @ epoch {epoch}.')

        
if __name__ == '__main__':
    main()
