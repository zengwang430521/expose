# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import sys
import os
import os.path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'


from threadpoolctl import threadpool_limits
from tqdm import tqdm
import datetime

import open3d as o3d

import time
import argparse
from collections import defaultdict
from loguru import logger
from collections import OrderedDict
import numpy as np

import torch

import resource

from expose.utils.plot_utils import HDRenderer
from expose.config.cmd_parser import set_face_contour
from expose.config import cfg
from expose.models.smplx_net import SMPLXNet
from expose.data import make_all_data_loaders
from expose.utils.checkpointer import Checkpointer
from expose.data.targets.image_list import to_image_list
from expose.evaluation import Evaluator
from expose.optimizers import build_optimizer, build_scheduler

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))


Vec3d = o3d.utility.Vector3dVector
Vec3i = o3d.utility.Vector3iVector


from typing import Iterable


@torch.no_grad()
def main(exp_cfg):

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        logger.error('CUDA is not available!')
        sys.exit(3)

    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''),
               level=exp_cfg.logger_level.upper(),
               colorize=True)

    model = SMPLXNet(exp_cfg)
    try:
        model = model.to(device=device)
    except RuntimeError:
        # Re-submit in case of a device error
        sys.exit(3)
    model.train()

    optim_cfg = exp_cfg.get('optim', {})
    optimizer = build_optimizer(model, optim_cfg)
    lr_scheduler = build_scheduler(optimizer, optim_cfg['scheduler'])

    checkpoint_folder = osp.join(exp_cfg.output_folder, exp_cfg.checkpoint_folder)
    checkpointer = Checkpointer(model, optimizer=optimizer,
                                scheduler=lr_scheduler,
                                save_dir=checkpoint_folder,
                                pretrained=exp_cfg.pretrained)

    arguments = {'iteration': 0, 'epoch_number': 0}
    extra_checkpoint_data = checkpointer.load_checkpoint()
    for key in arguments:
        if key in extra_checkpoint_data:
            arguments[key] = extra_checkpoint_data[key]

    dataloaders = make_all_data_loaders(exp_cfg, split='train')
    dataloader = dataloaders['body']

    print("Start training")
    start_time = time.time()
    for epoch in range(arguments['epoch_number'], optim_cfg['num_epochs']):
        train_stats = train_one_epoch(model, dataloader, optimizer, device, epoch)
        lr_scheduler.step()
        Checkpointer.save_checkpoint('checkpoint.pth')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    # for idx, batch in enumerate(tqdm(dataloader, dynamic_ncols=True)):
    #     # full_imgs_list, body_imgs, body_targets = batch
    #     # for target in body_targets:
    #     #     vname = target.get_field('vname')
    #     #     if not osp.exists(vname):
    #     #         param = {}
    #     #         param['body_pose'] = target.get_field('body_pose').body_pose
    #     #         param['left_hand_pose'] = target.get_field('hand_pose').left_hand_pose
    #     #         param['right_hand_pose'] = target.get_field('hand_pose').right_hand_pose
    #     #         param['jaw_pose'] = target.get_field('jaw_pose').jaw_pose
    #     #         param['betas'] = target.get_field('betas').betas
    #     #         param['expression'] = target.get_field('expression').expression
    #     #         param['global_orient'] = target.get_field('global_pose').global_pose
    #     #         for key in param:
    #     #             param[key] = param[key].to(device).unsqueeze(0)
    #     #         output = model.smplx.body_model(get_skin=True, return_shaped=True, **param)
    #     #         vertice = output.vertices[0].cpu().numpy()
    #     #
    #     #         v_dir = osp.dirname(vname)
    #     #         if not osp.exists(v_dir):
    #     #             os.makedirs(v_dir)
    #     #         np.save(vname, vertice)
    #
    #     full_imgs_list, body_imgs, body_targets = batch
    #     hand_imgs, hand_targets = None, None
    #     head_imgs, head_targets = None, None
    #     full_imgs = to_image_list(full_imgs_list)
    #     if full_imgs is not None:
    #         full_imgs = full_imgs.to(device=device)
    #     body_imgs = body_imgs.to(device=device)
    #     body_targets = [target.to(device) for target in body_targets]
    #
    #     model_output = model(
    #         body_imgs, body_targets,
    #         hand_imgs=hand_imgs, hand_targets=hand_targets,
    #         head_imgs=head_imgs, head_targets=head_targets,
    #         full_imgs=full_imgs,
    #         device=device)
    #
    #     t = 0


def train_one_epoch(model: torch.nn.Module, dataloader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int):
    model.train()
    print_freq = 100

    for idx, batch in enumerate(tqdm(dataloader, dynamic_ncols=True)):

        full_imgs_list, body_imgs, body_targets = batch
        hand_imgs, hand_targets = None, None
        head_imgs, head_targets = None, None
        full_imgs = to_image_list(full_imgs_list)

        if full_imgs is not None:
            full_imgs = full_imgs.to(device=device)
        body_imgs = body_imgs.to(device=device)
        body_targets = [target.to(device) for target in body_targets]

        output = model(
            body_imgs, body_targets,
            hand_imgs=hand_imgs, hand_targets=hand_targets,
            head_imgs=head_imgs, head_targets=head_targets,
            full_imgs=full_imgs,
            device=device)
        loss_dict = output['losses']

        losses = sum(loss_dict[k] for k in loss_dict)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return 0


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'PyTorch SMPL-X Regressor Debug'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--exp-cfg', type=str, dest='exp_cfg',
                        help='The configuration of the experiment')
    parser.add_argument('--output-folder', dest='output_folder',
                        default='demo_output', type=str,
                        help='The folder where the demo renderings will be' +
                        ' saved')
    parser.add_argument('--datasets', nargs='+',
                        default=['openpose'], type=str,
                        help='Datasets to process')
    parser.add_argument('--show', default=False,
                        type=lambda arg: arg.lower() in ['true'],
                        help='Display the results')
    parser.add_argument('--pause', default=-1, type=float,
                        help='How much to pause the display')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
                        nargs='*', help='Extra command line arguments')
    parser.add_argument('--focal-length', dest='focal_length', type=float,
                        default=5000,
                        help='Focal length')
    parser.add_argument('--degrees', type=float, nargs='*', default=[],
                        help='Degrees of rotation around the vertical axis')
    parser.add_argument('--save-vis', dest='save_vis', default=False,
                        type=lambda x: x.lower() in ['true'],
                        help='Whether to save visualizations')
    parser.add_argument('--save-mesh', dest='save_mesh', default=False,
                        type=lambda x: x.lower() in ['true'],
                        help='Whether to save meshes')
    parser.add_argument('--save-params', dest='save_params', default=False,
                        type=lambda x: x.lower() in ['true'],
                        help='Whether to save parameters')

    parser.add_argument('--model-type', type=str, dest='model_type', default='exp',
                        help='network type')

    cmd_args = parser.parse_args()

    show = cmd_args.show
    output_folder = cmd_args.output_folder
    pause = cmd_args.pause
    focal_length = cmd_args.focal_length
    save_vis = cmd_args.save_vis
    save_params = cmd_args.save_params
    save_mesh = cmd_args.save_mesh
    degrees = cmd_args.degrees

    cfg.model_type = cmd_args.model_type
    cfg.merge_from_file(cmd_args.exp_cfg)
    cfg.merge_from_list(cmd_args.exp_opts)

    cfg.is_training = True
    cfg.datasets.body.splits.test = cmd_args.datasets
    use_face_contour = cfg.datasets.use_face_contour
    set_face_contour(cfg, use_face_contour=use_face_contour)

    with threadpool_limits(limits=1):
        main(cfg)
