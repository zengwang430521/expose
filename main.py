import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import argparse
import datetime
import json
import random
import time
import numpy as np
import torch
import util.misc as utils
import util.samplers as samplers
import functools
from torch.utils.data import DataLoader
from util.train_one_epoch import train_one_epoch
from expose.config.cmd_parser import set_face_contour
from expose.config import cfg
from expose.models.smplx_net import SMPLXNet
from expose.data.build import make_all_datasets, collate_batch
from expose.optimizers import build_optimizer, build_scheduler
from torch.utils.data import ConcatDataset

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)


def get_args_parser():
    parser = argparse.ArgumentParser('EXPOSE train', add_help=False)

    parser.add_argument('--exp-cfg', type=str, dest='exp_cfg',
                        help='The configuration of the experiment')
    parser.add_argument('--output-folder', dest='output_folder',
                        default='demo_output', type=str,
                        help='The folder where the demo renderings will be' +
                        ' saved')
    parser.add_argument('--datasets', nargs='+',
                        default=['openpose'], type=str,
                        help='Datasets to process')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
                        nargs='*', help='Extra command line arguments')
    parser.add_argument('--model-type', type=str, dest='model_type', default='exp',
                        help='network type')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pretrain', default='', help='pretrained checkpoint')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=50, type=int)

    return parser


def main(args, exp_cfg):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model, criterion, postprocessors = build_model(args)
    model = SMPLXNet(exp_cfg)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # dataset_train = build_dataset(image_set='train', args=args)
    # dataset_val = build_dataset(image_set='val', args=args)
    datasets = make_all_datasets(exp_cfg, split='train')
    dataset_train = ConcatDataset(datasets['body'])

    sample_weight = [child_dataset.sample_weight for child_dataset in dataset_train.datasets]
    sample_weight = np.concatenate(sample_weight, axis=0)
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(dataset_train))
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.distributed:
        sampler_train = samplers.DistributedSampler(sampler_train)
        # sampler_val = samplers.DistributedSampler(sampler_val, shuffle=False)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    collate_fn = functools.partial(collate_batch,
                                   use_shared_memory=args.num_workers > 0,
                                   return_full_imgs=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)

    # data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
    #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
    #                              pin_memory=True)

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)

    optim_cfg = exp_cfg.get('optim', {})
    optimizer = build_optimizer(model, optim_cfg)
    lr_scheduler = build_scheduler(optimizer, optim_cfg['scheduler'])

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.pretrain:
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch)
        lr_scheduler.step()

        if args.output_dir:
            output_dir = args.output_dir
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 1 epochs
            if (epoch + 1) % args.save_freq == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EXPose train script', parents=[get_args_parser()])
    cmd_args = parser.parse_args()
    cfg.model_type = cmd_args.model_type
    cfg.merge_from_file(cmd_args.exp_cfg)
    cfg.merge_from_list(cmd_args.exp_opts)
    cfg.is_training = True
    use_face_contour = cfg.datasets.use_face_contour
    set_face_contour(cfg, use_face_contour=use_face_contour)

    main(cmd_args, cfg)
