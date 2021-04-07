from tqdm import tqdm
from expose.data.targets.image_list import to_image_list
from typing import Iterable
import torch
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils


def train_one_epoch(model: torch.nn.Module, dataloader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    data_iter = iter(dataloader)

    for _ in metric_logger.log_every(range(len(dataloader)), print_freq, header):
        batch = data_iter.next()

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

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced = {f'{k}': v for k, v in loss_dict_reduced.items()}
        losses_reduced = sum(loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # batch = data_iter.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
