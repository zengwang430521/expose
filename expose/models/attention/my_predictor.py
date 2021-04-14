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

from typing import List, Dict, Tuple, Callable, Optional, Union

from yacs.config import CfgNode
import time

from collections import defaultdict

import math
import os.path as osp

from loguru import logger

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hand_predictor import HandPredictor
from .head_predictor import HeadPredictor

from ..common.keypoint_loss import KeypointLoss
from ..common.smplx_loss_modules import SMPLXLossModule, RegularizerModule
from ..common.mix_loss_module import MyMixLossModule
from ..common.networks import FrozenBatchNorm2d
from ..common.mano_loss_modules import (
    MANOLossModule,
    RegularizerModule as MANORegularizer,
)
from ..common.flame_loss_modules import (
    FLAMELossModule,
    RegularizerModule as FLAMERegularizer
)


from smplx import build_layer as build_body_model
from smplx.utils import find_joint_kin_chain

from ..backbone import build_backbone
from ..common.networks import MLP, IterativeRegression
from ..common.bbox_sampler import CropSampler, ToCrops
from ..nnutils import init_weights
from ..common.pose_utils import build_all_pose_params
from ..camera import build_cam_proj, CameraParams
from ...losses import build_loss

from expose.data.targets import ImageList, ImageListPacked
from expose.data.targets.keypoints import KEYPOINT_NAMES, get_part_idxs
from expose.data.utils import flip_pose, bbox_iou, center_size_to_bbox

from expose.utils.typing_utils import Tensor


class SimpleSMPLXHead(nn.Module):

    def __init__(
            self,
            exp_cfg: CfgNode,
            dtype=torch.float32
    ) -> None:
        super().__init__()
        self.dtype = dtype

        # build smplx model
        self._build_smplx(exp_cfg)

        # build coefficients of SMPLX
        param_mean, param_dim, body_pose_mean, \
        left_hand_pose_mean, right_hand_pose_mean,\
        jaw_pose_mean = self._build_coefficinets(exp_cfg)

        # build networks
        self._build_network(exp_cfg, param_mean, param_dim)

        # build crop samplers and part joints idx
        self._build_sampler_idx(exp_cfg)

        # build regularizer
        loss_cfg = exp_cfg.get('losses', {})
        self.body_regularizer = RegularizerModule(
            loss_cfg, body_pose_mean=body_pose_mean,
            left_hand_pose_mean=left_hand_pose_mean,
            right_hand_pose_mean=right_hand_pose_mean,
            jaw_pose_mean=jaw_pose_mean
        )

        # build loss
        self._build_loss(exp_cfg)

    def _build_network(self, exp_cfg, param_mean, param_dim):
        # Construct the feature extraction backbone
        network_cfg = exp_cfg.get('network', {})
        attention_net_cfg = network_cfg.get('attention', {})
        smplx_net_cfg = attention_net_cfg.get('smplx', {})

        backbone_cfg = smplx_net_cfg.get('backbone', {})
        self.backbone, feat_dims = build_backbone(backbone_cfg)

        self.append_params = smplx_net_cfg.get('append_params', True)
        self.num_stages = smplx_net_cfg.get('num_stages', 1)

        self.body_feature_key = smplx_net_cfg.get('feature_key', 'avg_pooling')
        feat_dim = feat_dims[self.body_feature_key]

        # Build regressor
        regressor_cfg = smplx_net_cfg.get('mlp', {})
        regressor = MLP(feat_dim + self.append_params * param_dim, param_dim, **regressor_cfg)
        self.regressor = IterativeRegression(regressor, param_mean, num_stages=self.num_stages)

        self.freeze_body = attention_net_cfg.get('freeze_body', False)
        if self.freeze_body:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.regressor.parameters():
                param.requires_grad = False
            # Stop updating batch norm statistics
            self.backbone = FrozenBatchNorm2d.convert_frozen_batchnorm(self.backbone)
            self.regressor = FrozenBatchNorm2d.convert_frozen_batchnorm(self.regressor)

    def _build_smplx(self, exp_cfg):
        body_model_cfg = exp_cfg.get('body_model', {})
        dtype = self.dtype
        model_path = osp.expandvars(body_model_cfg.pop('model_folder', ''))
        model_type = body_model_cfg.pop('type', 'smplx')
        self.body_model = build_body_model(
            model_path,
            model_type=model_type,
            dtype=dtype,
            **body_model_cfg)
        logger.info(f'Body model: {self.body_model}')

    def _build_loss(self, exp_cfg):
        self.mix_loss = MyMixLossModule(
            exp_cfg,
        )

    def _build_sampler_idx(self, exp_cfg):
        network_cfg = exp_cfg.get('network', {})
        attention_net_cfg = network_cfg.get('attention', {})
        body_model_cfg = exp_cfg.get('body_model', {})
        body_use_face_contour = body_model_cfg.get('use_face_contour', True)

        # build crop samplers for hand and head
        hand_crop_size = exp_cfg.get('datasets', {}).get('hand', {}).get('transforms', {}).get('crop_size', 256)
        self.hand_scale_factor = attention_net_cfg.get('hand', {}).get('scale_factor', 2.0)
        self.hand_crop_size = hand_crop_size
        self.hand_cropper = CropSampler(hand_crop_size)

        head_crop_size = exp_cfg.get('datasets', {}).get('head', {}).get('transforms', {}).get('crop_size', 256)
        self.head_crop_size = head_crop_size
        self.head_scale_factor = network_cfg.get('head', {}).get('scale_factor', 2.0)
        self.head_cropper = CropSampler(head_crop_size)

        self.points_to_crops = ToCrops()

        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        head_idxs = idxs_dict['head']
        if not body_use_face_contour:
            head_idxs = head_idxs[:-17]

        self.register_buffer('body_idxs', torch.tensor(body_idxs))
        self.register_buffer('left_hand_idxs', torch.tensor(left_hand_idxs))
        self.register_buffer('right_hand_idxs', torch.tensor(right_hand_idxs))
        self.register_buffer('head_idxs', torch.tensor(head_idxs))

    def _build_coefficinets(self, exp_cfg):
        # Construct the feature extraction backbone
        network_cfg = exp_cfg.get('network', {})
        attention_net_cfg = network_cfg.get('attention', {})
        smplx_net_cfg = attention_net_cfg.get('smplx', {})
        body_model_cfg = exp_cfg.get('body_model', {})
        body_use_face_contour = body_model_cfg.get('use_face_contour', True)
        self.predict_body = network_cfg.get('predict_body', True)
        self.num_stages = smplx_net_cfg.get('num_stages', 3)
        self.append_params = smplx_net_cfg.get('append_params', True)
        self.pose_last_stage = smplx_net_cfg.get('pose_last_stage', False)
        self.body_model_cfg = body_model_cfg.copy()
        dtype = self.dtype

        # The number of shape coefficients
        num_betas = body_model_cfg.num_betas
        self.num_betas = num_betas

        shape_mean_path = body_model_cfg.get('shape_mean_path', '')
        shape_mean_path = osp.expandvars(shape_mean_path)
        if osp.exists(shape_mean_path):
            shape_mean = torch.from_numpy(
                np.load(shape_mean_path, allow_pickle=True)).to(
                dtype=dtype).reshape(1, -1)[:, :num_betas].reshape(-1)
        else:
            shape_mean = torch.zeros([num_betas], dtype=dtype)

        # The number of expression coefficients
        num_expression_coeffs = body_model_cfg.num_expression_coeffs
        self.num_expression_coeffs = num_expression_coeffs
        expression_mean = torch.zeros([num_expression_coeffs], dtype=dtype)

        # Build the pose parameterization for all the parameters
        pose_desc_dict = build_all_pose_params(
            body_model_cfg, 0, self.body_model,
            append_params=self.append_params, dtype=dtype)

        self.global_orient_decoder = pose_desc_dict['global_orient'].decoder
        global_orient_mean = pose_desc_dict['global_orient'].mean

        global_orient_type = body_model_cfg.get('global_orient', {}).get('param_type', 'cont_rot_repr')
        # Rotate the model 180 degrees around the x-axis
        if global_orient_type == 'aa':
            global_orient_mean[0] = math.pi
        elif global_orient_type == 'cont_rot_repr':
            global_orient_mean[3] = -1
        global_orient_dim = pose_desc_dict['global_orient'].dim

        self.body_pose_decoder = pose_desc_dict['body_pose'].decoder
        body_pose_mean = pose_desc_dict['body_pose'].mean
        body_pose_dim = pose_desc_dict['body_pose'].dim

        self.left_hand_pose_decoder = pose_desc_dict['left_hand_pose'].decoder
        left_hand_pose_mean = pose_desc_dict['left_hand_pose'].mean
        left_hand_pose_dim = pose_desc_dict['left_hand_pose'].dim
        left_hand_pose_ind_dim = pose_desc_dict['left_hand_pose'].ind_dim

        self.right_hand_pose_decoder = pose_desc_dict['right_hand_pose'].decoder
        right_hand_pose_mean = pose_desc_dict['right_hand_pose'].mean
        right_hand_pose_dim = pose_desc_dict['right_hand_pose'].dim
        right_hand_pose_ind_dim = pose_desc_dict['right_hand_pose'].ind_dim

        self.jaw_pose_decoder = pose_desc_dict['jaw_pose'].decoder
        jaw_pose_mean = pose_desc_dict['jaw_pose'].mean
        jaw_pose_dim = pose_desc_dict['jaw_pose'].dim

        mean_lst = []

        start = 0
        global_orient_idxs = list(range(start, start + global_orient_dim))
        global_orient_idxs = torch.tensor(global_orient_idxs, dtype=torch.long)
        self.register_buffer('global_orient_idxs', global_orient_idxs)
        start += global_orient_dim
        mean_lst.append(global_orient_mean.view(-1))

        body_pose_idxs = list(range(start, start + body_pose_dim))
        self.register_buffer('body_pose_idxs', torch.tensor(body_pose_idxs, dtype=torch.long))
        start += body_pose_dim
        mean_lst.append(body_pose_mean.view(-1))

        left_hand_pose_idxs = list(range(start, start + left_hand_pose_dim))
        self.register_buffer('left_hand_pose_idxs', torch.tensor(left_hand_pose_idxs, dtype=torch.long))
        start += left_hand_pose_dim
        mean_lst.append(left_hand_pose_mean.view(-1))

        right_hand_pose_idxs = list(range(start, start + right_hand_pose_dim))
        self.register_buffer('right_hand_pose_idxs', torch.tensor(right_hand_pose_idxs, dtype=torch.long))
        start += right_hand_pose_dim
        mean_lst.append(right_hand_pose_mean.view(-1))

        jaw_pose_idxs = list(range(start, start + jaw_pose_dim))
        self.register_buffer('jaw_pose_idxs', torch.tensor(jaw_pose_idxs, dtype=torch.long))
        start += jaw_pose_dim
        mean_lst.append(jaw_pose_mean.view(-1))

        shape_idxs = list(range(start, start + num_betas))
        self.register_buffer('shape_idxs', torch.tensor(shape_idxs, dtype=torch.long))
        start += num_betas
        mean_lst.append(shape_mean.view(-1))

        expression_idxs = list(range(start, start + num_expression_coeffs))
        self.register_buffer('expression_idxs', torch.tensor(expression_idxs, dtype=torch.long))
        start += num_expression_coeffs
        mean_lst.append(expression_mean.view(-1))

        camera_cfg = smplx_net_cfg.get('camera', {})
        camera_data = build_cam_proj(camera_cfg, dtype=dtype)
        self.projection = camera_data['camera']

        camera_param_dim = camera_data['dim']
        camera_mean = camera_data['mean']
        #  self.camera_mean = camera_mean
        self.register_buffer('camera_mean', camera_mean)
        self.camera_scale_func = camera_data['scale_func']

        camera_idxs = list(range(start, start + camera_param_dim))
        self.register_buffer('camera_idxs', torch.tensor(camera_idxs, dtype=torch.long))
        start += camera_param_dim
        mean_lst.append(camera_mean)

        param_mean = torch.cat(mean_lst).view(1, -1)
        param_dim = param_mean.numel()

        return param_mean, param_dim, body_pose_mean, \
               left_hand_pose_mean, right_hand_pose_mean, jaw_pose_mean

    def toggle_losses(self, iteration):
        self.body_loss.toggle_losses(iteration)
        self.keyp_loss.toggle_losses(iteration)

    def flat_body_params_to_dict(self, param_tensor):
        global_orient = torch.index_select(
            param_tensor, 1, self.global_orient_idxs)
        body_pose = torch.index_select(
            param_tensor, 1, self.body_pose_idxs)
        left_hand_pose = torch.index_select(
            param_tensor, 1, self.left_hand_pose_idxs)
        right_hand_pose = torch.index_select(
            param_tensor, 1, self.right_hand_pose_idxs)
        jaw_pose = torch.index_select(
            param_tensor, 1, self.jaw_pose_idxs)
        betas = torch.index_select(param_tensor, 1, self.shape_idxs)
        expression = torch.index_select(param_tensor, 1, self.expression_idxs)

        return {
            'betas': betas,
            'expression': expression,
            'global_orient': global_orient,
            'body_pose': body_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'jaw_pose': jaw_pose,
        }

    def find_joint_global_rotation(
            self,
            kin_chain: Tensor,
            root_pose: Tensor,
            body_pose: Tensor
    ) -> Tensor:
        ''' Computes the absolute rotation of a joint from the kinematic chain
        '''
        # Create a single vector with all the poses
        parents_pose = torch.cat(
            [root_pose, body_pose], dim=1)[:, kin_chain]
        output_pose = parents_pose[:, 0]
        for idx in range(1, parents_pose.shape[1]):
            output_pose = torch.bmm(
                parents_pose[:, idx], output_pose)
        return output_pose

    def forward(self,
                images: Tensor,
                targets: List = None,
                hand_imgs: Optional[Tensor] = None,
                hand_targets: Optional[List] = None,
                head_imgs: Optional[Tensor] = None,
                head_targets: Optional[List] = None,
                full_imgs: Optional[Union[ImageList, ImageListPacked]] = None,
                ) -> Dict[str, Dict[str, Tensor]]:
        ''' Forward pass of the attention predictor
        '''

        batch_size, _, crop_size, _ = images.shape
        device = images.device
        dtype = images.dtype

        # body_features = self.backbone(images)
        # body_parameters, body_deltas = self.regressor(body_features)

        feat_dict = self.backbone(images)
        body_features = feat_dict[self.body_feature_key]
        body_parameters, body_deltas = self.regressor(body_features)

        losses = {}
        toy_loss = sum(p.sum() for p in body_parameters) * 0.0 + \
                   sum(d.sum() for d in body_deltas) * 0.0 + \
                   sum(f.sum() for f in feat_dict.values()) * 0.0
        # body_features.sum() * 0.0

        losses['toy_loss'] = toy_loss

        # A list of dicts for the parameters predicted at each stage. The key
        # is the name of the parameters and the value is the prediction of the
        # model at the i-th stage of the iteration
        param_dicts = []

        # A dict of lists. Each key is the name of the parameter and the
        # corresponding item is a list of offsets that are predicted by the model
        deltas_dict = defaultdict(lambda: [])

        param_delta_iter = zip(body_parameters, body_deltas)
        for idx, (params, deltas) in enumerate(param_delta_iter):
            curr_params_dict = self.flat_body_params_to_dict(params)

            out_dict = {}
            for key, val in curr_params_dict.items():
                if hasattr(self, f'{key}_decoder'):
                    decoder = getattr(self, f'{key}_decoder')
                    out_dict[key] = decoder(val)
                    out_dict[f'raw_{key}'] = val.clone()
                else:
                    out_dict[key] = val

            param_dicts.append(out_dict)
            curr_params_dict.clear()
            for key, val in self.flat_body_params_to_dict(deltas).items():
                deltas_dict[key].append(val)

        for key in deltas_dict:
            deltas_dict[key] = torch.stack(deltas_dict[key], dim=1).sum(dim=1)

        if self.pose_last_stage:
            merged_params = param_dicts[-1]
        else:
            merged_params = {}
            for key in param_dicts[0].keys():
                param = []
                for idx in range(self.num_stages):
                    if param_dicts[idx][key] is None:
                        continue
                    param.append(param_dicts[idx][key])
                merged_params[key] = torch.cat(param, dim=0)

        # Compute the body surface using the current estimation of the pose and shape
        body_model_output = self.body_model(get_skin=True, return_shaped=True, **merged_params)

        # Split the vertices, joints, etc. to stages
        out_params = defaultdict(lambda: dict())
        for key in body_model_output:
            if torch.is_tensor(body_model_output[key]):
                curr_val = body_model_output[key]
                out_list = torch.split(curr_val, batch_size, dim=0)
                # If the number of outputs is equal to the number of stages
                # then store each stage
                if len(out_list) == self.num_stages:
                    for idx in range(len(out_list)):
                        out_params[f'stage_{idx:02d}'][key] = out_list[idx]
                # Else add only the last
                else:
                    out_key = f'stage_{self.num_stages - 1:02d}'
                    out_params[out_key][key] = out_list[-1]
                    param_dicts[self.num_stages-1][key] = out_list[-1]

        # Add the predicted parameters to the output dictionary
        for stage in range(self.num_stages):
            stage_key = f'stage_{stage:02d}'
            if len(out_params[stage_key]) < 1:
                continue
            out_params[stage_key].update(param_dicts[stage])
            out_params[stage_key]['faces'] = self.body_model.faces

        # Extract the camera parameters estimated by the body only image
        camera_params = torch.index_select(body_parameters[-1], 1, self.camera_idxs)
        scale = camera_params[:, 0].view(-1, 1)
        translation = camera_params[:, 1:3]
        # Pass the predicted scale through exp() to make sure that the
        # scale values are always positive
        scale = self.camera_scale_func(scale)

        # Project the joints on the image plane
        proj_joints = self.projection(
            out_params[f'stage_{self.num_stages - 1:02d}']['joints'],
            scale=scale, translation=translation)

        # Add the projected joints
        out_params['proj_joints'] = proj_joints
        # the number of stages
        out_params['num_stages'] = self.num_stages
        # and the camera parameters to the output
        out_params['camera_parameters'] = CameraParams(translation=translation, scale=scale)

        if self.training:
            # joints3d = out_params[f'stage_{self.num_stages - 1:02d}']['joints']
            # losses.update(self.keyp_loss(proj_joints, joints3d, targets, device))
            # losses.update(self.body_loss(param_dicts, targets, self.num_stages, device))

            pred_dict = param_dicts[-1]
            pred_dict['proj_joints'] = proj_joints
            losses.update(self.mix_loss(pred_dict, targets))

            # # Create the tensor of ground-truth HD keypoints
            # gt_crop_keypoints = []
            # gt_hd_keypoints = []
            # for t in targets:
            #     gt_hd_keypoints.append(t.get_field('keypoints_hd'))
            #     gt_crop_keypoints.append(t.get_field('keypoints3d'))
            #
            #
            # gt_hd_keypoints_with_conf = torch.tensor(gt_hd_keypoints, dtype=dtype, device=device)
            # gt_hd_keypoints_conf = gt_hd_keypoints_with_conf[:, :, -1]
            # gt_hd_keypoints = gt_hd_keypoints_with_conf[:, :, :-1]
            # out_params['gt_conf'] = gt_hd_keypoints_conf.detach()


        output = {
            'body': out_params,
            'losses': losses
        }

        return output
