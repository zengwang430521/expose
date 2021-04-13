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
import time
from typing import Dict, NewType

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict

from loguru import logger

from .rigid_alignment import RotationTranslationAlignment
from ...data.targets.keypoints import get_part_idxs
from ...losses import build_loss, build_prior

from ...data.targets.keypoints import (
    get_part_idxs, KEYPOINT_NAMES,
    BODY_CONNECTIONS, HAND_CONNECTIONS, FACE_CONNECTIONS)
from ...data.utils import points_to_bbox

Tensor = NewType('Tensor', torch.Tensor)


# def align_2d_points(origin_2d, target_2d):
#     tmp_o = origin_2d - origin_2d.mean(dim=0)
#     tmp_t = target_2d - target_2d.mean(dim=0)
#     scale = (tmp_t * tmp_o).sum() / (tmp_o * tmp_o).sum()
#     trans = target_2d.mean(dim=0) / scale - origin_2d.mean(dim=0)
#
#     # err = (origin_2d + trans) * scale - target_2d
#     # err = err.norm(p=2, dim=1).mean()
#     origin_aligned = (origin_2d + trans) * scale
#     return origin_aligned, (scale, trans)

# align points with only scale and translation, without rotation
def align_points(origin, target, conf):
    tmp_o = origin - origin.mean(dim=1, keepdim=True)
    tmp_t = target - target.mean(dim=1, keepdim=True)

    scale = (conf * tmp_t * tmp_o).sum(dim=-1).sum(dim=-1) \
            / ((conf * tmp_o * tmp_o).sum(dim=-1).sum(dim=-1) + 1e-8)
    scale = scale[:, None, None]

    trans = ((target - origin * scale) * conf).sum(dim=1, keepdim=True) \
            / (conf.sum(dim=1, keepdim=True) * scale + 1e-8)

    origin_aligned = (origin + trans) * scale
    return origin_aligned, (scale, trans)


class MyMixLossModule(nn.Module):
    '''
    Loss module to calculate all losses.
    '''

    def __init__(self, exp_cfg):
        super().__init__()
        self.loss_enabled = defaultdict(lambda: True)
        self.loss_activ_step = {}

        body_model_cfg = exp_cfg.get('body_model', {})
        body_use_face_contour = body_model_cfg.get('use_face_contour', True)
        loss_cfg = exp_cfg.get('losses', {})
        self.init_smplx_loss(loss_cfg, body_use_face_contour)
        self.init_keypoints_loss(exp_cfg)

    def init_smplx_loss(self, loss_cfg, use_face_contour=False):
        # self.stages_to_penalize = loss_cfg.get('stages_to_penalize', [-1])
        # logger.info(f'Stages to penalize: {self.stages_to_penalize}')

        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        face_idxs = idxs_dict['face']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        if not use_face_contour:
            face_idxs = face_idxs[:-17]

        self.register_buffer('body_idxs', torch.tensor(body_idxs))
        self.register_buffer('hand_idxs', torch.tensor(hand_idxs))
        self.register_buffer('face_idxs', torch.tensor(face_idxs))
        self.register_buffer('left_hand_idxs', torch.tensor(left_hand_idxs))
        self.register_buffer('right_hand_idxs', torch.tensor(right_hand_idxs))

        shape_loss_cfg = loss_cfg.shape
        self.shape_weight = shape_loss_cfg.get('weight', 0.0)
        self.shape_loss = build_loss(**shape_loss_cfg)
        self.loss_activ_step['shape'] = shape_loss_cfg.enable

        expression_cfg = loss_cfg.get('expression', {})
        self.expr_use_conf_weight = expression_cfg.get(
            'use_conf_weight', False)

        self.expression_weight = expression_cfg.weight
        if self.expression_weight > 0:
            self.expression_loss = build_loss(**expression_cfg)
            self.loss_activ_step['expression'] = expression_cfg.enable

        global_orient_cfg = loss_cfg.global_orient
        global_orient_loss_type = global_orient_cfg.type
        self.global_orient_loss_type = global_orient_loss_type
        self.global_orient_loss = build_loss(**global_orient_cfg)
        logger.debug('Global pose loss: {}', self.global_orient_loss)
        self.global_orient_weight = global_orient_cfg.weight
        self.loss_activ_step['global_orient'] = global_orient_cfg.enable

        self.body_pose_weight = loss_cfg.body_pose.weight
        body_pose_loss_type = loss_cfg.body_pose.type
        self.body_pose_loss_type = body_pose_loss_type
        self.body_pose_loss = build_loss(**loss_cfg.body_pose)
        logger.debug('Body pose loss: {}', self.global_orient_loss)
        self.body_pose_weight = loss_cfg.body_pose.weight
        self.loss_activ_step['body_pose'] = loss_cfg.body_pose.enable

        left_hand_pose_cfg = loss_cfg.get('left_hand_pose', {})
        left_hand_pose_loss_type = loss_cfg.left_hand_pose.type
        self.lhand_use_conf = left_hand_pose_cfg.get('use_conf_weight', False)

        self.left_hand_pose_weight = loss_cfg.left_hand_pose.weight
        if self.left_hand_pose_weight > 0:
            self.left_hand_pose_loss_type = left_hand_pose_loss_type
            self.left_hand_pose_loss = build_loss(**loss_cfg.left_hand_pose)
            self.loss_activ_step[
                'left_hand_pose'] = loss_cfg.left_hand_pose.enable

        right_hand_pose_cfg = loss_cfg.get('right_hand_pose', {})
        right_hand_pose_loss_type = loss_cfg.right_hand_pose.type
        self.right_hand_pose_weight = loss_cfg.right_hand_pose.weight
        self.rhand_use_conf = right_hand_pose_cfg.get('use_conf_weight', False)
        if self.right_hand_pose_weight > 0:
            self.right_hand_pose_loss_type = right_hand_pose_loss_type
            self.right_hand_pose_loss = build_loss(**loss_cfg.right_hand_pose)
            self.loss_activ_step[
                'right_hand_pose'] = loss_cfg.right_hand_pose.enable

        jaw_pose_loss_type = loss_cfg.jaw_pose.type
        self.jaw_pose_weight = loss_cfg.jaw_pose.weight

        jaw_pose_cfg = loss_cfg.get('jaw_pose', {})
        self.jaw_use_conf_weight = jaw_pose_cfg.get('use_conf_weight', False)
        if self.jaw_pose_weight > 0:
            self.jaw_pose_loss_type = jaw_pose_loss_type
            self.jaw_pose_loss = build_loss(**loss_cfg.jaw_pose)
            logger.debug('Jaw pose loss: {}', self.global_orient_loss)
            self.loss_activ_step['jaw_pose'] = loss_cfg.jaw_pose.enable

        edge_loss_cfg = loss_cfg.get('edge', {})
        self.edge_weight = edge_loss_cfg.get('weight', 0.0)
        self.edge_loss = build_loss(**edge_loss_cfg)
        self.loss_activ_step['edge'] = edge_loss_cfg.get('enable', 0)

    def init_keypoints_loss(self, exp_cfg):
        loss_cfg = exp_cfg.get('losses', {})

        self.left_hip_idx = KEYPOINT_NAMES.index('left_hip')
        self.right_hip_idx = KEYPOINT_NAMES.index('right_hip')

        # global 2D loss
        self.body_joints_2d_weight = exp_cfg.losses.body_joints_2d.weight
        if self.body_joints_2d_weight > 0:
            self.body_joints_2d_loss = build_loss(**exp_cfg.losses.body_joints_2d)
            logger.debug('2D body joints loss: {}', self.body_joints_2d_loss)

        hand_joints2d_cfg = exp_cfg.losses.hand_joints_2d
        self.hand_joints_2d_weight = hand_joints2d_cfg.weight
        self.hand_joints_2d_enable_at = hand_joints2d_cfg.enable
        self.hand_joints_2d_active = True
        if self.hand_joints_2d_weight > 0:
            hand_joints2d_cfg = exp_cfg.losses.hand_joints_2d
            self.hand_joints_2d_loss = build_loss(**hand_joints2d_cfg)
            logger.debug('2D hand joints loss: {}', self.hand_joints_2d_loss)

        face_joints2d_cfg = exp_cfg.losses.face_joints_2d
        self.face_joints_2d_weight = face_joints2d_cfg.weight
        self.face_joints_2d_enable_at = face_joints2d_cfg.enable
        self.face_joints_2d_active = True
        if self.face_joints_2d_weight > 0:
            self.face_joints_2d_loss = build_loss(**face_joints2d_cfg)
            logger.debug('2D face joints loss: {}', self.face_joints_2d_loss)

        use_face_contour = exp_cfg.datasets.use_face_contour
        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['left_hand']
        face_idxs = idxs_dict['face']
        if not use_face_contour:
            face_idxs = face_idxs[:-17]
        self.register_buffer('body_idxs', torch.tensor(body_idxs))
        self.register_buffer('hand_idxs', torch.tensor(hand_idxs))
        self.register_buffer('left_hand_idxs', torch.tensor(left_hand_idxs))
        self.register_buffer('right_hand_idxs', torch.tensor(right_hand_idxs))
        self.register_buffer('face_idxs', torch.tensor(face_idxs))

        # global 3D joints loss
        self.body_joints_3d_weight = exp_cfg.losses.body_joints_3d.weight
        if self.body_joints_3d_weight > 0:
            self.body_joints_3d_loss = build_loss(**exp_cfg.losses.body_joints_3d)
            logger.debug('3D body_joints loss: {}', self.body_joints_3d_loss)

        hand_joints3d_cfg = exp_cfg.losses.hand_joints_3d
        # print('debug!!!!!!!!!!!'); hand_joints3d_cfg.weight = 1
        self.hand_joints_3d_weight = hand_joints3d_cfg.weight
        self.hand_joints_3d_enable_at = hand_joints3d_cfg.enable
        if self.hand_joints_3d_weight > 0:
            self.hand_joints_3d_loss = build_loss(**hand_joints3d_cfg)
            logger.debug('3D hand joints loss: {}', self.hand_joints_3d_loss)
        self.hand_joints_3d_active = True

        face_joints3d_cfg = exp_cfg.losses.face_joints_3d
        self.face_joints_3d_weight = face_joints3d_cfg.weight
        self.face_joints_3d_enable_at = face_joints3d_cfg.enable
        if self.face_joints_3d_weight > 0:
            face_joints3d_cfg = exp_cfg.losses.face_joints_3d
            self.face_joints_3d_loss = build_loss(**face_joints3d_cfg)
            logger.debug('3D face joints loss: {}', self.face_joints_3d_loss)
        self.face_joints_3d_active = True

        body_edge_2d_cfg = exp_cfg.losses.get('body_edge_2d', {})
        self.body_edge_2d_weight = body_edge_2d_cfg.weight
        self.body_edge_2d_enable_at = body_edge_2d_cfg.enable
        if self.body_edge_2d_weight > 0:
            self.body_edge_2d_loss = build_loss(type='keypoint-edge',
                                                connections=BODY_CONNECTIONS,
                                                **body_edge_2d_cfg)
            logger.debug('2D body edge loss: {}', self.body_edge_2d_loss)
        self.body_edge_2d_active = True

        hand_edge_2d_cfg = exp_cfg.losses.get('hand_edge_2d', {})
        self.hand_edge_2d_weight = hand_edge_2d_cfg.get('weight', 0.0)
        self.hand_edge_2d_enable_at = hand_edge_2d_cfg.get('enable', 0)
        if self.hand_edge_2d_weight > 0:
            self.hand_edge_2d_loss = build_loss(type='keypoint-edge',
                                                connections=HAND_CONNECTIONS,
                                                **hand_edge_2d_cfg)
            logger.debug('2D hand edge loss: {}', self.hand_edge_2d_loss)
        self.hand_edge_2d_active = True

        face_edge_2d_cfg = exp_cfg.losses.get('face_edge_2d', {})
        self.face_edge_2d_weight = face_edge_2d_cfg.get('weight', 0.0)
        self.face_edge_2d_enable_at = face_edge_2d_cfg.get('enable', 0)
        if self.face_edge_2d_weight > 0:
            face_connections = []
            for conn in FACE_CONNECTIONS:
                if ('contour' in KEYPOINT_NAMES[conn[0]] or
                        'contour' in KEYPOINT_NAMES[conn[1]]):
                    if not use_face_contour:
                        continue
                face_connections.append(conn)

            self.face_edge_2d_loss = build_loss(
                type='keypoint-edge', connections=face_connections,
                **face_edge_2d_cfg)
            logger.debug('2D face edge loss: {}', self.face_edge_2d_loss)
        self.face_edge_2d_active = True


        # local 2D joints loss
        head_crop_keypoint_loss_cfg = loss_cfg.get('head_crop_keypoints')
        self.head_crop_keyps_weight = head_crop_keypoint_loss_cfg.get('weight', 0.0)
        self.head_crop_keyps_enable_at = head_crop_keypoint_loss_cfg.get('enable', True)
        if self.head_crop_keyps_weight > 0:
            self.head_crop_keyps_loss = build_loss(**head_crop_keypoint_loss_cfg)
            logger.info('2D Head crop keyps loss: {}', self.head_crop_keyps_loss)

        left_hand_crop_keypoint_loss_cfg = loss_cfg.get('left_hand_crop_keypoints')
        self.left_hand_crop_keyps_weight = (left_hand_crop_keypoint_loss_cfg.get('weight', 0.0))
        self.left_hand_crop_keyps_enable_at = (left_hand_crop_keypoint_loss_cfg.get('enable', True))
        if self.left_hand_crop_keyps_weight > 0:
            self.left_hand_crop_keyps_loss = build_loss(**left_hand_crop_keypoint_loss_cfg)
            logger.info('2D Left hand crop keyps loss: {}', self.left_hand_crop_keyps_loss)

        right_hand_crop_keypoint_loss_cfg = loss_cfg.get('right_hand_crop_keypoints')
        self.right_hand_crop_keyps_weight = (right_hand_crop_keypoint_loss_cfg.get('weight', 0.0))
        self.right_hand_crop_keyps_enable_at = (right_hand_crop_keypoint_loss_cfg.get('enable', True))
        if self.right_hand_crop_keyps_weight > 0:
            self.right_hand_crop_keyps_loss = build_loss(**right_hand_crop_keypoint_loss_cfg)
            logger.info('2D Left hand crop keyps loss: {}', self.right_hand_crop_keyps_loss)

        # global 3D joints loss
        self.local_hand_3d_weight = hand_joints3d_cfg.weight
        self.local_face_3d_weight = face_joints3d_cfg.weight

    def is_active(self) -> bool:
        return any(self.loss_enabled.values())

    def toggle_losses(self, step) -> None:
        for key in self.loss_activ_step:
            self.loss_enabled[key] = step >= self.loss_activ_step[key]

    def extra_repr(self) -> str:
        msg = []
        if self.shape_weight > 0:
            msg.append(f'Shape weight: {self.shape_weight}')
        if self.expression_weight > 0:
            msg.append(f'Expression weight: {self.expression_weight}')
        if self.global_orient_weight > 0:
            msg.append(f'Global pose weight: {self.global_orient_weight}')
        if self.body_pose_weight > 0:
            msg.append(f'Body pose weight: {self.body_pose_weight}')
        if self.left_hand_pose_weight > 0:
            msg.append(f'Left hand pose weight: {self.left_hand_pose_weight}')
        if self.right_hand_pose_weight > 0:
            msg.append(f'Right hand pose weight {self.right_hand_pose_weight}')
        if self.jaw_pose_weight > 0:
            msg.append(f'Jaw pose prior weight: {self.jaw_pose_weight}')
        return '\n'.join(msg)

    def smplx_loss(self, pred_params, target_params, conf_params):
        losses = {}

        compute_edge_loss = self.edge_weight > 0
        if compute_edge_loss:
            edge_loss_val = self.edge_loss(
                est_vertices=pred_params['vertices'],
                gt_vertices=target_params['vertices'],
            ) * self.edge_weight
            losses['mesh_edge_loss'] = edge_loss_val

        compute_shape_loss = (self.shape_weight > 0 and
                              self.loss_enabled['betas'] and
                              'betas' in target_params)
        if compute_shape_loss:
            losses['shape_loss'] = self.shape_loss(pred_params['betas'],
                                                   target_params['betas'],
                                                   conf_params['betas']) \
                                   * self.shape_weight

        compute_expr_loss = (self.expression_weight > 0 and
                             self.loss_enabled['expression'] and
                             'expression' in target_params)
        if compute_expr_loss:
            losses['expression_loss'] = self.expression_loss(pred_params['expression'],
                                                             target_params['expression'],
                                                             conf_params['expression']) \
                                        * self.expression_weight

        compute_global_orient_loss = (self.global_orient_weight > 0 and
                                      self.loss_enabled['betas'] and
                                      'global_orient' in target_params)

        if compute_global_orient_loss:
            losses['global_orient_loss'] = self.global_orient_loss(
                pred_params['global_orient'],
                target_params['global_orient'],
                conf_params['global_orient']
            ) * self.global_orient_weight

        compute_body_pose_loss = (self.body_pose_weight > 0 and
                                  self.loss_enabled['betas'] and
                                  'body_pose' in target_params)

        if compute_body_pose_loss:
            losses['body_pose_loss'] = self.body_pose_loss(
                pred_params['body_pose'],
                target_params['body_pose'],
                conf_params['body_pose']
            ) * self.body_pose_weight

        compute_left_hand_loss = (self.left_hand_pose_weight > 0
                                  and self.loss_enabled['left_hand_pose']
                                  and 'left_hand_pose' in target_params)

        if compute_left_hand_loss:
            losses['left_hand_pose_loss'] = self.left_hand_pose_loss(
                pred_params['left_hand_pose'],
                target_params['left_hand_pose'],
                conf_params['left_hand_pose']
            ) * self.left_hand_pose_weight

        compute_right_hand_loss = (self.right_hand_pose_weight > 0
                                   and self.loss_enabled['right_hand_pose']
                                   and 'right_hand_pose' in target_params)

        if compute_right_hand_loss:
            losses['right_hand_pose_loss'] = self.right_hand_pose_loss(
                pred_params['right_hand_pose'],
                target_params['right_hand_pose'],
                conf_params['right_hand_pose']
            ) * self.right_hand_pose_weight

        compute_jaw_loss = (self.jaw_pose_weight > 0
                            and self.loss_enabled['jaw_pose']
                            and 'jaw_pose' in target_params)

        if compute_jaw_loss:
            losses['jaw_pose_loss'] = self.jaw_pose_loss(
                pred_params['jaw_pose'],
                target_params['jaw_pose'],
                conf_params['jaw_pose']
            ) * self.jaw_pose_weight

        return losses

    def local_loss(self, pred_joints, target_joints, conf_joints, loss_func):
        valid_target = target_joints * (conf_joints[..., None] > 0)
        target_bbox_size = (valid_target.max(dim=1)[0] - valid_target.min(dim=1)[0]).max(dim=1)[0] + 1e-8
        target_bbox_size = target_bbox_size[:, None, None]
        # rescale
        pred_joints = pred_joints / target_bbox_size
        target_joints = target_joints / target_bbox_size

        # align
        pred_joints_aligned, _ = align_points(pred_joints, target_joints, conf_joints[..., None])

        return loss_func(pred_joints_aligned, target_joints, conf_joints)

    def keypoints_loss(self, proj_joints, joints3d, targets):
        losses = {}
        # 2D joints loss in body crop coordinates
        target_keypoints2d = torch.stack([target.smplx_keypoints for target in targets])
        target_conf = torch.stack([target.conf for target in targets])

        if self.body_joints_2d_weight > 0:
            body_joints_2d_loss = self.body_joints_2d_loss(
                proj_joints[:, self.body_idxs],
                target_keypoints2d[:, self.body_idxs],
                weights=target_conf[:, self.body_idxs]
            ) * self.body_joints_2d_weight
            losses.update(body_joints_2d_loss=body_joints_2d_loss)

        if self.hand_joints_2d_active and self.hand_joints_2d_weight > 0:
            hand_joints_2d_loss = self.hand_joints_2d_loss(
                proj_joints[:, self.hand_idxs],
                target_keypoints2d[:, self.hand_idxs],
                weights=target_conf[:, self.hand_idxs]
            ) * self.hand_joints_2d_weight
            losses.update(hand_joints_2d_loss=hand_joints_2d_loss)

        if self.face_joints_2d_active and self.face_joints_2d_weight > 0:
            face_joints_2d_loss = self.face_joints_2d_loss(
                proj_joints[:, self.face_idxs],
                target_keypoints2d[:, self.face_idxs],
                weights=target_conf[:, self.face_idxs]
            ) * self.face_joints_2d_weight
            losses.update(face_joints_2d_loss=face_joints_2d_loss)

        if self.body_edge_2d_weight > 0 and self.body_edge_2d_active:
            body_edge_2d_loss = self.body_edge_2d_loss(
                proj_joints,
                target_keypoints2d,
                weights=target_conf
            ) * self.body_edge_2d_weight
            losses.update(body_edge_2d_loss=body_edge_2d_loss)

        if self.hand_edge_2d_weight > 0 and self.hand_edge_2d_active:
            hand_edge_2d_loss = self.hand_edge_2d_loss(
                proj_joints,
                target_keypoints2d,
                weights=target_conf
            ) * self.hand_edge_2d_weight
            losses.update(hand_edge_2d_loss=hand_edge_2d_loss)

        if self.face_edge_2d_weight > 0 and self.face_edge_2d_active:
            face_edge_2d_loss = self.face_edge_2d_loss(
                proj_joints,
                target_keypoints2d,
                weights=target_conf
            ) * self.face_edge_2d_weight
            losses.update(face_edge_2d_loss=face_edge_2d_loss)

        # 2D joints loss in local crop coordinates.
        if self.head_crop_keyps_weight > 0:
            local_face_2d_loss = self.local_loss(
                proj_joints[:, self.face_idxs],
                target_keypoints2d[:, self.face_idxs],
                target_conf[:, self.face_idxs],
                self.head_crop_keyps_loss
            ) * self.head_crop_keyps_weight
            losses.update(local_face_2d_loss=local_face_2d_loss)

        if self.left_hand_crop_keyps_weight > 0:
            local_left_hand_2d_loss = self.local_loss(
                proj_joints[:, self.left_hand_idxs],
                target_keypoints2d[:, self.left_hand_idxs],
                target_conf[:, self.left_hand_idxs],
                self.head_crop_keyps_loss
            ) * self.head_crop_keyps_weight
            losses.update(local_left_hand_2d_loss=local_left_hand_2d_loss)

        if self.right_hand_pose_weight > 0:
            local_right_hand_2d_loss = self.local_loss(
                proj_joints[:, self.right_hand_idxs],
                target_keypoints2d[:, self.right_hand_idxs],
                target_conf[:, self.right_hand_idxs],
                self.head_crop_keyps_loss
            ) * self.head_crop_keyps_weight
            losses.update(local_right_hand_2d_loss=local_right_hand_2d_loss)

        #  3D joints loss in global coordinates
        if self.body_joints_3d_weight > 0 and joints3d is not None:
            # Get 3D keypoints
            target_joints3d = joints3d.new_zeros(joints3d.shape)
            conf_joints3d = joints3d.new_zeros(joints3d.shape[0], joints3d.shape[1])
            for idx, target in enumerate(targets):
                if target.has_field('keypoints3d'):
                    target_joints3d[idx] = target.get_field('keypoints3d').smplx_keypoints
                    conf_joints3d[idx] = target.get_field('keypoints3d')['conf']

            # Center the predictions using the pelvis
            pred_pelvis = joints3d[
                          :, [self.left_hip_idx, self.right_hip_idx], :
                          ].mean(dim=1, keepdim=True)
            centered_pred_joints = joints3d - pred_pelvis

            gt_pelvis = target_joints3d[
                        :, [self.left_hip_idx, self.right_hip_idx], :
                        ].mean(dim=1, keepdim=True)
            centered_gt_joints = target_joints3d - gt_pelvis

            body_joints_3d_loss = self.body_joints_3d_loss(
                centered_pred_joints[:, self.body_idxs],
                centered_gt_joints[:, self.body_idxs],
                weights=conf_joints3d[:, self.body_idxs]
            ) * self.body_joints_3d_weight
            losses.update(body_joints_3d_loss=body_joints_3d_loss)

            if self.hand_joints_3d_active and self.hand_joints_3d_weight > 0:
                hand_joints_3d_loss = self.hand_joints_3d_loss(
                    centered_pred_joints[:, self.hand_idxs],
                    centered_gt_joints[:, self.hand_idxs],
                    weights=conf_joints3d[:, self.hand_idxs]
                ) * self.hand_joints_3d_weight
                losses.update(hand_joints_3d_loss=hand_joints_3d_loss)

            if self.face_joints_3d_active and self.face_joints_3d_weight > 0:
                face_joints_3d_loss = self.face_joints_3d_loss(
                    centered_pred_joints[:, self.face_idxs],
                    centered_gt_joints[:, self.face_idxs],
                    weights=conf_joints3d[:, self.face_idxs]
                ) * self.face_joints_3d_weight
                losses.update(face_joints_3d_loss=face_joints_3d_loss)

            #  3D joints loss in local coordinates
            if self.hand_joints_3d_active and self.local_hand_3d_weight > 0:
                local_lhand_3d_loss = self.local_loss(
                    centered_pred_joints[:, self.left_hand_idxs],
                    centered_gt_joints[:, self.left_hand_idxs],
                    conf_joints3d[:, self.left_hand_idxs],
                    self.hand_joints_3d_loss
                ) * self.local_hand_3d_weight
                losses.update(local_lhand_3d_loss=local_lhand_3d_loss)

                local_rhand_3d_loss = self.local_loss(
                    centered_pred_joints[:, self.right_hand_idxs],
                    centered_gt_joints[:, self.right_hand_idxs],
                    conf_joints3d[:, self.left_hand_idxs],
                    self.hand_joints_3d_loss
                ) * self.local_hand_3d_weight
                losses.update(local_rhand_3d_loss=local_rhand_3d_loss)

            if self.face_joints_3d_active and self.local_face_3d_weight > 0:
                local_face_3d_loss = self.local_loss(
                    centered_pred_joints[:, self.face_idxs],
                    centered_gt_joints[:, self.face_idxs],
                    conf_joints3d[:, self.face_idxs],
                    self.hand_joints_3d_loss
                ) * self.local_hand_3d_weight
                losses.update(local_face_3d_loss=local_face_3d_loss)

        return losses

    def forward(self, pred_params, targets):
        batch_size = pred_params['betas'].shape[0]

        # get target params and conf.
        target_params = {}
        conf_params = {}
        keyp_confs = defaultdict(lambda: [])

        MY_KEYS = ['betas', 'expression', 'global_orient', 'body_pose',
                   'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'vertices']

        for param_key in MY_KEYS:
            param_pred = pred_params[param_key]
            target_params[param_key] = param_pred.new_zeros(param_pred.shape)
            conf_params[param_key] = param_pred.new_zeros(param_pred.shape[0])

        for idx, t in enumerate(targets):
            conf = t.conf
            keyp_confs['body'].append(conf[self.body_idxs])
            keyp_confs['left_hand'].append(conf[self.left_hand_idxs])
            keyp_confs['right_hand'].append(conf[self.right_hand_idxs])
            keyp_confs['face'].append(conf[self.face_idxs])

            # change key name
            for param_key in MY_KEYS:
                target_key, attr_name = param_key, param_key
                if param_key in ['left_hand_pose', 'right_hand_pose']:
                    target_key = 'hand_pose'
                if param_key in ['global_orient']:
                    target_key, attr_name = 'global_pose', 'global_pose'

                if t.has_field(target_key):
                    tmp = getattr(t.get_field(target_key), attr_name)
                    if tmp is not None:
                        target_params[param_key][idx] = tmp
                        conf_params[param_key][idx] = 1.0

        # Stack all
        for key in keyp_confs:
            keyp_confs[key] = torch.stack(keyp_confs[key])

        if self.expr_use_conf_weight:
            conf_params['expression'] *= keyp_confs['face'].mean(axis=1)
        if self.lhand_use_conf:
            conf_params['left_hand_pose'] *= keyp_confs['left_hand'].mean(axis=1)
        if self.rhand_use_conf:
            conf_params['right_hand_pose'] *= keyp_confs['right_hand'].mean(axis=1)
        if self.jaw_use_conf_weight:
            conf_params['jaw_pose'] *= keyp_confs['face'].mean(axis=1)

        # smplx parameters loss
        losses = self.smplx_loss(pred_params, target_params, conf_params)

        # keypoints loss
        proj_joints, joints3d = pred_params['proj_joints'], pred_params['joints']
        keyp_losses = self.keypoints_loss(proj_joints, joints3d, targets)
        losses.update(keyp_losses)

        return losses
