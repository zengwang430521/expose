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

import pickle
import json
import time

import torch
import torch.utils.data as dutils
import numpy as np

from loguru import logger

from ..utils.bbox import (
    keyps_to_bbox, bbox_to_center_scale, scale_to_bbox_size)
from ..targets import (Keypoints2D, BodyPose, GlobalPose, Betas, Vertices,
                       HandPose, JawPose, Expression, BoundingBox)
from ..targets.keypoints import (KEYPOINT_NAMES,
                                 get_part_idxs,
                                 dset_to_body_model)
from ...utils import nand

from ...utils.img_utils import read_img
FOLDER_MAP_FNAME = 'folder_map.pkl'


class H36M(dutils.Dataset):
    def __init__(self, img_folder, npz_file,
                 dtype=torch.float32,
                 use_face_contour=False,
                 binarization=True,
                 body_thresh=0.1,
                 hand_thresh=0.2,
                 face_thresh=0.4,
                 min_hand_keypoints=8,
                 min_head_keypoints=8,
                 transforms=None,
                 split='train',
                 return_shape=False,
                 return_full_pose=False,
                 return_params=True,
                 return_gender=False,
                 vertex_folder='vertices',
                 return_vertices=True,
                 vertex_flip_correspondences='',
                 **kwargs):
        super().__init__()

        self.img_folder = osp.expandvars(img_folder)
        self.transforms = transforms
        self.use_face_contour = use_face_contour
        self.body_thresh = body_thresh
        self.hand_thresh = hand_thresh
        self.face_thresh = face_thresh
        self.binarization = binarization
        self.dtype = dtype
        self.split = split

        self.min_hand_keypoints = min_hand_keypoints
        self.min_head_keypoints = min_head_keypoints

        self.return_vertices = return_vertices
        self.return_gender = return_gender
        self.return_params = return_params
        self.return_shape = return_shape
        self.return_full_pose = return_full_pose

        self.vertex_folder = osp.join(
            osp.split(self.img_folder)[0], vertex_folder)

        vertex_flip_correspondences = osp.expandvars(
            vertex_flip_correspondences)
        err_msg = (
            'Vertex flip correspondences path does not exist:' +
            f' {vertex_flip_correspondences}'
        )
        assert osp.exists(vertex_flip_correspondences), err_msg
        flip_data = np.load(vertex_flip_correspondences)
        self.bc = flip_data['bc']
        self.closest_faces = flip_data['closest_faces']

        npz_fn = osp.expandvars(npz_file)
        data = np.load(npz_fn)
        if 'genders' not in data and self.return_gender:
            data['genders'] = [''] * len(data['pose'])

        self.centers = data['center'].astype(np.float32)
        self.scales = data['scale'].astype(np.float32)
        self.poses = data['poses_smplx'].astype(np.float32)
        self.keypoints2d = data['part'].astype(np.float32)
        self.imgname = data['imgname'].astype(np.string_)

        if self.return_gender:
            self.gender = data['genders'].astype(np.string_)

        if self.return_shape:
            self.betas = data['betas_smplx'].astype(np.float32)

        self.num_items = data['scale'].shape[0]
        #  logger.info(self.num_items)

        source_idxs, target_idxs = dset_to_body_model(
            model_type='smplx', use_hands=True, use_face=True,
            dset='spin', use_face_contour=self.use_face_contour)
        self.source_idxs = np.asarray(source_idxs, dtype=np.int64)
        self.target_idxs = np.asarray(target_idxs, dtype=np.int64)

        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        face_idxs = idxs_dict['face']
        if not self.use_face_contour:
            face_idxs = face_idxs[:-17]
        self.body_idxs = np.asarray(body_idxs)
        self.hand_idxs = np.asarray(hand_idxs)
        self.face_idxs = np.asarray(face_idxs)

    def get_elements_per_index(self):
        return 1

    def name(self):
        return 'H36M/{}'.format(self.split)

    def only_2d(self):
        return False

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):

        imgname = self.imgname[index].decode()
        img_fn = osp.join(self.img_folder, imgname)
        img = read_img(img_fn)
        # print('debug!!!!!'); img = np.zeros([1000, 1000, 3], dtype=np.float32)
        keypoints2d = self.keypoints2d[index]

        output_keypoints2d = np.zeros([127 + 17 * self.use_face_contour,
                                       3], dtype=np.float32)

        output_keypoints2d[self.target_idxs] = keypoints2d[self.source_idxs]

        # Remove joints with negative confidence
        output_keypoints2d[output_keypoints2d[:, -1] < 0, -1] = 0
        if self.body_thresh > 0:
            # Only keep the points with confidence above a threshold
            body_conf = output_keypoints2d[self.body_idxs, -1]
            hand_conf = output_keypoints2d[self.hand_idxs, -1]
            face_conf = output_keypoints2d[self.face_idxs, -1]

            body_conf[body_conf < self.body_thresh] = 0.0
            hand_conf[hand_conf < self.hand_thresh] = 0.0
            face_conf[face_conf < self.face_thresh] = 0.0
            if self.binarization:
                body_conf = (
                    body_conf >= self.body_thresh).astype(
                        output_keypoints2d.dtype)
                hand_conf = (
                    hand_conf >= self.hand_thresh).astype(
                        output_keypoints2d.dtype)
                face_conf = (
                    face_conf >= self.face_thresh).astype(
                        output_keypoints2d.dtype)

            output_keypoints2d[self.body_idxs, -1] = body_conf
            output_keypoints2d[self.hand_idxs, -1] = hand_conf
            output_keypoints2d[self.face_idxs, -1] = face_conf

        target = Keypoints2D(
            output_keypoints2d, img.shape, flip_axis=0, dtype=self.dtype)

        keypoints = output_keypoints2d[:, :-1]
        conf = output_keypoints2d[:, -1]
        _, _, bbox_size = bbox_to_center_scale(
            keyps_to_bbox(keypoints, conf, img_size=img.shape),
            dset_scale_factor=1.2
        )
        center = self.centers[index]
        scale = self.scales[index]
        target.add_field('center', center)
        target.add_field('scale', scale)
        target.add_field('bbox_size', bbox_size)
        target.add_field('keypoints_hd', output_keypoints2d)

        if self.return_params:
            # pose = self.poses[index].reshape(-1, 3)
            pose = self.poses[index].copy()

            global_pose_target = GlobalPose(pose[0].reshape(-1))
            target.add_field('global_pose', global_pose_target)
            if self.return_full_pose:
                body_pose = pose[1:]
            else:
                body_pose = pose[1:22]
            body_pose_target = BodyPose(body_pose.reshape(-1))
            target.add_field('body_pose', body_pose_target)

        if self.return_shape:
            betas = self.betas[index]
            target.add_field('betas', Betas(betas))
        if self.return_vertices:
            vertex_fname = osp.join(
                self.vertex_folder, f'{index:06d}.npy')

            target.add_field('vname', vertex_fname)
            # fname = osp.join(self.vertex_folder, f'{index:06d}.npy')

            H, W, _ = img.shape
            fscale = H / bbox_size
            intrinsics = np.array([[5000 * fscale, 0, 0],
                                   [0, 5000 * fscale, 0],
                                   [0, 0, 1]], dtype=np.float32)

            target.add_field('intrinsics', intrinsics)
            vertices = np.load(vertex_fname, allow_pickle=True)
            vertex_field = Vertices(
                vertices, bc=self.bc, closest_faces=self.closest_faces)
            target.add_field('vertices', vertex_field)

        if self.transforms is not None:
            force_flip = False
            full_img, cropped_image, cropped_target = self.transforms(
                img, target, dset_scale_factor=1.2, force_flip=force_flip)
        target.add_field('name', self.name())

        dict_key = ['h36m', self.imgname[index].decode('utf-8'), index]
        if hasattr(self, 'gender') and self.return_gender:
            gender = self.gender[index].decode('utf-8')
            if gender == 'F' or gender == 'M':
                target.add_field('gender', gender)
            dict_key.append(gender)

        # Add the key used to access the fit dict
        dict_key = tuple(dict_key)
        target.add_field('dict_key', dict_key)

        return full_img, cropped_image, cropped_target, index
