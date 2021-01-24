# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com), Haoyi Zhu
# -----------------------------------------------------

import platform
import random
import copy
import cv2
import numpy as np
import torch

from ..bbox import (_box_to_center_scale, _center_scale_to_box,
                    _clip_aspect_ratio)
from ..transforms import (addDPG, affine_transform, flip_joints_3d,
                          get_affine_transform, im_to_torch)

# Only windows visual studio 2013 ~2017 support compile c/cuda extensions
# If you force to compile extension on Windows and ensure appropriate visual studio
# is intalled, you can try to use these ext_modules.
if platform.system() != 'Windows':
    from ..roi_align import RoIAlign


class SimpleTransform(object):
    """Generation of cropped input person and pose heatmaps from SimplePose.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, h, w)`.
    label: dict
        A dictionary with 4 keys:
            `bbox`: [xmin, ymin, xmax, ymax]
            `joints_3d`: numpy.ndarray with shape: (n_joints, 2),
                    including position and visible flag
            `width`: image width
            `height`: image height
    dataset:
        The dataset to be transformed, must include `joint_pairs` property for flipping.
    scale_factor: int
        Scale augmentation.
    input_size: tuple
        Input image size, as (height, width).
    output_size: tuple
        Heatmap size, as (height, width).
    rot: int
        Ratation augmentation.
    train: bool
        True for training trasformation.
    """

    def __init__(self, dataset, scale_factor, add_dpg,
                 input_size, output_size, rot, sigma,
                 train, gpu_device=None, loss_type='MSELoss'):
        self._joint_pairs = dataset.joint_pairs
        self._scale_factor = scale_factor
        self._rot = rot
        self._add_dpg = add_dpg
        self._gpu_device = gpu_device

        self._input_size = input_size
        self._heatmap_size = output_size

        self._sigma = sigma
        self._train = train
        self._loss_type = loss_type
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h
        self._feat_stride = np.array(input_size) / np.array(output_size)

        self.pixel_std = 1

        if train:
            self.num_joints_half_body = dataset.num_joints_half_body
            self.prob_half_body = dataset.prob_half_body

            self.upper_body_ids = dataset.upper_body_ids
            self.lower_body_ids = dataset.lower_body_ids
        if platform.system() != 'Windows':
            self.roi_align = RoIAlign(self._input_size, sample_num=-1)
            if gpu_device is not None:
                self.roi_align = self.roi_align.to(gpu_device)

    def test_transform(self, src, bbox):
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio)
        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size

        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        return img, bbox

    def align_transform(self, image, boxes):
        """
        Performs Region of Interest (RoI) Align operator described in Mask R-CNN

        Arguments:
            input (ndarray [H, W, 3]): input images
            boxes (Tensor[K, 4]): the box coordinates in (x1, y1, x2, y2)
                format where the regions will be taken from.

        Returns:
            cropped_img (Tensor[K, C, output_size[0], output_size[1]])
            boxes (Tensor[K, 4]): new box coordinates
        """
        tensor_img = im_to_torch(image)
        tensor_img[0].add_(-0.406)
        tensor_img[1].add_(-0.457)
        tensor_img[2].add_(-0.480)

        new_boxes = _clip_aspect_ratio(boxes, self._aspect_ratio)
        cropped_img = self.roi_align(tensor_img.unsqueeze(0).to(self._gpu_device), new_boxes.to(self._gpu_device))
        return cropped_img, new_boxes[:, 1:]

    def _target_generator(self, joints_3d, num_joints):
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target = np.zeros((num_joints, self._heatmap_size[0], self._heatmap_size[1]),
                          dtype=np.float32)
        tmp_size = self._sigma * 3

        for i in range(num_joints):
            mu_x = int(joints_3d[i, 0, 0] / self._feat_stride[0] + 0.5)
            mu_y = int(joints_3d[i, 1, 0] / self._feat_stride[1] + 0.5)
            # check if any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if (ul[0] >= self._heatmap_size[1] or ul[1] >= self._heatmap_size[0] or br[0] < 0 or br[1] < 0):
                # return image as is
                target_weight[i] = 0
                continue

            # generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # the gaussian is not normalized, we want the center value to be equal to 1
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self._sigma ** 2)))

            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self._heatmap_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self._heatmap_size[0]) - ul[1]
            # image range
            img_x = max(0, ul[0]), min(br[0], self._heatmap_size[1])
            img_y = max(0, ul[1]), min(br[1], self._heatmap_size[0])

            v = target_weight[i]
            if v > 0.5:
                target[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, np.expand_dims(target_weight, -1)

    def _integral_target_generator(self, joints_3d, num_joints, patch_height, patch_width, source=None):
        target_weight = np.ones((num_joints, 2), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]
        if num_joints == 136:
            target_weight[:26, :] = target_weight[:26, :] * 2
        elif num_joints == 133:
            target_weight[:23, :] = target_weight[:23, :] * 2
            #target_weight[23:-42, :] = target_weight[23:-42, :] * 0.5
        
        if source == 'frei' or source == 'partX' or source == 'OneHand' or source == 'hand_labels_synth' \
        or source == 'hand143_panopticdb' or source == 'RHD_published_v2' or source == 'interhand':
            if target_weight[-21:,:].sum() > 0 and target_weight[-42:-21].sum() == 0:
                target_weight[-42:-21] += 1
            elif target_weight[-21:,:].sum() == 0 and target_weight[-42:-21].sum() > 0:
                target_weight[-21:,:] += 1

        target = np.zeros((num_joints, 2), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def __call__(self, src, label, source=None):
        bbox = list(label['bbox'])
        face_bbox = list(label['face_bbox'])
        lefthand_bbox = list(label['lefthand_bbox'])
        righthand_bbox = list(label['righthand_bbox'])
        gt_joints = label['joints_3d']

        imgwidth, imght = label['width'], label['height']
        assert imgwidth == src.shape[1] and imght == src.shape[0]
        self.num_joints = gt_joints.shape[0]

        joints_vis = np.zeros((self.num_joints, 1), dtype=np.float32)
        joints_vis[:, 0] = gt_joints[:, 0, 1]

        input_size = self._input_size

        if self._add_dpg and self._train:
            bbox = addDPG(bbox, imgwidth, imght)
            #face_bbox = addDPG(face_bbox, imgwidth, imght)
            #lefthand_bbox = addDPG(lefthand_bbox, imgwidth, imght)
            #righthand_bbox = addDPG(righthand_bbox, imgwidth, imght)

        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio)

        # half body transform
        if self._train and (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
            c_half_body, s_half_body = self.half_body_transform(
                gt_joints[:, :, 0], joints_vis
            )

            if c_half_body is not None and s_half_body is not None:
                center, scale = c_half_body, s_half_body

        # rescale
        if self._train:
            sf = self._scale_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            #scale_f = scale_f * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            #scale_rh = scale_rh * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            #scale_lh = scale_lh * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        else:
            scale = scale * 1.0
            #scale_f = scale_f * 1.0
            #scale_rh = scale_rh * 1.0
            #scale_lh = scale_lh * 1.0

        # rotation
        if self._train:
            if source == 'frei' or source == 'partX' or source == 'OneHand' or source == 'interhand':
                rf = 180
            else:
                rf = self._rot
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
        else:
            r = 0

        joints = gt_joints
        if random.random() > 0.5 and self._train:
            # src, fliped = random_flip_image(src, px=0.5, py=0)
            # if fliped[0]:
            assert src.shape[2] == 3
            src = src[:, ::-1, :]

            joints = flip_joints_3d(joints, imgwidth, self._joint_pairs)
            center[0] = imgwidth - center[0] - 1

        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)

        # deal with joints visibility
        for i in range(self.num_joints):
            if joints[i, 0, 1] > 0.0:
                joints[i, 0:2, 0] = affine_transform(joints[i, 0:2, 0], trans)

        # -----------------------------------------------------
        # get face/ hand box through wholebody joints
        # deepcopy
        face_joints = copy.deepcopy(joints[23:-42,:,:])
        lefthand_joints = copy.deepcopy(joints[-42:-21,:,:])
        righthand_joints = copy.deepcopy(joints[-21:,:,:])

        xmin_f, ymin_f, xmax_f, ymax_f = self.get_xyxy(face_joints)
        center_f, scale_f = _box_to_center_scale(
            xmin_f, ymin_f, xmax_f - xmin_f, ymax_f - ymin_f, self._aspect_ratio)

        xmin_lh, ymin_lh, xmax_lh, ymax_lh = self.get_xyxy(lefthand_joints)
        center_lh, scale_lh = _box_to_center_scale(
            xmin_lh, ymin_lh, xmax_lh - xmin_lh, ymax_lh - ymin_lh, self._aspect_ratio)

        xmin_rh, ymin_rh, xmax_rh, ymax_rh = self.get_xyxy(righthand_joints)
        center_rh, scale_rh = _box_to_center_scale(
            xmin_rh, ymin_rh, xmax_rh - xmin_rh, ymax_rh - ymin_rh, self._aspect_ratio)

        # -----------------------------------------------------
        # Affine transformation for face/hand box
        trans_f = get_affine_transform(center_f, scale_f, 0, [inp_w, inp_h])
        # img = cv2.warpAffine(src, trans_f, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        for i in range(68):
            if face_joints[i, 0, 1] > 0.0:
                face_joints[i, 0:2, 0] = affine_transform(face_joints[i, 0:2, 0], trans_f)

        trans_lh = get_affine_transform(center_lh, scale_lh, 0, [inp_w, inp_h])
        # img = cv2.warpAffine(src, trans_lf, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        for i in range(21):
            if lefthand_joints[i, 0, 1] > 0.0:
                lefthand_joints[i, 0:2, 0] = affine_transform(lefthand_joints[i, 0:2, 0], trans_lh)

        trans_rh = get_affine_transform(center_rh, scale_rh, 0, [inp_w, inp_h])
        # img = cv2.warpAffine(src, trans_lf, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        for i in range(21):
            if righthand_joints[i, 0, 1] > 0.0:
                righthand_joints[i, 0:2, 0] = affine_transform(righthand_joints[i, 0:2, 0], trans_rh)

        # -----------------------------------------------------

        # generate training targets
        #joints = joints + face_joints + lefthand_joints + righthand_joints
        if self._loss_type == 'MSELoss':
            target, target_weight = self._target_generator(joints, self.num_joints)
        elif 'JointRegression' in self._loss_type:
            target, target_weight = self._integral_target_generator(joints, self.num_joints, inp_h, inp_w, source)
            target_f, target_weight_f = self._integral_target_generator(face_joints, 68, inp_h, inp_w)
            target_lh, target_weight_lh = self._integral_target_generator(lefthand_joints, 21, inp_h, inp_w)
            target_rh, target_weight_rh = self._integral_target_generator(righthand_joints, 21, inp_h, inp_w)

        bbox = _center_scale_to_box(center, scale)
        face_bbox = _center_scale_to_box(center_f, scale_f)
        lefthand_bbox = _center_scale_to_box(center_lh, scale_lh)
        righthand_bbox = _center_scale_to_box(center_rh, scale_rh)

        all_boxes = [bbox,lefthand_bbox,righthand_bbox,face_bbox]
        target_cat = np.concatenate((target,target_lh,target_rh,target_f),axis=0)
        target_weight_cat = np.concatenate((target_weight,target_weight_lh,target_weight_rh,target_weight_f),axis=0)

        img = im_to_torch(img)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        return img, torch.from_numpy(target_cat), torch.from_numpy(target_weight_cat), torch.Tensor(all_boxes)

    def get_xyxy(self, joints):
        joints = np.array(joints)
        x = joints[:,0,0]
        y = joints[:,1,0]
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        y_min = float(np.min(y))
        y_max = float(np.max(y))

        return x_min, y_min, x_max, y_max


    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self._aspect_ratio * h:
            h = w * 1.0 / self._aspect_ratio
        elif w < self._aspect_ratio * h:
            w = h * self._aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale
