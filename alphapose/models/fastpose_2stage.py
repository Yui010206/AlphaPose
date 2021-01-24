# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torch.nn as nn
import torch
import torchvision.ops.roi_align as RoI_Align
from .builder import SPPE
from .layers.DUC import DUC
from .layers.SE_Resnet import SEResnet
from alphapose.utils.transforms import get_func_heatmap_to_coord, _integral_tensor
from alphapose.models.criterion import IngetralCoordinate

def get_box(x,y,scale, factor=1.5):
    x_min = torch.min(x,dim=1)[0]
    x_max = torch.max(x,dim=1)[0]
    y_min = torch.min(y,dim=1)[0]
    y_max = torch.max(y,dim=1)[0]

    center_x = (x_min + x_max)/2
    center_y = (y_min + y_max)/2
    width = (x_max - x_min)*factor
    height = (y_max - y_min)*factor

    crop_x1 = (center_x - width/2)/scale
    crop_y1 = (center_y - height/2)/scale
    crop_x2 = (center_x + width/2)/scale
    crop_y2 = (center_y + height/2)/scale
    batch = x_max.shape[0]
    boxes = []
    for i in range(batch):
        boxes.append([i,crop_x1[i],crop_y1[i],crop_x2[i],crop_y2[i]])
    return torch.tensor(boxes, dtype=x.dtype, device=x.device)

def reshape_boxes(boxes,device):
    bacth = boxes.shape[0]
    new_boxes = []
    for i in range(bacth):
        new_boxes.append([i]+(boxes[i]/32).cpu().tolist())

    return torch.tensor(new_boxes, device=device)

@SPPE.register_module
class FastPose_2Stage(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(FastPose_2Stage, self).__init__()
        self._preset_cfg = cfg['PRESET']
        self.crop_resize = cfg['CROP_RESIZE']
        if 'CONV_DIM' in cfg.keys():
            self.conv_dim = cfg['CONV_DIM']
        else:
            self.conv_dim = 128
        if 'DCN' in cfg.keys():
            stage_with_dcn = cfg['STAGE_WITH_DCN']
            dcn = cfg['DCN']
            self.preact = SEResnet(
                f"resnet{cfg['NUM_LAYERS']}", dcn=dcn, stage_with_dcn=stage_with_dcn)
        else:
            self.preact = SEResnet(f"resnet{cfg['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm   # noqa: F401,F403
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2, norm_layer=norm_layer)
        if self.conv_dim == 256:
            self.duc2 = DUC(256, 1024, upscale_factor=2, norm_layer=norm_layer)
        else:
            self.duc2 = DUC(256, 512, upscale_factor=2, norm_layer=norm_layer)

        self.conv_out = nn.Conv2d(
            self.conv_dim, self._preset_cfg['NUM_JOINTS'], kernel_size=3, stride=1, padding=1)

        # branch networks
        self.suffle1_hand = nn.PixelShuffle(2)
        #self.suffle1_face = nn.PixelShuffle(2)

        self.duc1_hand = DUC(512, 1024, upscale_factor=2, norm_layer=norm_layer)
        #self.duc1_face = DUC(512, 1024, upscale_factor=2, norm_layer=norm_layer)

        if self.conv_dim == 256:
            self.duc2_hand = DUC(256, 1024, upscale_factor=2, norm_layer=norm_layer)
            #self.duc2_face = DUC(256, 1024, upscale_factor=2, norm_layer=norm_layer)
        else:
            self.duc2_hand = DUC(256, 512, upscale_factor=2, norm_layer=norm_layer)
            #self.duc2_face = DUC(256, 512, upscale_factor=2, norm_layer=norm_layer)

        self.conv_out_hand = nn.Conv2d(
            self.conv_dim, int(cfg['NUM_HAND_JOINTS']/2), kernel_size=3, stride=1, padding=1)
        #self.conv_out_face = nn.Conv2d(
            #self.conv_dim, cfg['NUM_FACE_JOINTS'], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #print("input:", x.shape)
        feature = self.preact(x)

        out = self.suffle1(feature)
        out = self.duc1(out)
        out = self.duc2(out)
        out = self.conv_out(out)

        return out,feature

    def forward_branch(self,out, feature, lefthand_boxes, righthand_boxes):

        lefthand_feature = RoI_Align(feature, reshape_boxes(lefthand_boxes,device=out.device), self.crop_resize)
        righthand_feature = RoI_Align(feature, reshape_boxes(righthand_boxes,device=out.device), self.crop_resize)

        out_lhand = self.suffle1_hand(lefthand_feature)
        out_lhand = self.duc1_hand(out_lhand)
        out_lhand = self.duc2_hand(out_lhand)
        out_lhand = self.conv_out_hand(out_lhand)

        out_rhand = self.suffle1_hand(righthand_feature)
        out_rhand = self.duc1_hand(out_rhand)
        out_rhand = self.duc2_hand(out_rhand)
        out_rhand = self.conv_out_hand(out_rhand)

        out_cat = torch.cat((out,out_lhand,out_rhand),dim=1)

        return out_cat

        # forward 133 --- forward hand feature/dagaid handbox duibian  ---feature resize + offset() 用 human box yuce的kpts
        # 手的点 去掉 0 0 边界
        # 
        # forword hand

            # hm_width = out.shape[-1]
            # hm_height = out.shape[-2]

            # pred_jts, pred_scores = _integral_tensor(
            # out, self._preset_cfg['NUM_JOINTS'], False, hm_width, hm_height, 1, 
            # integral_operation=IngetralCoordinate.apply, norm_type='softmax')

            # x_kpt = pred_jts[:,::2]
            # y_kpt = pred_jts[:,1::2]
            # #face_x_kpts = x_kpt[:,23:-42]
            # #face_y_kpts = y_kpt[:,23:-42]

            # lefthands_x_kpts = x_kpt[:,-42:-21]
            # lefthands_y_kpts = y_kpt[:,-42:-21]

            # righthands_x_kpts = x_kpt[:,-21:]
            # righthands_y_kpts = y_kpt[:,-21:]

            # #face_boxes = get_box(face_x_kpts,face_y_kpts,32)
            # lefthand_boxes = get_box(lefthands_x_kpts,lefthands_y_kpts,32)
            # righthand_boxes = get_box(righthands_x_kpts,righthands_y_kpts,32)

            #face_feature = RoI_Align(feature, reshape_boxes(face_boxes), self.crop_resize)

            # lefthand_feature = RoI_Align(feature, reshape_boxes(lefthand_boxes), self.crop_resize)
            # righthand_feature = RoI_Align(feature, reshape_boxes(righthand_boxes), self.crop_resize)
            
            # out_face = self.suffle1_face(face_feature)
            # out_face = self.duc1_face(out_face)
            # out_face = self.duc2_face(out_face)
            # out_face = self.conv_out_face(out_face)

            # out_lhand = self.suffle1_hand(lefthand_feature)
            # out_lhand = self.duc1_hand(out_lhand)
            # out_lhand = self.duc2_hand(out_lhand)
            # out_lhand = self.conv_out_hand(out_lhand)

            # out_rhand = self.suffle1_hand(righthand_feature)
            # out_rhand = self.duc1_hand(out_rhand)
            # out_rhand = self.duc2_hand(out_rhand)
            # out_rhand = self.conv_out_hand(out_rhand)

            #out_ = torch.cat((out[:,:-42,:,:], out_lhand,out_rhand), dim=1)
            # cat(out,)
        
            # out_cat = torch.cat((out, out_lhand, out_rhand), dim=1)

        # else:
        #     pred_jts, pred_scores = _integral_tensor(
        #     out, self._preset_cfg['NUM_JOINTS'], False, hm_width, hm_height, 1, 
        #     integral_operation=IngetralCoordinate.apply, norm_type='softmax')

        #     x_kpt = pred_jts[:,::2]
        #     y_kpt = pred_jts[:,1::2]

        #     lefthands_x_kpts = x_kpt[:,-42:-21]
        #     lefthands_y_kpts = y_kpt[:,-42:-21]

        #     righthands_x_kpts = x_kpt[:,-21:]
        #     righthands_y_kpts = y_kpt[:,-21:]
        #     test_lefthand_boxes = get_box(lefthands_x_kpts,lefthands_y_kpts,32)
        #     test_righthand_boxes = get_box(righthands_x_kpts,righthands_y_kpts,32)


        #     lefthand_feature = RoI_Align(feature, test_lefthand_boxes, self.crop_resize)
        #     righthand_feature = RoI_Align(feature, test_righthand_boxes, self.crop_resize)


        # out_cat = torch.cat((out[:,:-42,:,:],out_lhand,out_rhand),dim=1)

        # return out_cat

        #out_cat = torch.cat((out_body,out_face,out_hand),dim=1)
        #else:
            #return out

    def _initialize(self):
        for m in self.conv_out.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        for m in self.conv_out_hand.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

