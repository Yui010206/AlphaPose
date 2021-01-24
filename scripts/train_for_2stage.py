"""Script for multi-gpu training."""
import json
import os

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from alphapose.models import builder
from alphapose.opt import cfg, logger, opt
from alphapose.utils.logger import board_writing, debug_writing
from alphapose.utils.metrics import DataLogger, calc_accuracy, calc_integral_accuracy, evaluate_mAP
from alphapose.utils.transforms import get_func_heatmap_to_coord, _integral_tensor,get_box_for_align,integral_op,get_affine_transform,affine_transform,transform_preds
from alphapose.models.criterion import IngetralCoordinate

num_gpu = torch.cuda.device_count()
valid_batch = 1 * num_gpu
if opt.sync:
    norm_layer = nn.SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d



def train(opt, train_loader, m, criterion, optimizer, writer):
    loss_logger = DataLogger()
    acc_logger = DataLogger()
    m.train()
    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    num_joints = cfg.DATA_PRESET.get('NUM_JOINTS',133)
    train_branch = cfg.OTHERS.get('TRAIN_BRANCH',True)

    train_loader = tqdm(train_loader, dynamic_ncols=True)

    for i, (inps, labels, label_masks, _, bboxes) in enumerate(train_loader):
        if isinstance(inps, list):
            inps = [inp.cuda().requires_grad_() for inp in inps]
        else:
            inps = inps.cuda().requires_grad_()

        out, feature = m(inps)

        # train for finer hands
        if train_branch:
            out = m.module.forward_branch(out,feature,bboxes[:,1,:],bboxes[:,2,:])
            labels = labels[:,:-68*2].cuda()
            label_masks = label_masks[:,:-68*2].cuda()

        else:

            labels = labels[:,:133*2].cuda()
            label_masks = label_masks[:,:133*2].cuda()

        loss = criterion(out, labels, label_masks)
        acc = calc_integral_accuracy(out, labels, label_masks, output_3d=False, norm_type=norm_type)

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        loss_logger.update(loss.item(), batch_size)
        acc_logger.update(acc, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opt.trainIters += 1
        # Tensorboard
        if opt.board:
            board_writing(writer, loss_logger.avg, acc_logger.avg, opt.trainIters, 'Train')

        # Debug
        if opt.debug and not i % 10:
            debug_writing(writer, output, labels, inps, opt.trainIters)

        # TQDM
        train_loader.set_description(
            'loss: {loss:.8f} | acc: {acc:.4f}'.format(
                loss=loss_logger.avg,
                acc=acc_logger.avg)
        )

    train_loader.close()

    return loss_logger.avg, acc_logger.avg


def validate(m, opt, heatmap_to_coord, batch_size=20):
    det_dataset = builder.build_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt)
    det_loader = torch.utils.data.DataLoader(
        det_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False)
    kpt_json = []
    eval_joints = det_dataset.EVAL_JOINTS
    test_branch = cfg.OTHERS.get('TEST_BRANCH',True)
    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    for inps, crop_bboxes, bboxes, img_ids, scores, imghts, imgwds in tqdm(det_loader, dynamic_ncols=True):
        if isinstance(inps, list):
            inps = [inp.cuda() for inp in inps]
        else:
            inps = inps.cuda()

        output,_ = m(inps,crop_bboxes[:,1,:],crop_bboxes[:,2,:],crop_bboxes[:,3,:])

        pred = output
        assert pred.dim() == 4
        pred = pred[:, eval_joints, :, :]

        for i in range(output.shape[0]):
            bbox = crop_bboxes[i][0].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred[i][det_dataset.EVAL_JOINTS], bbox, hm_shape=hm_size, norm_type=norm_type)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            #data['bbox'] = bboxes[i, 0].tolist()
            data['bbox'] = bbox
            data['image_id'] = int(img_ids[i])
            data['score'] = float(scores[i] + np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    with open(os.path.join(opt.work_dir, 'test_kpt.json'), 'w') as fid:
                
        json.dump(kpt_json, fid)
    res = evaluate_mAP(os.path.join(opt.work_dir, 'test_kpt.json'), ann_type='keypoints', ann_file='/ssd3/Benchmark/coco/annotations/coco_wholebody_val_133.json')#ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN))
    return res


def validate_gt(m, opt, cfg, heatmap_to_coord, batch_size=20):
    gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False)
    eval_joints = gt_val_dataset.EVAL_JOINTS
    test_branch = cfg.OTHERS.get('TEST_BRANCH',True)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False)
    kpt_json = []
    kpt_json_branch = []
    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    for inps, labels, label_masks, img_ids, bboxes in tqdm(gt_val_loader, dynamic_ncols=True):
        if isinstance(inps, list):
            inps = [inp.cuda() for inp in inps]
        else:
            inps = inps.cuda()
        output,feature = m(inps)

        pred = copy.deepcopy(output)
        assert pred.dim() == 4
        pred = pred[:, eval_joints, :, :]

        for i in range(output.shape[0]):
            bbox = bboxes[i][0].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred[i][gt_val_dataset.EVAL_JOINTS], bbox, hm_shape=hm_size, norm_type=norm_type)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            #data['bbox'] = bboxes[i, 0].tolist()
            data['bbox'] = bbox
            data['image_id'] = int(img_ids[i])
            data['score'] = float(np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

        if test_branch:
            hm_height, hm_width = hm_size
            # regression the joints of wholeboy in stage1
            pred_jts, pred_score = _integral_tensor(
            pred, 133, False, hm_width, hm_height, 1, integral_operation=integral_op, norm_type='sigmoid')
            pred_jts = pred_jts.reshape(pred_jts.shape[0], 133, 2)

            # get the coords with the size of heatmap
            coords_x = (pred_jts[:, :, 0] + 0.5) * hm_width
            coords_y = (pred_jts[:, :, 1] + 0.5) * hm_height

            # get the box of hands for roi align
            lefthand_boxes = get_box_for_align(coords_x[:,-42:-21],coords_y[:,-42:-21])
            righthand_boxes = get_box_for_align(coords_x[:,-21:],coords_y[:,-21:])
            # stage2 testing
            fine_out = m.forward_branch(output, feature, lefthand_boxes, righthand_boxes)
            # output contains the finer and amplified hands kpts, need to apply aff
            fine_pred_jts, fine_pred_score = _integral_tensor(
            fine_out[:,-42:,:,:], 42, False, hm_width, hm_height, 1, integral_operation=integral_op, norm_type='sigmoid')
            fine_pred_jts = fine_pred_jts.reshape(fine_pred_jts.shape[0], 42, 2)
            

            lefthand_jts = fine_pred_jts[:,:21,:]
            righthand_jts = fine_pred_jts[:,21:,:]
            lefthand_jts[:,:,0] = (lefthand_jts[:,:,0]+0.5)*hm_width
            lefthand_jts[:,:,1] = (lefthand_jts[:,:,1]+0.5)*hm_height
            righthand_jts[:,:,0] = (righthand_jts[:,:,0]+0.5)*hm_width
            righthand_jts[:,:,1] = (righthand_jts[:,:,1]+0.5)*hm_height

            center_hm = np.array([hm_width/2.0,hm_height/2.0])
            scale_hm = np.array([hm_size[1],hm_size[0]])

            lefthand_kpts = copy.deepcopy(lefthand_jts.cpu().numpy().astype(np.float32))
            righthand_kpts = copy.deepcopy(righthand_jts.cpu().numpy().astype(np.float32))
            # apply affine trans to lefthand and add offset
            for j in range(lefthand_jts.shape[0]):
                box = lefthand_boxes[j].tolist()
                width = np.array(box[2] - box[0])
                height = np.array(box[3] - box[1])
                output_size = [box[2]-box[0],box[3]-box[1]]
                offset = np.array([box[0],box[1]])
                trans = get_affine_transform(center_hm,scale_hm,0,output_size) 
                for k in range(21):
                    lefthand_kpts[j ,k, 0:2] = affine_transform(lefthand_kpts[j ,k, 0:2], trans)
        
                lefthand_kpts[j,:,0] = (lefthand_kpts[j,:,0]) + offset[0]
                lefthand_kpts[j,:,1] = (lefthand_kpts[j,:,1])+ offset[1]
            #--------------------------------------------------
            # apply affine trans to righthand and add offset
            for j in range(righthand_jts.shape[0]):
                box = righthand_boxes[j].tolist()
                width = np.array(box[2] - box[0])
                height = np.array(box[3] - box[1])
                output_size = [box[2]-box[0],box[3]-box[1]]
                offset = np.array([box[0],box[1]])
                trans = get_affine_transform(center_hm,scale_hm,0,output_size)
                for k in range(21):
                    righthand_kpts[j,k, 0:2] = affine_transform(righthand_kpts[j ,k, 0:2], trans)
                
                righthand_kpts[j,:,0] = (righthand_kpts[j,:,0]) + offset[0]
                righthand_kpts[j,:,1] = (righthand_kpts[j,:,1]) + offset[1]
            #--------------------------------------------------

            bodyface_kpts = copy.deepcopy(pred_jts[:,:-42,:].cpu().numpy().astype(np.float32))
            bodyface_kpts[:,:,0] = (bodyface_kpts[:,:,0]+0.5)*hm_width
            bodyface_kpts[:,:,1] = (bodyface_kpts[:,:,1]+0.5)*hm_height

            fine_kpts = np.concatenate((bodyface_kpts,lefthand_kpts,righthand_kpts), axis=1)
            fine_socre = np.concatenate((pred_score[:,:-42,:].cpu().numpy(),fine_pred_score.cpu().numpy()), axis=1)
            
            for n in range(output.shape[0]):
                bbox = bboxes[n][0].tolist()
                xmin, ymin, xmax, ymax = bbox
                w = xmax - xmin
                h = ymax - ymin
                center = np.array([xmin + w * 0.5, ymin + h * 0.5])
                scale = np.array([w, h])
                for l in range(fine_kpts.shape[1]):
                    fine_kpts[n, l, 0:2] = transform_preds(fine_kpts[n, l, 0:2], center, scale,
                                               [hm_size[1],hm_size[0]])

                keypoints = np.concatenate((fine_kpts[n], fine_socre[n]), axis=1)
                keypoints = keypoints.reshape(-1).tolist()

                data_branch = dict()
                #data['bbox'] = bboxes[i, 0].tolist()
                data_branch['bbox'] = bbox
                data_branch['image_id'] = int(img_ids[n])
                data_branch['score'] = float(np.mean(fine_socre) + np.max(fine_socre))
                data_branch['category_id'] = 1
                data_branch['keypoints'] = keypoints
                kpt_json_branch.append(data_branch)
    

    with open(os.path.join(opt.work_dir, 'test_gt_kpt.json'), 'w') as fid:
        json.dump(kpt_json, fid)

    res = evaluate_mAP(os.path.join(opt.work_dir, 'test_gt_kpt.json'), ann_type='keypoints', ann_file='/ssd3/Benchmark/coco/annotations/coco_wholebody_val_133.json')#ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN))
        
    if test_branch:
        with open(os.path.join(opt.work_dir, 'test_gt_kpt_2branch.json'), 'w') as fid2:
            json.dump(kpt_json_branch, fid2)
        res_branch = evaluate_mAP(os.path.join(opt.work_dir, 'test_gt_kpt_2branch.json'), ann_type='keypoints', ann_file='/ssd3/Benchmark/coco/annotations/coco_wholebody_val_133.json')
        
        return res,res_branch
    else:
        return res, 0

def main():
    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')

    # Model Initialize
    m = preset_model(cfg)
    m = nn.DataParallel(m).cuda()

    criterion = builder.build_loss(cfg.LOSS).cuda()

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(), lr=cfg.TRAIN.LR)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)

    writer = SummaryWriter('.tensorboard/{}-{}'.format(opt.exp_id, cfg.FILE_NAME))

    train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=True, num_workers=opt.nThreads)

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    opt.trainIters = 0

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')

        # Training
        loss, miou = train(opt, train_loader, m, criterion, optimizer, writer)
        logger.epochInfo('Train', opt.epoch, loss, miou)

        lr_scheduler.step()

        if (i + 1) % opt.snapshot == 0:
            # Save checkpoint
            torch.save(m.module.state_dict(), './exp/{}-{}/model_{}.pth'.format(opt.exp_id, cfg.FILE_NAME, opt.epoch))
            # Prediction Test
            with torch.no_grad():
                gt_AP, gt_AP_branch = validate_gt(m.module, opt, cfg, heatmap_to_coord)
                #rcnn_AP = validate(m.module, opt, heatmap_to_coord)
                #logger.info(f'##### Epoch {opt.epoch} | gt mAP: {gt_AP}) | rcnn mAP: {rcnn_AP} #####')
                logger.info(f'##### Epoch {opt.epoch} | gt mAP: {gt_AP}) | gt mAP fine hand: {gt_AP_branch} #####')

        # Time to add DPG
        if i == cfg.TRAIN.DPG_MILESTONE:
            torch.save(m.module.state_dict(), './exp/{}-{}/final.pth'.format(opt.exp_id, cfg.FILE_NAME))
            # Adjust learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.TRAIN.LR
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.DPG_STEP, gamma=0.1)
            # Reset dataset
            train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True, dpg=True)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=True, num_workers=opt.nThreads)

    torch.save(m.module.state_dict(), './exp/{}-{}/final_DPG.pth'.format(opt.exp_id, cfg.FILE_NAME))


def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()

    return model


if __name__ == "__main__":
    main()
