import os
import sys
import json
import pickle
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.config import CONF
from lib.dataset import ScannetReferenceDataset 
from lib.solver import Solver
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from models.vgnet import VGNet
from utils.box_util import get_3d_box
from data.scannet.model_util_scannet import ScannetDatasetConfig

SCANREFER_TEST = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_test.json")))
DC = ScannetDatasetConfig()
def get_dataloader(args, scanrefer, all_scene_list, split, config):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer,
        scanrefer_all_scene=all_scene_list,
        split=split,
        num_points=args.num_points,
        use_height=args.use_height,
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        augment=False,
        lang_emb_type=args.lang_emb_type
    )
    print("predict for {} samples".format(len(dataset)))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    return dataset, dataloader

def get_model(args, config):
    # load model
    if args.use_multiview:
        if args.fuse_multi_mode == 'early':
            input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(args.use_height)
        elif args.fuse_multi_mode == 'late':
            input_channels = int(args.use_normal) * 3 + int(args.use_color) * 3 + int(args.use_height)
    else:
        input_channels = int(args.use_normal) * 3 + int(args.use_color) * 3 + int(args.use_height)
    model = VGNet(
        input_feature_dim=input_channels,
        args=CONF,
        data_config=DC,
    ).cuda()

    path = os.path.join(CONF.PATH.OUTPUT, args.folder, "model.pth")
    model.load_state_dict(torch.load(path), strict=True)
    model.eval()

    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(args):
    scanrefer = SCANREFER_TEST
    scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
    scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]

    return scanrefer, scene_list

def predict(args):
    print("predict bounding boxes...")
    # constant

    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "test", DC)

    # model
    model = get_model(args, DC)

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    } if not args.no_nms else None

    # predict
    print("predicting...")
    pred_bboxes = []
    for data_dict in tqdm(dataloader):
        for key in data_dict:
            if key != 'scene_id':
                data_dict[key] = data_dict[key].cuda()

        # feed
        with torch.no_grad():
            data_dict = model(data_dict)
        _, data_dict = get_loss(
            data_dict=data_dict, 
            config=DC, 
            args=CONF
        )

        #objectness_preds_batch = (data_dict['last_objectness_scores'] > 0).squeeze(2).long()
        objectness_preds_batch = torch.ones_like(data_dict['last_objectness_scores']).squeeze(2).long()
        if POST_DICT:
            _ = parse_predictions(data_dict, POST_DICT, prefix='last_')
            nms_masks = torch.LongTensor(data_dict['pred_mask']).cuda()

            # construct valid mask
            pred_masks = (nms_masks * objectness_preds_batch == 1).float()
        else:
            # construct valid mask
            pred_masks = (objectness_preds_batch == 1).float()

        pred_ref = torch.argmax(data_dict['last_ref_scores'] * pred_masks, 1) # (B,)
        pred_center = data_dict['last_center'] # (B,K,3)
        pred_heading_class = torch.argmax(data_dict['last_heading_scores'], -1) # B,num_proposal
        pred_heading_residual = torch.gather(data_dict['last_heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2) # B,num_proposal
        pred_size_class = torch.argmax(data_dict['last_size_scores'], -1) # B,num_proposal
        pred_size_residual = torch.gather(data_dict['last_size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_class = pred_size_class
        pred_size_residual = pred_size_residual.squeeze(2) # B,num_proposal,3

        for i in range(pred_ref.shape[0]):
            # compute the iou
            pred_ref_idx = pred_ref[i]
            pred_obb = DC.param2obb(
                pred_center[i, pred_ref_idx, 0:3].detach().cpu().numpy(), 
                pred_heading_class[i, pred_ref_idx].detach().cpu().numpy(), 
                pred_heading_residual[i, pred_ref_idx].detach().cpu().numpy(),
                pred_size_class[i, pred_ref_idx].detach().cpu().numpy(), 
                pred_size_residual[i, pred_ref_idx].detach().cpu().numpy()
            )
            pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])

            # construct the multiple mask
            multiple = data_dict["unique_multiple"][i].item()

            # construct the others mask
            others = 1 if data_dict["object_cat"][i] == 17 else 0

            # store data
            scanrefer_idx = data_dict["scan_idx"][i].item()
            pred_data = {
                "scene_id": scanrefer[scanrefer_idx]["scene_id"],
                "object_id": scanrefer[scanrefer_idx]["object_id"],
                "ann_id": scanrefer[scanrefer_idx]["ann_id"],
                "bbox": pred_bbox.tolist(),
                "unique_multiple": multiple,
                "others": others
            }
            pred_bboxes.append(pred_data)

    # dump
    print("dumping...")
    pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "pred.json")
    with open(pred_path, "w") as f:
        json.dump(pred_bboxes, f, indent=4)

    print("done!")

if __name__ == "__main__":

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = CONF.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    # reproducibility
    torch.manual_seed(CONF.manual_seed)
    torch.cuda.manual_seed_all(CONF.manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(CONF.manual_seed)

    predict(CONF)
