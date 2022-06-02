import os
import sys
import json
import h5py
import random
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.config import CONF
from models.vgnet import VGNet
from torch.nn.parallel import DistributedDataParallel

if CONF.debug:
    SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))[0:CONF.batch_size * 10]
    SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))[0:CONF.batch_size * 10]
else:
    SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
    SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
DC = ScannetDatasetConfig()


def get_dataloader(args, scanrefer, all_scene_list):
    # Init datasets and dataloaders
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
 
    # Create Dataset and Dataloader
    train_dataset = ScannetReferenceDataset(
        scanrefer=scanrefer['train'],
        scanrefer_all_scene=all_scene_list,
        split='train',
        num_points=args.num_points,
        use_height=args.use_height,
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        augment=args.use_augment,
        lang_emb_type=args.lang_emb_type
    )
    if args.distribute:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                worker_init_fn=my_worker_init_fn,
                                                pin_memory=False,
                                                sampler=train_sampler,
                                                drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
 
    val_dataset = ScannetReferenceDataset(
        scanrefer=scanrefer['val'],
        scanrefer_all_scene=all_scene_list,
        split='val',
        num_points=args.num_points,
        use_height=args.use_height,
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        augment=False,
        lang_emb_type=args.lang_emb_type
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
 
    return train_dataset, train_loader, val_dataset, val_loader

def get_model(args):
    # initiate model
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
    )
      
    # to CUDA
    model = model.cuda()
    
    if args.distribute:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)

    return model

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader):
    model = get_model(args)
    if args.det_decoder_lr:
        param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "decoder" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "decoder" in n and 'text' in n and p.requires_grad],
            "lr": args.decoder_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "decoder" in n and 'text' not in n and p.requires_grad],
            "lr": args.det_decoder_lr,
        }
    ]
    else:
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "decoder" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "decoder" in n and p.requires_grad],
                "lr": args.decoder_lr,
            },
        ]

    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.wd)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    LR_DECAY_STEP = args.lr_decay_step
    LR_DECAY_RATE = args.lr_decay_rate
    BN_DECAY_STEP = args.bn_decay_step
    BN_DECAY_RATE = args.bn_decay_rate

    solver = Solver(
        model=model, 
        data_config=DC, 
        dataloader=dataloader, 
        optimizer=optimizer, 
        stamp=stamp, 
        val_freq=args.val_freq,
        args=args,
        detection=not args.no_detection,
        reference=not args.no_reference, 
        use_lang_classifier=not args.no_lang_cls,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE,
        distributed_rank=args.local_rank if args.distribute else None
    )
    num_params = get_num_params(model)

    return solver, num_params, root

def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes):
    # get initial scene list
    train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
    if num_scenes == -1: 
        num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= num_scenes
    
    # slice train_scene_list
    train_scene_list = train_scene_list[:num_scenes]

    # filter data in chosen scenes
    new_scanrefer_train = []
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            new_scanrefer_train.append(data)

    new_scanrefer_val = scanrefer_val

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list

def train(args):
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes)
    #scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(SCANREFER_VAL, SCANREFER_VAL, args.num_scenes)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }

    # dataloader
    train_dataset, train_dataloader, val_dataset, val_dataloader = get_dataloader(args, scanrefer, all_scene_list)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)

def init():
    # # copy important files to backup
    # backup_dir = os.path.join(CONF.exp_path, 'backup_files')
    # os.makedirs(backup_dir, exist_ok=True)
    # os.system('cp {}/scripts/train.py {}'.format(CONF.PATH.BASE, backup_dir))
    # os.system('cp {} {}'.format(CONF.config, backup_dir))
    # os.system('cp {} {}'.format(CONF.PATH.BASE+'/models/util.py', backup_dir))
    # os.system('cp {}/models/{}.py {}'.format(CONF.PATH.BASE, CONF.model, backup_dir))
    # os.system('cp {}/models/{}.py {}'.format(CONF.PATH.BASE, CONF.language_module, backup_dir))

    # random seed
    random.seed(CONF.manual_seed)
    np.random.seed(CONF.manual_seed)
    torch.manual_seed(CONF.manual_seed)
    torch.cuda.manual_seed_all(CONF.manual_seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    if CONF.distribute:
        torch.cuda.set_device(CONF.local_rank)
        torch.backends.cudnn.benchmark = False      # to avoid random
        torch.distributed.init_process_group(backend='nccl', init_method='env://')   

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = CONF.gpu
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    init()
    train(CONF)
