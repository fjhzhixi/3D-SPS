'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import torch
import numpy as np
from torch.utils.data import dataloader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn as nn
from data.scannet.model_util_scannet import ScannetDatasetConfig
sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths, parse_ref_predictions, \
    parse_ref_groundtruths
from lib.config import CONF
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from utils.eta import decode_eta
from lib.pointnet2.pytorch_utils import BNMomentumScheduler


ITER_REPORT_TEMPLATE = """
-------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] train_loss: {train_loss}
[loss] train_ref_loss: {train_ref_loss}
[loss] train_ref_mask_loss: {train_ref_mask_loss}
[loss] train_lang_cls_loss: {train_lang_cls_loss}
[loss] train_objectness_loss: {train_objectness_loss}
[loss] train_kps_loss: {train_kps_loss}
[loss] train_box_loss: {train_box_loss}
[loss] train_sem_cls_loss: {train_sem_cls_loss}
[loss] train_lang_cls_acc: {train_lang_cls_acc}
[sco.] train_ref_acc: {train_ref_acc}
[sco.] train_obj_acc: {train_obj_acc}
[sco.] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[sco.] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_eval_time: {mean_eval_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
---------------------------------summary---------------------------------
[train] train_loss: {train_loss}
[train] train_ref_loss: {train_ref_loss}
[train] train_ref_mask_loss: {train_ref_mask_loss}
[train] train_lang_cls_loss: {train_lang_cls_loss}
[train] train_objectness_loss: {train_objectness_loss}
[train] train_kps_loss: {train_kps_loss}
[train] train_box_loss: {train_box_loss}
[train] train_sem_cls_loss: {train_sem_cls_loss}
[train] train_lang_cls_acc: {train_lang_cls_acc}
[train] train_ref_acc: {train_ref_acc}
[train] train_obj_acc: {train_obj_acc}
[train] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[train] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}
[val]   val_loss: {val_loss}
[val]   val_ref_loss: {val_ref_loss}
[val]   val_ref_mask_loss: {val_ref_mask_loss}
[val]   val_lang_cls_loss: {val_lang_cls_loss}
[val]   val_objectness_loss: {val_objectness_loss}
[val]   val_kps_loss: {val_kps_loss}
[val]   val_box_loss: {val_box_loss}
[val]   val_sem_cls_loss: {val_sem_cls_loss}
[val]   val_lang_cls_acc: {val_lang_cls_acc}
[val]   val_ref_acc: {val_ref_acc}
[val]   val_obj_acc: {val_obj_acc}
[val]   val_pos_ratio: {val_pos_ratio}, val_neg_ratio: {val_neg_ratio}
[val]   val_iou_rate_0.25: {val_iou_rate_25}, val_iou_rate_0.5: {val_iou_rate_5}
"""

BEST_REPORT_TEMPLATE = """
--------------------------------------best--------------------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[loss] ref_loss: {ref_loss}
[loss] ref_mask_loss: {ref_mask_loss}
[loss] lang_cls_loss: {lang_cls_loss}
[loss] objectness_loss: {objectness_loss}
[loss] kps_loss: {kps_loss}
[loss] box_loss: {box_loss}
[loss] sem_cls_loss: {sem_cls_loss}
[loss] lang_cls_acc: {lang_cls_acc}
[sco.] ref_acc: {ref_acc}
[sco.] obj_acc: {obj_acc}
[sco.] pos_ratio: {pos_ratio}, neg_ratio: {neg_ratio}
[sco.] iou_rate_0.25: {iou_rate_25}, iou_rate_0.5: {iou_rate_5}
"""


class Solver():
    def __init__(self, model, data_config, dataloader, optimizer, stamp, val_freq=1, args=None,
    detection=True, reference=True, use_lang_classifier=True,
    lr_decay_step=None, lr_decay_rate=None, bn_decay_step=None, bn_decay_rate=None, distributed_rank=None):

        self.epoch = 0                    # set in __call__
        self.verbose = 0                  # set in __call__
        
        self.model = model
        self.data_config = data_config
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_freq = val_freq
        self.args = args

        self.detection = detection
        self.reference = reference
        self.use_lang_classifier = use_lang_classifier

        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.bn_decay_step = bn_decay_step
        self.bn_decay_rate = bn_decay_rate

        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "ref_loss": float("inf"),
            "ref_mask_loss": float("inf"),
            "lang_cls_loss": float("inf"),
            "objectness_loss": float("inf"),
            "kps_loss": float("inf"),
            "box_loss": float("inf"),
            "sem_cls_loss": float("inf"),
            "lang_cls_acc": -float("inf"),
            "ref_acc": -float("inf"),
            "obj_acc": -float("inf"),
            "pos_ratio": -float("inf"),
            "neg_ratio": -float("inf"),
            "iou_rate_0.25": -float("inf"),
            "iou_rate_0.5": -float("inf"),
            "det_mAP_0.25": -float("inf"),
            "det_mAP_0.5": -float("inf")
        }

        # init log
        # contains all necessary info for all phases
        self.log = {
            "train": {},
            "val": {}
        }
        
        # tensorboard
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        self._log_writer = {
            "train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train")),
            "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        }

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._global_epoch_id = 0
        self._total_iter = {}             # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE

        # lr scheduler
        if lr_decay_step and lr_decay_rate:
            if isinstance(lr_decay_step, list):
                self.lr_scheduler = MultiStepLR(optimizer, lr_decay_step, lr_decay_rate)
            else:
                self.lr_scheduler = StepLR(optimizer, lr_decay_step, lr_decay_rate)
        else:
            self.lr_scheduler = None

        # bn scheduler
        if bn_decay_step and bn_decay_rate:
            bn_lbmd = lambda it: max(self.args.bn_momentum_init * bn_decay_rate**(int(it / bn_decay_step)), self.args.bn_momentum_min)
            self.bn_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=-1)
        else:
            self.bn_scheduler = None
        
        #if distributed_rank:
        #    nn.SyncBatchNorm.convert_sync_batchnorm(self.model)


        # EVAL
        # config dict
        self.CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
                       'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
                       'per_class_proposal': True, 'conf_thresh': 0.0,
                       'dataset_config': ScannetDatasetConfig()}

        self.AP_IOU_THRESHOLDS = [0.25, 0.5]
        
        # add for distributed
        self.distributed_rank = distributed_rank


    def __call__(self, epoch, verbose):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["val"]) * int(epoch / self.val_freq)
        for epoch_id in range(epoch):
            try:
                self._log("epoch {} starting...".format(epoch_id + 1))
                
                # feed one epoch
                self._feed(self.dataloader["train"], "train", epoch_id)

                # save model
                self._log("saving last models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))
                
                # validation
                self._val(self.dataloader["val"], "val", epoch_id)

                # update lr scheduler
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                    print("update learning rate --> {}\n".format(self.lr_scheduler.get_lr()))

                # update bn scheduler
                if self.bn_scheduler:
                    self.bn_scheduler.step()
                    print("update batch normalization momentum --> {}\n".format(self.bn_scheduler.lmbd(self.bn_scheduler.last_epoch)))
                # add epoch id
                self._global_epoch_id += 1
                
            except KeyboardInterrupt:
                # finish training
                self._finish(epoch_id)
                exit()

        # finish training
        self._finish(epoch_id)

    def _log(self, info_str):
        # only main thread print once log
        if self.distributed_rank:
            if self.distributed_rank != 0:
                return
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str)

    def _reset_log(self, phase):
        self.log[phase] = {
            # info
            "forward": [],
            "backward": [],
            "eval": [],
            "fetch": [],
            "iter_time": [],
            # loss (float, not torch.cuda.FloatTensor)
            "loss": [],
            "ref_loss": [],
            "ref_mask_loss": [],
            "lang_cls_loss": [],
            "objectness_loss": [],
            "kps_loss": [],
            "box_loss": [],
            "sem_cls_loss": [],
            # scores (float, not torch.cuda.FloatTensor)
            "lang_cls_acc": [],
            "ref_acc": [],
            "obj_acc": [],
            "pos_ratio": [],
            "neg_ratio": [],
            "iou_rate_0.25": [],
            "iou_rate_0.5": []
        }

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, data_dict):
        data_dict = self.model(data_dict)

        return data_dict

    def _backward(self):
        # optimize
        self.optimizer.zero_grad()
        self._running_log["loss"].backward()
        self.optimizer.step()

    def _compute_loss(self, data_dict):
        _, data_dict = get_loss(
            data_dict=data_dict, 
            config=self.data_config,
            args=self.args,
        )

        # dump
        self._running_log["ref_loss"] = data_dict["ref_loss"]
        self._running_log["ref_mask_loss"] = data_dict["ref_mask_loss"]
        self._running_log["lang_cls_loss"] = data_dict["lang_cls_loss"]
        self._running_log["objectness_loss"] = data_dict["objectness_loss"]
        self._running_log["kps_loss"] = data_dict["kps_loss"]
        self._running_log["box_loss"] = data_dict["box_loss"]
        self._running_log["sem_cls_loss"] = data_dict["sem_cls_loss"]
        self._running_log["loss"] = data_dict["loss"]

    def _eval(self, data_dict):
        data_dict = get_eval(
            data_dict=data_dict,
            config=self.data_config,
            reference=self.reference,
            use_lang_classifier=self.use_lang_classifier
        )

        # dump
        self._running_log["lang_cls_acc"] = data_dict["lang_cls_acc"].item()
        self._running_log["ref_acc"] = np.mean(data_dict["ref_acc"])
        self._running_log["obj_acc"] = data_dict["obj_acc"].item()
        self._running_log["pos_ratio"] = data_dict["last_pos_ratio"].item()
        self._running_log["neg_ratio"] = data_dict["last_neg_ratio"].item()
        self._running_log["iou_rate_0.25"] = np.mean(data_dict["ref_iou_rate_0.25"])
        self._running_log["iou_rate_0.5"] = np.mean(data_dict["ref_iou_rate_0.5"])
        
    def _reset_running_log(self):
        self._running_log = {
            # loss
            "loss": 0,
            "ref_loss": 0,
            "ref_mask_loss": 0,
            "lang_cls_loss": 0,
            "objectness_loss": 0,
            "kps_loss": 0,
            "box_loss": 0,
            "sem_cls_loss": 0,
            # acc
            "lang_cls_acc": 0,
            "ref_acc": 0,
            "obj_acc": 0,
            "pos_ratio": 0,
            "neg_ratio": 0,
            "iou_rate_0.25": 0,
            "iou_rate_0.5": 0
        }
    def _record_log(self, phase):
        self.log[phase]["loss"].append(self._running_log["loss"].item())
        self.log[phase]["ref_loss"].append(self._running_log["ref_loss"].item())
        self.log[phase]["ref_mask_loss"].append(self._running_log["ref_mask_loss"].item())
        self.log[phase]["lang_cls_loss"].append(self._running_log["lang_cls_loss"].item())
        self.log[phase]["objectness_loss"].append(self._running_log["objectness_loss"].item())
        self.log[phase]["kps_loss"].append(self._running_log["kps_loss"].item())
        self.log[phase]["box_loss"].append(self._running_log["box_loss"].item())
        self.log[phase]["sem_cls_loss"].append(self._running_log["sem_cls_loss"].item())
        if not self.args.no_reference:
            self.log[phase]["lang_cls_acc"].append(self._running_log["lang_cls_acc"])
            self.log[phase]["ref_acc"].append(self._running_log["ref_acc"])
            self.log[phase]["obj_acc"].append(self._running_log["obj_acc"])
            self.log[phase]["pos_ratio"].append(self._running_log["pos_ratio"])
            self.log[phase]["neg_ratio"].append(self._running_log["neg_ratio"])
            self.log[phase]["iou_rate_0.25"].append(self._running_log["iou_rate_0.25"])
            self.log[phase]["iou_rate_0.5"].append(self._running_log["iou_rate_0.5"])
        else:
            self.log[phase]["lang_cls_acc"].append(0)
            self.log[phase]["ref_acc"].append(0)
            self.log[phase]["obj_acc"].append(0)
            self.log[phase]["pos_ratio"].append(0)
            self.log[phase]["neg_ratio"].append(0)
            self.log[phase]["iou_rate_0.25"].append(0)
            self.log[phase]["iou_rate_0.5"].append(0)

    def _feed(self, dataloader, phase, epoch_id):
        if self.distributed_rank:
            dataloader.sampler.set_epoch(epoch_id)
    
        # switch mode
        self._set_phase(phase)

        # re-init log
        self._reset_log(phase)

        # change dataloader
        # dataloader = dataloader if phase == "train" else tqdm(dataloader)

        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                if key != 'scene_id':
                    data_dict[key] = data_dict[key].cuda()

            # initialize the running loss
            self._reset_running_log()

            # load
            self.log[phase]["fetch"].append(data_dict["load_time"].sum().item())

            with torch.autograd.set_detect_anomaly(True):
                # forward
                start = time.time()
                data_dict = self._forward(data_dict)
                self._compute_loss(data_dict)
                self.log[phase]["forward"].append(time.time() - start)
                # backward
                start = time.time()
                self._backward()
                self.log[phase]["backward"].append(time.time() - start)
            
            # eval on train dataset
            start = time.time()
            if not self.args.no_reference:
                self._eval(data_dict)
            self.log[phase]["eval"].append(time.time() - start)

            # record log
            self._record_log(phase)

            # report
            iter_time = self.log[phase]["fetch"][-1]
            iter_time += self.log[phase]["forward"][-1]
            iter_time += self.log[phase]["backward"][-1]
            iter_time += self.log[phase]["eval"][-1]
            self.log[phase]["iter_time"].append(iter_time)
            if (self._global_iter_id + 1) % self.verbose == 0:
                self._train_report(epoch_id)

            # dump log
            self._dump_log(phase)
            self._global_iter_id += 1
    
    def _val(self, dataloader, phase, epoch_id):
        # evaluation
        print("evaluating...")
        # switch mode
        self._set_phase(phase)
        # re-init log
        self._reset_log(phase)
        # change dataloader
        dataloader = tqdm(dataloader)

        for data_dict in dataloader:
            self._reset_running_log()
            # move to cuda
            for key in data_dict:
                if key != 'scene_id':
                    data_dict[key] = data_dict[key].cuda()
            with torch.no_grad():
                # forward
                data_dict = self._forward(data_dict)
                self._compute_loss(data_dict)
                if not self.args.no_reference:
                    self._eval(data_dict)
                self._record_log(phase)
            
        # test mAP
        if self.args.eval_det:
            self.evaluate_detection_one_epoch(self.dataloader["val"], self.args)
        if self.args.eval_ref:
            self.evaluate_reference_one_epoch(self.dataloader["val"], self.args)
        
        self._dump_log("val")
        self._epoch_report(epoch_id)
        
        if self.args.no_reference:
            cur_criterion = 'det_mAP_0.5'
        else:
            cur_criterion = "iou_rate_0.5"
        cur_best = np.mean(self.log[phase][cur_criterion])
        if cur_best > self.best[cur_criterion]:
            self._log("best {} achieved: {}".format(cur_criterion, cur_best))
            self._log("current train_loss: {}".format(np.mean(self.log["train"]["loss"])))
            self._log("current val_loss: {}".format(np.mean(self.log["val"]["loss"])))
            self.best["epoch"] = epoch_id + 1
            self.best["loss"] = np.mean(self.log[phase]["loss"])
            self.best["ref_loss"] = np.mean(self.log[phase]["ref_loss"])
            self.best["ref_mask_loss"] = np.mean(self.log[phase]["ref_mask_loss"])
            self.best["lang_cls_loss"] = np.mean(self.log[phase]["lang_cls_loss"])
            self.best["objectness_loss"] = np.mean(self.log[phase]["objectness_loss"])
            self.best["kps_loss"] = np.mean(self.log[phase]["kps_loss"])
            self.best["box_loss"] = np.mean(self.log[phase]["box_loss"])
            self.best["sem_cls_loss"] = np.mean(self.log[phase]["sem_cls_loss"])
            self.best["lang_cls_acc"] = np.mean(self.log[phase]["lang_cls_acc"])
            self.best["ref_acc"] = np.mean(self.log[phase]["ref_acc"])
            self.best["obj_acc"] = np.mean(self.log[phase]["obj_acc"])
            self.best["pos_ratio"] = np.mean(self.log[phase]["pos_ratio"])
            self.best["neg_ratio"] = np.mean(self.log[phase]["neg_ratio"])
            self.best["iou_rate_0.25"] = np.mean(self.log[phase]["iou_rate_0.25"])
            self.best["iou_rate_0.5"] = np.mean(self.log[phase]["iou_rate_0.5"])
            self.best["det_mAP_0.25"] = np.mean(self.log[phase]["det_mAP_0.25"])
            self.best["det_mAP_0.5"] = np.mean(self.log[phase]["det_mAP_0.5"])

            # save model
            self._log("saving best models...\n")
            model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
            torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))

    def _dump_log(self, phase):
        log = {
            "train": {
                "loss": ["loss", "ref_loss", "ref_mask_loss", "lang_cls_loss", "objectness_loss", "kps_loss", "box_loss", "sem_cls_loss"],
                "score": ["lang_cls_acc", "ref_acc", "obj_acc", "pos_ratio", "neg_ratio", "iou_rate_0.25", "iou_rate_0.5"]
            },
            'val': {
                "loss": ["loss", "ref_loss", "ref_mask_loss", "lang_cls_loss", "objectness_loss", "kps_loss", "box_loss", "sem_cls_loss"],
                "score": ["lang_cls_acc", "ref_acc", "obj_acc", "pos_ratio", "neg_ratio", "iou_rate_0.25", "iou_rate_0.5", "det_mAP_0.25", "det_mAP_0.5"]
            }
        }
        index = {
            "train": self._global_iter_id,
            "val": self._global_iter_id,
        }
        if self.distributed_rank:
            if self.distributed_rank != 0:
                return
        for key in log[phase]:
            for item in log[phase][key]:
                self._log_writer[phase].add_scalar(
                    "{}/{}".format(key, item),
                    np.mean([v for v in self.log[phase][item]]),
                    index[phase]
                )

    def _finish(self, epoch_id):
        # print best
        self._best_report()

        # save check point
        self._log("saving checkpoint...\n")
        save_dict = {
            "epoch": epoch_id,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint.tar"))

        # save model
        self._log("saving last models...\n")
        model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

        # export
        for phase in ["train", "val"]:
            self._log_writer[phase].export_scalars_to_json(os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = self.log["train"]["fetch"]
        forward_time = self.log["train"]["forward"]
        backward_time = self.log["train"]["backward"]
        eval_time = self.log["train"]["eval"]
        iter_time = self.log["train"]["iter_time"]

        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        eta_sec = (self._total_iter["train"] - self._global_iter_id - 1) * mean_train_time
        eta_sec += (self._total_iter["val"] - int(self._global_epoch_id / self.val_freq * len(self.dataloader["val"]))) * mean_est_val_time
        eta = decode_eta(eta_sec)

        # print report
        iter_report = self.__iter_report_template.format(
            epoch_id=epoch_id + 1,
            iter_id=self._global_iter_id + 1,
            total_iter=self._total_iter["train"],
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_ref_loss=round(np.mean([v for v in self.log["train"]["ref_loss"]]), 5),
            train_ref_mask_loss=round(np.mean([v for v in self.log["train"]["ref_mask_loss"]]), 5),
            train_lang_cls_loss=round(np.mean([v for v in self.log["train"]["lang_cls_loss"]]), 5),
            train_objectness_loss=round(np.mean([v for v in self.log["train"]["objectness_loss"]]), 5),
            train_kps_loss=round(np.mean([v for v in self.log["train"]["kps_loss"]]), 5),
            train_box_loss=round(np.mean([v for v in self.log["train"]["box_loss"]]), 5),
            train_sem_cls_loss=round(np.mean([v for v in self.log["train"]["sem_cls_loss"]]), 5),
            train_lang_cls_acc=round(np.mean([v for v in self.log["train"]["lang_cls_acc"]]), 5),
            train_ref_acc=round(np.mean([v for v in self.log["train"]["ref_acc"]]), 5),
            train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
            train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
            train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
            train_iou_rate_25=round(np.mean([v for v in self.log["train"]["iou_rate_0.25"]]), 5),
            train_iou_rate_5=round(np.mean([v for v in self.log["train"]["iou_rate_0.5"]]), 5),
            mean_fetch_time=round(np.mean(fetch_time), 5),
            mean_forward_time=round(np.mean(forward_time), 5),
            mean_backward_time=round(np.mean(backward_time), 5),
            mean_eval_time=round(np.mean(eval_time), 5),
            mean_iter_time=round(np.mean(iter_time), 5),
            eta_h=eta["h"],
            eta_m=eta["m"],
            eta_s=eta["s"]
        )
        self._log(iter_report)

    def _epoch_report(self, epoch_id):
        self._log("epoch [{}/{}] done...".format(epoch_id+1, self.epoch))
        epoch_report = self.__epoch_report_template.format(
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_ref_loss=round(np.mean([v for v in self.log["train"]["ref_loss"]]), 5),
            train_ref_mask_loss=round(np.mean([v for v in self.log["train"]["ref_mask_loss"]]), 5),
            train_lang_cls_loss=round(np.mean([v for v in self.log["train"]["lang_cls_loss"]]), 5),
            train_objectness_loss=round(np.mean([v for v in self.log["train"]["objectness_loss"]]), 5),
            train_kps_loss=round(np.mean([v for v in self.log["train"]["kps_loss"]]), 5),
            train_box_loss=round(np.mean([v for v in self.log["train"]["box_loss"]]), 5),
            train_sem_cls_loss=round(np.mean([v for v in self.log["train"]["sem_cls_loss"]]), 5),
            train_lang_cls_acc=round(np.mean([v for v in self.log["train"]["lang_cls_acc"]]), 5),
            train_ref_acc=round(np.mean([v for v in self.log["train"]["ref_acc"]]), 5),
            train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
            train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
            train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
            train_iou_rate_25=round(np.mean([v for v in self.log["train"]["iou_rate_0.25"]]), 5),
            train_iou_rate_5=round(np.mean([v for v in self.log["train"]["iou_rate_0.5"]]), 5),
            val_loss=round(np.mean([v for v in self.log["val"]["loss"]]), 5),
            val_ref_loss=round(np.mean([v for v in self.log["val"]["ref_loss"]]), 5),
            val_ref_mask_loss=round(np.mean([v for v in self.log["val"]["ref_mask_loss"]]), 5),
            val_lang_cls_loss=round(np.mean([v for v in self.log["val"]["lang_cls_loss"]]), 5),
            val_objectness_loss=round(np.mean([v for v in self.log["val"]["objectness_loss"]]), 5),
            val_kps_loss=round(np.mean([v for v in self.log["val"]["kps_loss"]]), 5),
            val_box_loss=round(np.mean([v for v in self.log["val"]["box_loss"]]), 5),
            val_sem_cls_loss=round(np.mean([v for v in self.log["val"]["sem_cls_loss"]]), 5),
            val_lang_cls_acc=round(np.mean([v for v in self.log["val"]["lang_cls_acc"]]), 5),
            val_ref_acc=round(np.mean([v for v in self.log["val"]["ref_acc"]]), 5),
            val_obj_acc=round(np.mean([v for v in self.log["val"]["obj_acc"]]), 5),
            val_pos_ratio=round(np.mean([v for v in self.log["val"]["pos_ratio"]]), 5),
            val_neg_ratio=round(np.mean([v for v in self.log["val"]["neg_ratio"]]), 5),
            val_iou_rate_25=round(np.mean([v for v in self.log["val"]["iou_rate_0.25"]]), 5),
            val_iou_rate_5=round(np.mean([v for v in self.log["val"]["iou_rate_0.5"]]), 5),
        )
        self._log(epoch_report)
    
    def _best_report(self):
        self._log("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            loss=round(self.best["loss"], 5),
            ref_loss=round(self.best["ref_loss"], 5),
            ref_mask_loss=round(self.best["ref_mask_loss"], 5),
            lang_cls_loss=round(self.best["lang_cls_loss"], 5),
            objectness_loss=round(self.best["objectness_loss"], 5),
            kps_loss=round(self.best["kps_loss"], 5),
            box_loss=round(self.best["box_loss"], 5),
            sem_cls_loss=round(self.best["sem_cls_loss"], 5),
            lang_cls_acc=round(self.best["lang_cls_acc"], 5),
            ref_acc=round(self.best["ref_acc"], 5),
            obj_acc=round(self.best["obj_acc"], 5),
            pos_ratio=round(self.best["pos_ratio"], 5),
            neg_ratio=round(self.best["neg_ratio"], 5),
            iou_rate_25=round(self.best["iou_rate_0.25"], 5),
            iou_rate_5=round(self.best["iou_rate_0.5"], 5),
        )
        self._log(best_report)
        with open(os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)

    def evaluate_detection_one_epoch(self, test_loader, config):
        self._log('=====================>DETECTION EVAL<=====================')
        stat_dict = {}

        if config.num_decoder_layers > 0:
            prefixes = ['last_', 'proposal_'] + [f'{i}head_' for i in range(config.num_decoder_layers - 1)]
        else:
            prefixes = ['proposal_']  # only proposal
        ap_calculator_list = [APCalculator(iou_thresh, self.data_config.class2type) \
                              for iou_thresh in self.AP_IOU_THRESHOLDS]
        mAPs = [[iou_thresh, {k: 0 for k in prefixes}] for iou_thresh in self.AP_IOU_THRESHOLDS]

        self.model.eval()  # set model to eval mode (for bn and dp)
        batch_pred_map_cls_dict = {k: [] for k in prefixes}
        batch_gt_map_cls_dict = {k: [] for k in prefixes}

        scene_set = set()

        for batch_idx, batch_data_label in enumerate(test_loader):
            scene_idx_list = []
            for i, scene_id in enumerate(batch_data_label['scene_id']):
                if scene_id not in scene_set:
                    scene_set.add(scene_id)
                    scene_idx_list.append(i)

            if len(scene_idx_list) == 0:
                continue

            for key in batch_data_label:
                if key != 'scene_id':
                    batch_data_label[key] = batch_data_label[key][scene_idx_list, ...].cuda(non_blocking=True)

            # Forward pass
            with torch.no_grad():
                data_dict = self.model(batch_data_label)

            # Compute loss
            for key in batch_data_label:
                if key in data_dict:
                    continue
                data_dict[key] = batch_data_label[key]
            loss, data_dict = get_loss(data_dict, self.data_config, self.args)

            # Accumulate statistics and print out
            for key in data_dict:
                if 'loss' in key or 'acc' in key or 'ratio' in key:
                    if 'weights' in key:    # add for not save loss_weight
                        continue
                    if key not in stat_dict: stat_dict[key] = 0
                    if isinstance(data_dict[key], float):
                        stat_dict[key] += data_dict[key]
                    else:
                        stat_dict[key] += data_dict[key].item()

            for prefix in prefixes:
                batch_pred_map_cls = parse_predictions(data_dict, self.CONFIG_DICT, prefix,
                                                       size_cls_agnostic=config.size_cls_agnostic)
                batch_gt_map_cls = parse_groundtruths(data_dict, self.CONFIG_DICT,
                                                      size_cls_agnostic=config.size_cls_agnostic)
                batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
                batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

        mAP = 0.0
        for prefix in prefixes:
            for (batch_pred_map_cls, batch_gt_map_cls) in zip(batch_pred_map_cls_dict[prefix],
                                                              batch_gt_map_cls_dict[prefix]):
                for ap_calculator in ap_calculator_list:
                    ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
            # Evaluate average precision
            for i, ap_calculator in enumerate(ap_calculator_list):
                metrics_dict = ap_calculator.compute_metrics()
                self._log(f'=====================>{prefix} IOU THRESH: {self.AP_IOU_THRESHOLDS[i]}<=====================')
                for key in metrics_dict:
                    self._log(f'{key} {metrics_dict[key]}')
                if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                    mAP = metrics_dict['mAP']
                mAPs[i][1][prefix] = metrics_dict['mAP']
                ap_calculator.reset()

        for mAP in mAPs:
            self._log(
                f'IoU[{mAP[0]}]:\t' + ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))
        # add map to log
        for mAP in mAPs:
            self.log['val']["det_mAP_{}".format(mAP[0])] = [mAP[1]['last_']]
        return mAP, mAPs

    def evaluate_reference_one_epoch(self, test_loader, config):
        self._log('=====================>REFERENCE EVAL<=====================')
        stat_dict = {}

        if config.num_decoder_layers > 0:
            prefixes = ['last_']
        else:
            return
        ap_calculator_list = [APCalculator(iou_thresh, self.data_config.class2type) \
                              for iou_thresh in self.AP_IOU_THRESHOLDS]
        mAPs = [[iou_thresh, {k: 0 for k in prefixes}] for iou_thresh in self.AP_IOU_THRESHOLDS]

        self.model.eval()  # set model to eval mode (for bn and dp)
        batch_pred_map_cls_dict = {k: [] for k in prefixes}
        batch_gt_map_cls_dict = {k: [] for k in prefixes}

        for batch_idx, batch_data_label in enumerate(test_loader):
            for key in batch_data_label:
                if key != 'scene_id':
                    batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

            # Forward pass
            with torch.no_grad():
                data_dict = self.model(batch_data_label)

            # Compute loss
            for key in batch_data_label:
                if key in data_dict:
                    continue
                data_dict[key] = batch_data_label[key]
            loss, data_dict = get_loss(data_dict, self.data_config, self.args)

            # Accumulate statistics and print out
            for key in data_dict:
                if 'loss' in key or 'acc' in key or 'ratio' in key:
                    if 'weights' in key:    # add for not save loss_weight
                        continue
                    if key not in stat_dict: stat_dict[key] = 0
                    if isinstance(data_dict[key], float):
                        stat_dict[key] += data_dict[key]
                    else:
                        stat_dict[key] += data_dict[key].item()

            for prefix in prefixes:
                batch_pred_map_cls = parse_ref_predictions(data_dict, self.CONFIG_DICT, prefix,
                                                       size_cls_agnostic=config.size_cls_agnostic)
                batch_gt_map_cls = parse_ref_groundtruths(data_dict, self.CONFIG_DICT,
                                                      size_cls_agnostic=config.size_cls_agnostic)
                batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
                batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

        mAP = 0.0
        for prefix in prefixes:
            for (batch_pred_map_cls, batch_gt_map_cls) in zip(batch_pred_map_cls_dict[prefix],
                                                              batch_gt_map_cls_dict[prefix]):
                for ap_calculator in ap_calculator_list:
                    ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
            # Evaluate average precision
            for i, ap_calculator in enumerate(ap_calculator_list):
                metrics_dict = ap_calculator.compute_metrics()
                self._log(f'=====================>{prefix} IOU THRESH: {self.AP_IOU_THRESHOLDS[i]}<=====================')
                for key in metrics_dict:
                    self._log(f'{key} {metrics_dict[key]}')
                if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                    mAP = metrics_dict['mAP']
                mAPs[i][1][prefix] = metrics_dict['mAP']
                ap_calculator.reset()

        for mAP in mAPs:
            self._log(
                f'IoU[{mAP[0]}]:\t' + ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))

        return mAP, mAPs
    