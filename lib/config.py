import os
import sys
import argparse
import yaml
from easydict import EasyDict

class Config():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--config', type=str, default='config/default.yaml', help='path to config file')
        self.parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
        self.parser.add_argument("--folder", type=str, help="Folder containing the model")
        self.parser.add_argument("--force", action="store_true", help="enforce the generation of results")
        self.parser.add_argument("--repeat", type=int, default=1, help="Number of times for evaluation")
        self.parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
        self.parser.add_argument("--reference", action="store_true", help="evaluate the reference localization results")
        self.parser.add_argument("--detection", action="store_true", help="evaluate the object detection results")
        self.parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
        self.parser.add_argument("--use_test", action="store_true", help="Use test split in evaluation.")
        self.parser.add_argument("--use_oracle", action="store_true", help="Use ground truth bounding boxes.")
        self.parser.add_argument("--use_cat_rand", action="store_true", help="Use randomly selected bounding boxes from correct categories as outputs.")
        self.parser.add_argument("--use_best", action="store_true", help="Use best bounding boxes as outputs.")
        self.parser.add_argument("--scene_id", type=str, help="scene id", default="")
        self.parser.add_argument("--maxpool", action="store_true", help="use max pooling to aggregate features (use majority voting in label projection mode)")
        
    def get_config(self):
        cfgs = self.parser.parse_args()
        assert cfgs.config is not None
        with open(cfgs.config, 'r') as f:
            config = yaml.safe_load(f)
        for key in config:
            for k, v in config[key].items():
                setattr(cfgs, k, v)
        self.set_paths_cfg(cfgs)
        return cfgs

    def set_paths_cfg(self, CONF):
        CONF.PATH = EasyDict()
        CONF.PATH.BASE = CONF.root_path
        CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
        CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")
        CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
        CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
        CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

        # append to syspath
        for _, path in CONF.PATH.items():
            sys.path.append(path)

        # scannet data
        CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
        CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
        CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, CONF.scannet_data_folder)

        # scanref data
        CONF.SCANNET_FRAMES_ROOT = os.path.join(CONF.scanref_data_root, 'frames_square')
        CONF.PROJECTION = os.path.join(CONF.scanref_data_root, 'multiview_projection_scanrefer')
        CONF.ENET_FEATURES_ROOT = os.path.join(CONF.scanref_data_root, 'enet_features')
        CONF.ENET_FEATURES_SUBROOT = os.path.join(CONF.ENET_FEATURES_ROOT, "{}") # scene_id
        CONF.ENET_FEATURES_PATH = os.path.join(CONF.ENET_FEATURES_SUBROOT, "{}.npy") # frame_id
        CONF.SCANNET_FRAMES = os.path.join(CONF.SCANNET_FRAMES_ROOT, "{}/{}") # scene_id, mode 
        # CONF.SCENE_NAMES = sorted(os.listdir(CONF.PATH.SCANNET_SCANS))
        CONF.ENET_WEIGHTS = os.path.join(CONF.PATH.BASE, "data/scannetv2_enet.pth")
        CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats_maxpool.hdf5")
        CONF.NYU40_LABELS = os.path.join(CONF.PATH.SCANNET_META, "nyu40_labels.csv")

        # scannet split
        CONF.SCANNETV2_TRAIN = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_train.txt")
        CONF.SCANNETV2_VAL = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_val.txt")
        CONF.SCANNETV2_TEST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_test.txt")
        CONF.SCANNETV2_LIST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2.txt")

        # output
        CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")
              
CONF = Config().get_config()