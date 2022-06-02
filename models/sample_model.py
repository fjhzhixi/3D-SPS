import torch
import torch.nn as nn
import numpy as np
import sys
import os
import torch.nn.functional as F

from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(os.getcwd(), "lib"))
sys.path.append(os.path.join(os.getcwd(), "lib", 'pointnet2'))
import pointnet2_utils


class SamplingModule(nn.Module):
    """
    Sample object proposal.
    """
    def __init__(self, sampling_method, num_proposal, feat_dim, lang_dim, args):
        super().__init__()
        self.sampling_method = sampling_method
        self.num_proposal = num_proposal
        self.args = args
        if sampling_method == 'fps':
            self.fps_module = FPSModule(num_proposal)
        elif sampling_method == 'kps':
            self.points_obj_cls = PointsObjClsModule(feat_dim)
            self.gsample_module = GeneralSamplingModule()
        elif sampling_method == 'kpsa-lang-filter':
            self.points_obj_cls = PointsObjClsModule(feat_dim)
            self.sa_module = SaSamplingModule(int((1024 + self.num_proposal) / 2), feat_dim)
            self.match_module = MatchModule(object_dim=feat_dim, lang_dim=lang_dim, fusion_dim=args.kps_fusion_dim)
            self.gsample_module = GeneralSamplingModule()
        else:
            raise NotImplementedError

    def forward(self, xyz, features, data_dict):
        # xyz, features, sample_inds = (None, None, None)
        # points_obj_cls_logits = None
        if self.sampling_method == 'fps':
            xyz, features, sample_inds = self.fps_module(xyz, features)
            # cluster_feature = features
            # cluster_xyz = xyz
            data_dict['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            data_dict['query_points_feature'] = features  # (batch_size, C, num_proposal)
            data_dict['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        elif self.sampling_method == 'kps':
            points_obj_cls_logits = self.points_obj_cls(features)  # (batch_size, 1, num_seed)
            data_dict['seeds_obj_cls_logits'] = points_obj_cls_logits
            points_obj_cls_scores = torch.sigmoid(points_obj_cls_logits).squeeze(1)
            sample_inds = torch.topk(points_obj_cls_scores, self.num_proposal)[1].int()
            xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
            # cluster_feature = features
            # cluster_xyz = xyz
            data_dict['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            data_dict['query_points_feature'] = features  # (batch_size, C, num_proposal)
            data_dict['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        elif self.sampling_method == 'kpsa-lang-filter':
            points_obj_cls_logits = self.points_obj_cls(features)  # (batch_size, 1, num_seed)
            data_dict['seeds_obj_cls_logits'] = points_obj_cls_logits
            points_obj_cls_scores = torch.sigmoid(points_obj_cls_logits).squeeze(1)
            sample_inds = torch.topk(points_obj_cls_scores, int((1024 + self.num_proposal) / 2))[1].int()
            xyz, features, sample_inds = self.sa_module(xyz, features, sample_inds)
            # cluster_feature = features
            # cluster_xyz = xyz
            data_dict['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            data_dict['query_points_feature'] = features  # (batch_size, C, num_proposal)
            data_dict['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
            ref_scores = self.match_module(features, data_dict['lang_hidden'], data_dict)
            ref_scores = torch.sigmoid(ref_scores).squeeze(1)   # [B, N]
            sample_inds = torch.topk(ref_scores, self.num_proposal)[1].int()
            xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
            data_dict['ref_query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            data_dict['ref_query_points_feature'] = features  # (batch_size, C, num_proposal)
            data_dict['ref_query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        else:
            raise NotImplementedError
 
        return data_dict, xyz, features

class MatchModule(nn.Module):
    def __init__(self, object_dim, lang_dim, fusion_dim):
        super().__init__()

        self.match = nn.Sequential(
            nn.Conv1d(object_dim + lang_dim, fusion_dim, 1),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Conv1d(fusion_dim, fusion_dim, 1),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Conv1d(fusion_dim, 1, 1),
        )

    def forward(self, object_feat, lang_feat, data_dict):
        """
        Args:
            object_feat: (B,C,num_proposal)
            lang_feat: (B,C)
        Returns:
            scores: (B,1,num_proposal)
        """
        # fuse
        num_proposal = object_feat.shape[-1]
        lang_feat = lang_feat.unsqueeze(-1).repeat(1, 1, num_proposal)   # [B, C, N]
        features = torch.cat([object_feat, lang_feat], dim=1) # [B, C, N]

        # match
        confidences = self.match(features)          # [B, 1, N]
        
        data_dict['kps_ref_score'] = confidences

        return confidences

class AttenModule(nn.Module):
    def __init__(self, object_dim, lang_dim, dropout):
        super().__init__()
        self.object_att_fc = nn.Sequential(
            nn.Linear(object_dim, lang_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(lang_dim, lang_dim, bias=True),
        )
        self.atten_fc = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(lang_dim * 2, 1, bias=True)
        )
        self.object_score_fc = nn.Sequential(
            nn.Linear(object_dim, lang_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(lang_dim, lang_dim, bias=True),
        )
        self.score_fc = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(lang_dim, 1, bias=True)
        )
    def forward(self, object_feat, lang_feat, lang_mask, data_dict):
        """
        Args:
            object_feat: (B,N,C1)
            lang_feat: (B,M,C2)
            lang_mask: (B,M)
        Returns:
            scores: (B,num_proposal,1)
        """
        # attention for lang word
        object_num = object_feat.shape[1]
        lang_num = lang_feat.shape[1]
        object_att = self.object_att_fc(object_feat)                            # [B, N, C2]
        object_att = object_att.unsqueeze(dim=2).expand(-1, -1, lang_num, -1)   # [B, N, M, C2]
        lang_att = lang_feat.unsqueeze(dim=1).expand(-1, object_num, -1, -1)    # [B, N, M, C2]
        att_score = torch.cat([object_att, lang_att], dim=-1)                   # [B, N, M, 2 * C2]
        att_score = self.atten_fc(att_score)                                    # [B, N, M, 1]
        lang_mask = lang_mask[:, None, :, None].expand(-1, object_num, -1, -1)  # [B, N, M, 1]
        att_score[lang_mask == 0] = float('-inf')                               # [B, N, M, 1]
        att_score = torch.softmax(att_score, dim=2)                             # [B, N, M, 1]
        att_score = torch.sum(att_score * lang_att, dim=2)                      # [B, N, C2]
        # score for object
        object_score = self.object_score_fc(object_feat)                        # [B, N, C2]
        object_score = att_score * object_score                                 # [B, N, C2]
        object_score = F.normalize(F.relu(object_score, inplace=True), dim=2)   # [B, N, C2]
        object_score = self.score_fc(object_score).transpose(1,2)               # [B, 1, N]
        
        data_dict['kps_ref_score'] = object_score
        return object_score

class PointsObjClsModule(nn.Module):
    def __init__(self, seed_feature_dim):
        """ object candidate point prediction from seed point features.

        Args:
            seed_feature_dim: int
                number of channels of seed point features
        """
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        self.conv3 = torch.nn.Conv1d(self.in_dim, 1, 1)

    def forward(self, seed_features):
        """ Forward pass.

        Arguments:
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            logits: (batch_size, 1, num_seed)
        """
        net = F.relu(self.bn1(self.conv1(seed_features)))
        net = F.relu(self.bn2(self.conv2(net)))
        logits = self.conv3(net)  # (batch_size, 1, num_seed)

        return logits

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class FPSModule(nn.Module):
    def __init__(self, num_proposal):
        super().__init__()
        self.num_proposal = num_proposal

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        """
        # Farthest point sampling (FPS)
        sample_inds = pointnet2_utils.furthest_point_sample(xyz, self.num_proposal)
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sample_inds).transpose(1, 2).contiguous()
        new_features = pointnet2_utils.gather_operation(features, sample_inds).contiguous()

        return new_xyz, new_features, sample_inds


class GeneralSamplingModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz, features, sample_inds):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        """
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sample_inds).transpose(1, 2).contiguous()
        new_features = pointnet2_utils.gather_operation(features, sample_inds).contiguous()

        return new_xyz, new_features, sample_inds


class SaSamplingModule(nn.Module):
    def __init__(self, num_proposal, seed_feat_dim, mlp_layer_num=4):
        super().__init__()
        self.sa_module = PointnetSAModuleVotes(
            npoint=num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[seed_feat_dim for _ in range(mlp_layer_num)],
            use_xyz=True,
            normalize_xyz=True
        )

    def forward(self, xyz, features, sample_inds=None):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        """
        new_xyz, new_features, new_sample_inds = self.sa_module(xyz, features, sample_inds)
        # new_sample_inds should be same as sample_inds if it is not None

        return new_xyz, new_features, new_sample_inds
