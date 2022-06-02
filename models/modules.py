import torch
import torch.nn as nn
import numpy as np
import sys
import os
import torch.nn.functional as F
from models.mlp import MLP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(os.getcwd(), "lib"))
sys.path.append(os.path.join(os.getcwd(), "lib", 'pointnet2'))


class PredictHead(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster,
                 mean_size_arr, num_proposal, seed_feat_dim=256, lang_feat_dim=256, 
                 use_ref_branch=False, use_cls_branch=False, use_ref_mask=False, use_objectness=True,
                 args = None):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim
        self.lang_feat_dim = lang_feat_dim
        self.use_ref_branch = use_ref_branch
        self.use_cls_branch = use_cls_branch
        self.use_ref_mask = use_ref_mask
        self.args = args
        self.use_objectness = use_objectness

        # Object proposal/detection
        # Objectness scores (1), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(seed_feat_dim, seed_feat_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(seed_feat_dim)
        self.conv2 = torch.nn.Conv1d(seed_feat_dim, seed_feat_dim, 1)
        self.bn2 = torch.nn.BatchNorm1d(seed_feat_dim)
        if self.use_objectness:
            self.objectness_scores_head = torch.nn.Conv1d(seed_feat_dim, 1, 1)
        self.center_residual_head = torch.nn.Conv1d(seed_feat_dim, 3, 1)
        self.heading_class_head = torch.nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
        self.heading_residual_head = torch.nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
        self.size_class_head = torch.nn.Conv1d(seed_feat_dim, num_size_cluster, 1)
        self.size_residual_head = torch.nn.Conv1d(seed_feat_dim, num_size_cluster * 3, 1)
        self.sem_cls_scores_head = torch.nn.Conv1d(seed_feat_dim, self.num_class, 1)
        if self.use_ref_branch:
            in_channel = self.seed_feat_dim + self.lang_feat_dim
            out_channel = int(in_channel / 2)
            #self.lang_ref_branch = AttentionModule(self.lang_feat_dim, self.lang_feat_dim, 1, self.args.dropout, self.args.dropout)
            self.fusion_layer = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, 1),
                nn.ReLU(),
                nn.BatchNorm1d(out_channel),
                nn.Conv1d(out_channel, out_channel, 1),
                nn.ReLU(),
                nn.BatchNorm1d(out_channel),
            )
            self.ref_scores_head = nn.Conv1d(out_channel, 1, 1)
        if self.use_cls_branch:
            #self.lang_clf_branch = AttentionModule(self.args.transformer_feat_dim, self.args.transformer_feat_dim, 1, self.args.dropout, self.args.dropout)
            self.lang_clf = MLP(self.args.transformer_feat_dim, [128, 256, self.num_class], dropout_rate=self.args.dropout)
        if self.use_ref_mask:
            self.ref_mask_scores_head = torch.nn.Conv1d(out_channel, 1, 1)

    def forward(self, features, base_xyz, end_points, prefix='', lang_feat=None, lang_mask=None, 
                cross_object_feat=None, cross_lang_feat=None, prefix_index=None):
        """
        Args:
            features: (B,C,num_proposal)
            lang_feat: (B, M, C), lang_mask: (B, M)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        batch_size = features.shape[0]
        num_proposal = features.shape[-1]
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))
        # objectness
        if self.use_objectness:
            objectness_scores = self.objectness_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, 1)
        else:
            if prefix_index > 0 and (prefix_index-1) in self.args.ref_filter_steps:
                objectness_scores = torch.gather(end_points['proposal_objectness_scores'], 1, 
                                                 end_points[f'{prefix_index-1}head_ref_mask_inds'].unsqueeze(-1))
            else:
                objectness_scores = end_points['proposal_objectness_scores']
            
        # center
        center_residual = self.center_residual_head(net).transpose(2, 1)  # (batch_size, num_proposal, 3)
        center = base_xyz + center_residual  # (batch_size, num_proposal, 3)

        # heading
        heading_scores = self.heading_class_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_heading_bin)
        # (batch_size, num_proposal, num_heading_bin) (should be -1 to 1)
        heading_residuals_normalized = self.heading_residual_head(net).transpose(2, 1)
        heading_residuals = heading_residuals_normalized * (np.pi / self.num_heading_bin)

        # size
        mean_size_arr = torch.from_numpy(self.mean_size_arr.astype(np.float32)).cuda()  # (num_size_cluster, 3)
        mean_size_arr = mean_size_arr.unsqueeze(0).unsqueeze(0)  # (1, 1, num_size_cluster, 3)
        size_scores = self.size_class_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_size_cluster)
        size_residuals_normalized = self.size_residual_head(net).transpose(2, 1).view(
            [batch_size, num_proposal, self.num_size_cluster, 3])  # (batch_size, num_proposal, num_size_cluster, 3)
        size_residuals = size_residuals_normalized * mean_size_arr  # (batch_size, num_proposal, num_size_cluster, 3)
        size_recover = size_residuals + mean_size_arr  # (batch_size, num_proposal, num_size_cluster, 3)
        pred_size_class = torch.argmax(size_scores, -1)  # batch_size, num_proposal
        pred_size_class = pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)
        pred_size = torch.gather(size_recover, 2, pred_size_class)  # batch_size, num_proposal, 1, 3
        pred_size = pred_size.squeeze_(2)  # batch_size, num_proposal, 3

        # class
        sem_cls_scores = self.sem_cls_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_class)
        
        # ref
        if self.use_ref_branch:
            # fuse object and lang feat
            lang_mask = lang_mask.unsqueeze(-1) * lang_mask.unsqueeze(-2)                                   # [B, M, M]
            #cross_lang_feat = self.lang_ref_branch(cross_lang_feat, cross_lang_feat, cross_lang_feat, lang_mask).max(dim=1)[0]      # [B, C]
            cross_lang_feat = cross_lang_feat.max(dim=1)[0]
            cross_lang_feat = cross_lang_feat.unsqueeze(-1).repeat(1, 1, net.shape[-1])                                 # [B, C, N]
            fusion_feat = torch.cat([cross_object_feat, cross_lang_feat], dim=1)                                           # [B, C', N]
            objectness_masks = (objectness_scores > 0).float()  # (batch_size, num_proposal, 1)
            objectness_masks = objectness_masks.permute(0, 2, 1).contiguous() # batch_size, 1, num_proposals
            fusion_feat = fusion_feat * objectness_masks
            fusion_feat = self.fusion_layer(fusion_feat)
            ref_scores = self.ref_scores_head(fusion_feat).squeeze_(1)  # (batch_size, num_proposal)
            end_points[f'{prefix}ref_scores'] = ref_scores
        # ref mask
        if self.use_ref_mask:
            ref_mask_scores = self.ref_mask_scores_head(fusion_feat).transpose(2, 1)    # (batch_size, num_proposal, 1)
            end_points[f'{prefix}ref_mask_scores'] = ref_mask_scores
        
        if self.use_cls_branch:
            lang_mask_2d = lang_mask.unsqueeze(-1) * lang_mask.unsqueeze(-2)
            #lang_feat_clf = self.lang_clf_branch(lang_feat, lang_feat, lang_feat, lang_mask_2d).max(dim=1)[0]         # B x C
            lang_feat_clf = lang_feat.max(dim=1)[0]
            end_points[f'{prefix}lang_logits'] = self.lang_clf(lang_feat_clf)                                         # [B, n_class]

        end_points[f'{prefix}base_xyz'] = base_xyz
        end_points[f'{prefix}objectness_scores'] = objectness_scores
        end_points[f'{prefix}center'] = center
        end_points[f'{prefix}heading_scores'] = heading_scores
        end_points[f'{prefix}heading_residuals_normalized'] = heading_residuals_normalized
        end_points[f'{prefix}heading_residuals'] = heading_residuals
        end_points[f'{prefix}size_scores'] = size_scores
        end_points[f'{prefix}size_residuals_normalized'] = size_residuals_normalized
        end_points[f'{prefix}size_residuals'] = size_residuals
        end_points[f'{prefix}pred_size'] = pred_size
        end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores

        # # used to check bbox size
        # l = pred_size[:, :, 0]
        # h = pred_size[:, :, 1]
        # w = pred_size[:, :, 2]
        # x_corners = torch.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], -1)  # N Pq 8
        # y_corners = torch.stack([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], -1)  # N Pq 8
        # z_corners = torch.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], -1)  # N Pq 8
        # corners = torch.stack([x_corners, y_corners, z_corners], -1)  # N Pq 8 3
        # bbox = center.unsqueeze(2) + corners
        # end_points[f'{prefix}bbox_check'] = bbox
        return center, pred_size


class ClsAgnosticPredictHead(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_proposal, seed_feat_dim=256):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim

        # Object proposal/detection
        # Objectness scores (1), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(seed_feat_dim, seed_feat_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(seed_feat_dim)
        self.conv2 = torch.nn.Conv1d(seed_feat_dim, seed_feat_dim, 1)
        self.bn2 = torch.nn.BatchNorm1d(seed_feat_dim)

        self.objectness_scores_head = torch.nn.Conv1d(seed_feat_dim, 1, 1)
        self.center_residual_head = torch.nn.Conv1d(seed_feat_dim, 3, 1)
        self.heading_class_head = torch.nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
        self.heading_residual_head = torch.nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
        self.size_pred_head = torch.nn.Conv1d(seed_feat_dim, 3, 1)
        self.sem_cls_scores_head = torch.nn.Conv1d(seed_feat_dim, self.num_class, 1)

    def forward(self, features, base_xyz, end_points, prefix=''):
        """
        Args:
            features: (B,C,num_proposal)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        batch_size = features.shape[0]
        num_proposal = features.shape[-1]
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))
        # objectness
        objectness_scores = self.objectness_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, 1)
        # center
        center_residual = self.center_residual_head(net).transpose(2, 1)  # (batch_size, num_proposal, 3)
        center = base_xyz + center_residual  # (batch_size, num_proposal, 3)

        # heading
        heading_scores = self.heading_class_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_heading_bin)
        # (batch_size, num_proposal, num_heading_bin) (should be -1 to 1)
        heading_residuals_normalized = self.heading_residual_head(net).transpose(2, 1)
        heading_residuals = heading_residuals_normalized * (np.pi / self.num_heading_bin)

        # size
        pred_size = self.size_pred_head(net).transpose(2, 1).view(
            [batch_size, num_proposal, 3])  # (batch_size, num_proposal, 3)

        # class
        sem_cls_scores = self.sem_cls_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_class)

        end_points[f'{prefix}base_xyz'] = base_xyz
        end_points[f'{prefix}objectness_scores'] = objectness_scores
        end_points[f'{prefix}center'] = center
        end_points[f'{prefix}heading_scores'] = heading_scores
        end_points[f'{prefix}heading_residuals_normalized'] = heading_residuals_normalized
        end_points[f'{prefix}heading_residuals'] = heading_residuals
        end_points[f'{prefix}pred_size'] = pred_size
        end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores

        return center, pred_size