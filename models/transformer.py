import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from attention_module import *

class TransformerFilter(nn.Module):
    """
                    | CA |      | CA |
                    | CA | 
                    | SA |      | SA |         
                | Obj Feat | | Text Feat | | Point Feat |
        Obj Feat:  B C Pq
        Text Feat: 
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.05, activation="relu", 
                 object_position_embedding='none', point_position_embedding='none', lang_position_embedding='none', ret_att=False):
        super().__init__()
        self.obj_sa = AttentionModule(d_model, dim_feedforward, nhead, dropout, dropout)
        self.text_sa = AttentionModule(d_model, dim_feedforward, nhead, dropout, dropout)
        self.obj_point_ca = AttentionModule(d_model, dim_feedforward, nhead, dropout, dropout)
        self.obj_text_ca = AttentionModule(d_model, dim_feedforward, nhead, dropout, dropout, ret_att)
        self.text_obj_ca = AttentionModule(d_model, dim_feedforward, nhead, dropout, dropout, ret_att)
        self.object_posembed, self.point_posembed, self.lang_posembed = get_position_embedding(
            object_position_embedding, point_position_embedding, lang_position_embedding, d_model)
        

    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, object_feat, object_pose, object_mask,
                      point_feat, point_pose,
                      lang_feat, lang_pose, lang_mask):
        """
        object_feat: [B,K,C]     object_mask: [B,K]      object_pose: [B,K,6]
        point_feat: [B,N,C]                              point_pose: [B,N,3]
        lang_feat: [B,M,C]       lang_mask: [B,M]        lang_pose:[B,M,1]
        """
        if self.object_posembed is not None:
            object_pos_embed = self.object_posembed(object_pose).permute(0, 2, 1)   # [B, K, C]
            object_feat = self.with_pos_embed(object_feat, object_pos_embed)
        if self.point_posembed is not None:
            point_pose_embed = self.point_posembed(point_pose).permute(0, 2, 1)     # [B, N, C]
            point_feat = self.with_pos_embed(point_feat, point_pose_embed)
        if self.lang_posembed is not None:
            lang_pose_embed = self.lang_posembed(lang_pose).permute(0, 2, 1)        # [B, M, C]
            lang_feat = self.with_pos_embed(lang_feat, lang_pose_embed)
        
        # add mask to 1 dim
        object_mask = object_mask.unsqueeze(-1)     # [B, K, 1]
        lang_mask = lang_mask.unsqueeze(-1)         # [B, M, 1]
        
        # input is q, k, v, mask
        # object sa layer
        mask = object_mask * object_mask.transpose(-1, -2)    # [B, K, K]
        object_feat, _ = self.obj_sa(object_feat, object_feat, object_feat, mask)
        # object ca point layer
        mask = object_mask * torch.ones((point_feat.shape[0], 1, point_feat.shape[1])).cuda() # [B, K, N]
        object_feat, _ = self.obj_point_ca(object_feat, point_feat, point_feat, mask)

        # lang sa layer
        mask = lang_mask * lang_mask.transpose(-1, -2)    # [B, M, M]
        lang_feat, _ = self.text_sa(lang_feat, lang_feat, lang_feat, mask)
        
        # object ca lang layer
        mask = object_mask * lang_mask.transpose(-1, -2)    # [B, K, M]
        cross_object_feat, cross_obj_att = self.obj_text_ca(object_feat, lang_feat, lang_feat, mask)
        
        # lang ca object layer
        mask = lang_mask * object_mask.transpose(-1, -2)    # [B, M, K]
        cross_lang_feat, cross_lang_att = self.text_obj_ca(lang_feat, object_feat, object_feat, mask)
        
        return cross_object_feat, cross_lang_feat, object_feat, lang_feat, cross_obj_att, cross_lang_att

def get_position_embedding(self_position_embedding, cross_position_embedding, lang_position_embedding, d_model):
    self_posembed = None 
    if self_position_embedding == 'none':
        self_posembed = None
    elif self_position_embedding == 'xyz_learned':
        self_posembed = PositionEmbeddingLearned(3, d_model)
    elif self_position_embedding == 'loc_learned':
        self_posembed = PositionEmbeddingLearned(6, d_model)
    else:
        raise NotImplementedError(f"self_position_embedding not supported {self_position_embedding}")
    cross_posembed = None
    if cross_position_embedding == 'none':
        cross_posembed = None
    elif cross_position_embedding == 'xyz_learned':
        cross_posembed = PositionEmbeddingLearned(3, d_model)
    elif cross_position_embedding == 'loc_learned':
        cross_posembed = PositionEmbeddingLearned(6, d_model)
    else:
        raise NotImplementedError(f"cross_position_embedding not supported {cross_position_embedding}")
    lang_posembed = None
    if lang_position_embedding == 'none':
        lang_posembed = None
    return self_posembed, cross_posembed, lang_posembed

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

class PositionalEncoding(nn.Module):
    """
        PE(pos, 2i)=sin(pos/(10000^(2*i/dim)))
        PE(pos, 2i+1)=cos(pos/(10000^(2*i/dim)))
    """
    def __init__(self, dim, max_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # max_len x 1
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1 x max_len x dim
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: B x nword x D
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)