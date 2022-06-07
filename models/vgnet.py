import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.backbone_module import Pointnet2Backbone
from models.modules import PredictHead, ClsAgnosticPredictHead
from models.lang_module import GruLayer, ClipModule
from models.sample_model import SamplingModule
from models.transformer import *
from models.mlp import MLP

class VGNet(nn.Module):

    def __init__(self, input_feature_dim, args, data_config):
        super().__init__()
        self.args = args
        self.num_class = data_config.num_class
        self.num_heading_bin = data_config.num_heading_bin
        self.num_size_cluster = data_config.num_size_cluster
        self.mean_size_arr = data_config.mean_size_arr
        assert(self.mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim,
                                              output_feature_dim=self.args.point_feat_dim)
        
        # Lang Feat Encoding
        if args.lang_emb_type == 'glove':
            self.lang_module = GruLayer(use_bidir=self.args.use_bidir, emb_size=self.args.embedding_size,
                                        hidden_size=self.args.gru_hidden_size, num_layers=self.args.gru_num_layer,
                                        out_dim=self.args.transformer_feat_dim)
            self.lang_feat_dim = self.args.gru_hidden_size * 2 if self.args.use_bidir else self.args.gru_hidden_size
        elif args.lang_emb_type == 'clip':
            self.lang_module = ClipModule(output_dim=self.args.transformer_feat_dim)
            self.lang_feat_dim = self.args.transformer_feat_dim
        else:
            raise NotImplementedError('Language embedding type unsupported!')
        
        # Object candidate sampling
        self.sampling_module = SamplingModule(
            sampling_method = args.sampling,
            num_proposal = args.num_proposal,
            feat_dim=self.args.point_feat_dim,
            lang_dim=self.lang_feat_dim,
            args = self.args
        )

        # Proposal 
        if self.args.size_cls_agnostic:
            self.proposal_head = ClsAgnosticPredictHead(
                self.num_class,
                self.num_heading_bin,
                args.num_proposal,
                self.args.point_feat_dim)
        else:
            self.proposal_head = PredictHead(
                self.num_class, 
                self.num_heading_bin, 
                self.num_size_cluster,
                self.mean_size_arr, 
                args.num_proposal, 
                self.args.point_feat_dim,
                self.args.transformer_feat_dim,
                use_ref_branch=False,
                use_cls_branch=False,
                args=self.args)

        # Transformer Decoder Projection
        self.object_proj_layer = nn.Conv1d(self.args.point_feat_dim, self.args.transformer_feat_dim - self.args.vis_feat_dim
                                           if (args.use_multiview and args.fuse_multi_mode == 'late') else self.args.transformer_feat_dim, kernel_size=1)
        self.point_proj_layer = nn.Conv1d(self.args.point_feat_dim, self.args.transformer_feat_dim, kernel_size=1)

        # obj feat & lang feat cross attn
        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for i in range(self.args.num_decoder_layers):
            decode_layer = TransformerFilter(d_model=self.args.transformer_feat_dim, dim_feedforward=self.args.ffn_dim, 
                                     nhead=self.args.n_head, dropout=self.args.transformer_dropout,
                                     object_position_embedding=self.args.object_position_embedding,
                                     point_position_embedding=self.args.point_position_embedding,
                                     lang_position_embedding=self.args.lang_position_embedding,
                                     ret_att=True if (self.args.use_att_score and i in self.args.ref_filter_steps) else False)
            self.decoder.append(decode_layer)

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.args.num_decoder_layers):
            use_ref_branch = False
            if self.args.ref_each_stage:
                use_ref_branch = True
            else:
                if i == self.args.num_decoder_layers - 1:
                    use_ref_branch = True
            use_cls_branch = False
            if self.args.cls_each_stage:
                use_cls_branch = True
            else:
                if i == self.args.num_decoder_layers - 1:
                    use_cls_branch = True
            if args.size_cls_agnostic:
                self.prediction_heads.append(ClsAgnosticPredictHead(self.num_class, 
                    self.num_heading_bin, 
                    args.num_proposal, 
                    seed_feat_dim=self.args.transformer_feat_dim))
            else:
                self.prediction_heads.append(PredictHead(self.num_class, 
                    self.num_heading_bin, 
                    self.num_size_cluster,
                    self.mean_size_arr, 
                    args.num_proposal, 
                    seed_feat_dim=self.args.transformer_feat_dim,
                    lang_feat_dim=self.args.transformer_feat_dim,
                    use_ref_branch=use_ref_branch,
                    use_cls_branch=use_cls_branch,
                    use_ref_mask=self.args.use_ref_mask,
                    use_objectness=self.args.use_objectness,
                    args=self.args))
        
        # Init
        self.init_weights()
        self.init_bn_momentum()
        if args.distribute:
            nn.SyncBatchNorm.convert_sync_batchnorm(self)

        if args.use_pretrained:
            print("loading pretrained Model...")
            pretrained_path = args.pretrain_path
            if args.trans_pre_model:
                self.load_state_dict({k.replace('module.',''):v for k,v in torch.load(pretrained_path, map_location=torch.device('cpu')).items()}, strict=False)
            else:
                self.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')), strict=False)

    def forward(self, data_dict):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        # --------- POINTCLOUD FEATURE ---------
        data_dict = self.backbone_net(data_dict)
        points_xyz = data_dict['fp2_xyz']                   # [B, n_point, 3]
        points_features = data_dict['fp2_features']         # [B, c, n_point]
        data_dict['seed_inds'] = data_dict['fp2_inds']      # [B, n_point]
        data_dict['seed_xyz'] = points_xyz                  # [B, n_point, 3]
        data_dict['seed_features'] = points_features        # [B, c, n_point]

        # lang module
        data_dict = self.lang_module(data_dict)
        lang_feat = data_dict['lang_emb']                   # [B, M, C]
        lang_mask = data_dict['lang_mask']                  # [B, M]
        # --------- SAMPLING ---------
        data_dict, xyz, features = self.sampling_module(points_xyz, points_features, data_dict)
        # features: object candidate [B, c, n_proposal]
        # point_obj_cls_logits: [B, 1, n_point]

        # --------- PROPOSAL ---------
        proposal_center, proposal_size = self.proposal_head(features,
                                                            base_xyz=xyz,
                                                            end_points=data_dict,
                                                            prefix='proposal_',
                                                            lang_feat=lang_feat,
                                                            lang_mask=lang_mask)  # N num_proposal 3
        base_xyz = proposal_center.detach().clone()     # [B, n_proposal, 3]
        base_size = proposal_size.detach().clone()      # [B, n_proposal, 3]

        # Transformer Decoder and Prediction
        object_feat = self.object_proj_layer(features).transpose(1,2)                 # [B, n_proposal, C]
        point_feat = self.point_proj_layer(points_features).transpose(1,2)            # [B, n_point, C]

        if self.args.use_multiview and self.args.fuse_multi_mode == 'late':
            point_multiview = data_dict['multiview'].gather(dim=1, index=data_dict['seed_inds'].unsqueeze(-1)
                                                            .repeat(1, 1, data_dict['multiview'].shape[2]).long())
            obj_multiview = point_multiview.gather(dim=1, index=data_dict['query_points_sample_inds'].unsqueeze(-1)
                                                   .repeat(1, 1, point_multiview.shape[2]).long())
            obj_multiview = obj_multiview.gather(dim=1, index=data_dict['ref_query_points_sample_inds'].unsqueeze(-1)
                                                 .repeat(1, 1, obj_multiview.shape[2]).long())
            object_feat = torch.cat((object_feat, obj_multiview), dim=-1)

        # Position Embedding for point
        if self.args.point_position_embedding == 'none':
            point_pos = None
        elif self.args.point_position_embedding in ['xyz_learned']:
            point_pos = points_xyz
        
        # Position Embedding for lang
        if self.args.lang_position_embedding == 'none':
            lang_pos = None
        
        input_object_feat = object_feat
        input_lang_feat = lang_feat
        for i in range(self.args.num_decoder_layers):
            prefix = 'last_' if (i == self.args.num_decoder_layers - 1) else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.args.object_position_embedding == 'none':
                object_pos = None
            elif self.args.object_position_embedding == 'xyz_learned':
                object_pos = base_xyz
            elif self.args.object_position_embedding == 'loc_learned':
                object_pos = torch.cat([base_xyz, base_size], -1)        # [B, n_proposal, 6]
            else:
                raise NotImplementedError(f"object_position_embedding not supported {self.object_position_embedding}")
            
            object_mask = torch.ones((object_feat.shape[0], object_feat.shape[1])).cuda()
            # Transformer Decoder Layer
            # objec: [B, n_proposal, C] pos: [B, n_proposal, 6]
            # point: [B, n_point, C] pos: [B, n_point, 3]
            # lang_feat: [B, n_word, C]

            cross_object_feat, cross_lang_feat, object_feat, lang_feat, cross_object_att, cross_lang_att = self.decoder[i](
                input_object_feat, object_pos, object_mask, point_feat, point_pos, input_lang_feat, lang_pos, lang_mask)

            object_feat_pre = object_feat.permute(0, 2, 1)      # [B, C, K]
            cross_object_feat_pre = cross_object_feat.permute(0, 2, 1)      # [B, C, K]
            # Prediction
            # base_xyz is updated for each stage or not?
            base_xyz, base_size = self.prediction_heads[i](object_feat_pre,
                                                        base_xyz=xyz,
                                                        end_points=data_dict,
                                                        prefix=prefix,
                                                        lang_feat=lang_feat,
                                                        lang_mask=lang_mask,
                                                        cross_object_feat=cross_object_feat_pre,
                                                        cross_lang_feat=cross_lang_feat,
                                                        prefix_index=i)
            
            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()
            if self.args.use_att_score and i in self.args.ref_filter_steps:
                filter_scores = cross_lang_att.mean(dim=1)           # [B, K]
                select_num = int(filter_scores.shape[1] * self.args.ref_mask_scale)
                select_index = torch.topk(filter_scores, select_num, dim=1)[1].long()
                data_dict[f'{prefix}ref_mask_inds'] = select_index
                object_feat = torch.gather(object_feat, 1, select_index.unsqueeze(-1).repeat(1,1,object_feat.shape[-1]))
                cross_object_feat = torch.gather(cross_object_feat, 1, select_index.unsqueeze(-1).repeat(1,1,cross_object_feat.shape[-1]))
                base_xyz = torch.gather(base_xyz, 1, select_index.unsqueeze(-1).repeat(1,1,base_xyz.shape[-1]))
                base_size = torch.gather(base_size, 1, select_index.unsqueeze(-1).repeat(1,1,base_size.shape[-1]))
                xyz = torch.gather(xyz, 1, select_index.unsqueeze(-1).repeat(1,1,base_size.shape[-1]))
            
            if self.args.transformer_mode == 'serial':
                input_object_feat = cross_object_feat
                input_lang_feat = cross_lang_feat
            else:
                input_object_feat = object_feat
                input_lang_feat = lang_feat

        return data_dict

    def init_weights(self):
            # initialize transformer
            for m in self.decoder.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.args.bn_momentum_init
