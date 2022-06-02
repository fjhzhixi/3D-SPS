import os
import sys
import torch
import torch.nn as nn
import clip

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GruLayer(nn.Module):
    def __init__(self, use_bidir=False, 
        emb_size=300, hidden_size=256, num_layers=4,
        out_dim=256):
        super().__init__()
        self.use_bidir = use_bidir
        self.num_bidir = 2 if self.use_bidir else 1
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_dim = out_dim

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.use_bidir
        )
        in_dim = hidden_size * 2 if use_bidir else hidden_size
        self.mlps = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
        )

    def forward(self, data_dict):
        """
        encode the input descriptions
        """

        word_embs = data_dict["lang_feat"]  # [B, MAX_DES_LEN, 300]
        max_des_len = word_embs.shape[1]
        # word_embs = self.word_projection(word_embs)
        lang_feat = pack_padded_sequence(word_embs, data_dict["lang_len"].cpu(), batch_first=True, enforce_sorted=False)
    
        # encode description
        lang_feat, hidden_feat = self.gru(lang_feat)  # lang_feat:[B, MAX_DES_LEN, D * hidden_size] hidden_feat:[D * num_layer, B, hidden_size]
        # [D, num_layer, B, hidden_size] choose last gru layer hidden [D, B, hidden_size]
        hidden_feat = hidden_feat.view(self.num_bidir, self.num_layers, hidden_feat.shape[1], hidden_feat.shape[2])[:, -1, :, :]     
        hidden_feat = hidden_feat.permute(1, 0, 2).contiguous().flatten(start_dim=1) # [B, D * hidden_size]

        lang_feat, _ = pad_packed_sequence(lang_feat, batch_first=True, total_length=max_des_len)
        
        lang_feat = lang_feat.transpose(-1, -2)     # [B, C, N]
        lang_feat = self.mlps(lang_feat)
        lang_feat = lang_feat.transpose(-1, -2)     # [B, N, C]

        # store the encoded language features
        data_dict["lang_emb"] = lang_feat           # [B, N, C]
        data_dict["lang_hidden"] = hidden_feat      # [B, C]

        return data_dict
    
    
class ClipModule(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device='cuda')
        self.lang_feat_projection = nn.Linear(512, output_dim)
        self.eot_feat_projection = nn.Linear(512, output_dim)

    def forward(self, data_dict):
        with torch.no_grad():
            lang_feat, eot_feat = self.model.encode_text(data_dict["lang_feat"], return_unprojected=True)

        lang_feat, eot_feat = lang_feat.float(), eot_feat.float()

        lang_feat = self.lang_feat_projection(lang_feat)
        eot_feat = self.eot_feat_projection(eot_feat)

        data_dict["lang_emb"] = lang_feat
        data_dict["lang_hidden"] = eot_feat  # [B, C]
        return data_dict