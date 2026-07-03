# model_stageB_gnn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- CNN ENCODER ----------

class CNNEncoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 96x96 -> 48x48
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 48x48 -> 24x24
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 24x24 -> 12x12
            nn.Conv2d(128, d_model, 3, padding=1), nn.ReLU(),             # 12x12
        )

    def forward(self, x):
        # x: (B,1,H,W)
        feat = self.conv(x)   # (B,D,H',W')
        return feat


# ---------- SIMPLE GRID GNN ----------

class GridGNNLayer(nn.Module):
    """
    Message passing on a 4-neighbor grid graph using learned linear weights.
    """
    def __init__(self, d_model):
        super().__init__()
        self.self_lin = nn.Linear(d_model, d_model)
        self.nei_lin  = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, feat):
        # feat: (B, D, H, W)
        B, D, H, W = feat.shape

        # neighbors: up, down, left, right with zero padding
        up    = F.pad(feat[:, :, 1:, :], (0, 0, 0, 1))     # shift down
        down  = F.pad(feat[:, :, :-1, :], (0, 0, 1, 0))    # shift up
        left  = F.pad(feat[:, :, :, 1:], (0, 1, 0, 0))     # shift right
        right = F.pad(feat[:, :, :, :-1], (1, 0, 0, 0))    # shift left

        nei = (up + down + left + right) / 4.0            # (B,D,H,W)

        x = feat.permute(0,2,3,1)   # (B,H,W,D)
        n = nei.permute(0,2,3,1)    # (B,H,W,D)

        out = self.self_lin(x) + self.nei_lin(n)
        out = F.relu(out)
        out = self.norm(out)
        out = out.permute(0,3,1,2)  # (B,D,H,W)
        return out


class GridGNNEncoder(nn.Module):
    def __init__(self, d_model=256, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([GridGNNLayer(d_model) for _ in range(num_layers)])

    def forward(self, feat):
        x = feat
        for layer in self.layers:
            x = layer(x)
        # flatten grid to sequence
        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, D)   # (B,L,D)
        return x


# ---------- POSITIONAL ENCODING ----------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]


# ---------- FULL MODEL: CNN + GNN + TRANSFORMER DECODER ----------

class Im2LatexGNNTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_dec_layers=4, dim_feedforward=512,
                 dropout=0.1, pad_idx=0, gnn_layers=2):
        super().__init__()
        self.cnn = CNNEncoder(d_model)
        self.gnn = GridGNNEncoder(d_model, num_layers=gnn_layers)

        self.pos_enc_enc = PositionalEncoding(d_model)
        self.pos_enc_dec = PositionalEncoding(d_model)

        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=1,      # light encoder over GNN output
            num_decoder_layers=num_dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1).bool()
        return mask

    def forward(self, imgs, tgt_in):
        # imgs: (B,1,H,W)
        # tgt_in: (B,T)
        B, T = tgt_in.size()

        # CNN -> grid features
        feat = self.cnn(imgs)              # (B,D,H',W')

        # GNN on grid
        mem = self.gnn(feat)               # (B,L,D)
        mem = self.pos_enc_enc(mem)        # (B,L,D)
        mem = mem.transpose(0, 1)          # (L,B,D)  = src

        # Decoder input
        tgt_emb = self.tok_embed(tgt_in)   # (B,T,D)
        tgt_emb = self.pos_enc_dec(tgt_emb).transpose(0, 1)  # (T,B,D)

        tgt_mask = self.generate_square_subsequent_mask(T).to(imgs.device)
        tgt_key_padding = (tgt_in == self.pad_idx)  # (B,T)

        out = self.transformer(
            src=mem,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding
        )   # (T,B,D)

        out = out.transpose(0, 1)          # (B,T,D)
        logits = self.fc_out(out)          # (B,T,V)
        return logits
