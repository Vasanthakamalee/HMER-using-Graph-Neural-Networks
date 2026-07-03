# model_stageA.py
import torch
import torch.nn as nn
import math

class CNNEncoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 96x96 -> 48x48
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 48x48 -> 24x24
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 24x24 -> 12x12
            nn.Conv2d(128, d_model, 3, padding=1), nn.ReLU(),             # 12x12 -> 12x12, d_model channels
        )

    def forward(self, x):
        # x: (B,1,H,W)
        feat = self.conv(x)               # (B,d_model,H',W')
        B, D, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(B, H*W, D)  # (B, L, D)
        return feat                      # encoder "memory" sequence


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]


class Im2LatexTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_layers=6, dim_feedforward=768, dropout=0.1, pad_idx=0):
        super().__init__()
        self.encoder = CNNEncoder(d_model)
        self.pos_enc_enc = PositionalEncoding(d_model)
        self.pos_enc_dec = PositionalEncoding(d_model)
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=1,       # minimal encoder layer
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1).bool()
        return mask
    

    def forward(self, imgs, tgt_in):
        B, T = tgt_in.size()

    # Encoder: CNN -> memory
        mem = self.encoder(imgs)             # (B,L,D)
        mem = self.pos_enc_enc(mem)          # (B,L,D)
        mem = mem.transpose(0, 1)            # (L,B,D) = src

    # Decoder input
        tgt_emb = self.tok_embed(tgt_in)     # (B,T,D)
        tgt_emb = self.pos_enc_dec(tgt_emb).transpose(0, 1)  # (T,B,D)

        tgt_mask = self.generate_square_subsequent_mask(T).to(imgs.device)
        tgt_key_padding = (tgt_in == self.pad_idx)  # (B,T)

    # call Transformer with src=mem, tgt=tgt_emb
        out = self.transformer(
        src=mem,
        tgt=tgt_emb,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_key_padding
    )  # (T,B,D)

        out = out.transpose(0, 1)            # (B,T,D)
        logits = self.fc_out(out)            # (B,T,V)
        return logits

