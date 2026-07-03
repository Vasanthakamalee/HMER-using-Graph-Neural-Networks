# model_stageC_tamer_simplified.py
import math
import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, d_model, 3, padding=1), nn.ReLU(),
        )

    def forward(self, x):
        # x: (B,1,H,W)
        return self.conv(x)   # (B,D,H',W')


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1,max_len,d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B,L,D)
        L = x.size(1)
        return x + self.pe[:, :L, :]


class Im2LatexTamerLike(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_dec_layers=4, dim_feedforward=512,
                 dropout=0.1, pad_idx=0, max_len=80):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.vocab_size = vocab_size

        # CNN encoder
        self.cnn = CNNEncoder(d_model)

        # memory positional encoding
        self.pos_enc_mem = PositionalEncoding(d_model)

        # token embedding
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # Transformer decoder (batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_dec_layers)

        # heads
        self.fc_out = nn.Linear(d_model, vocab_size)   # token head
        self.tree_head = nn.Linear(d_model, max_len)   # parent-index head

    def generate_square_subsequent_mask(self, sz, device):
        # (T,T) mask with 0 for allowed, -inf for future
        return torch.triu(
            torch.full((sz, sz), float("-inf"), device=device),
            diagonal=1
        )

    def forward(self, imgs, tgt_in):
        """
        imgs:   (B,1,H,W)
        tgt_in: (B,T) token indices (teacher forcing)
        """
        # ----- encoder -----
        feat = self.cnn(imgs)                     # (B,D,H',W')
        Bf, D, H, W = feat.shape
        mem = feat.permute(0, 2, 3, 1).reshape(Bf, H * W, D)  # (B,L,D)
        mem = self.pos_enc_mem(mem)

        # ----- decoder input -----
        tgt_emb = self.tok_embed(tgt_in) * math.sqrt(self.d_model)  # (B,T,D)
        T = tgt_emb.size(1)
        tgt_mask = self.generate_square_subsequent_mask(T, tgt_emb.device)

        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=mem,
            tgt_mask=tgt_mask,
        )                                           # (B,T,D)

        # ----- heads -----
        token_logits = self.fc_out(dec_out)         # (B,T,V)
        tree_logits  = self.tree_head(dec_out)      # (B,T,max_len)

        return token_logits, tree_logits
