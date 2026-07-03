# train_stageA.py
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import HMERDataset, collate_fn
from model_stageB_gnn import Im2LatexGNNTransformer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(r"C:\\Users\\Vasantha Kamalee\\OneDrive\\Desktop\\research project\\CHROME_merged")
train_list = ROOT / "train_list_png.txt"
val_list = ROOT / "val_list_png.txt"
vocab_json = ROOT / "vocab.json"

# Dataset & loaders
train_ds = HMERDataset(train_list, ROOT, vocab_json)
val_ds = HMERDataset(val_list, ROOT, vocab_json)

with open(vocab_json, 'r', encoding='utf-8') as f:
    idx2tok = json.load(f)["idx2tok"]
pad_idx = idx2tok.index('<pad>')
sos_idx = idx2tok.index('<sos>')
eos_idx = idx2tok.index('<eos>')
vocab_size = len(idx2tok)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, pad_idx))
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, pad_idx))

# Model
model = Im2LatexGNNTransformer(
    vocab_size=vocab_size,
    d_model=256,
    nhead=8,
    num_dec_layers=4,
    dim_feedforward=512,
    pad_idx=pad_idx,
    gnn_layers=2
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_one_epoch():
    model.train()
    total_loss = 0.0
    for imgs, tgt in train_loader:
        imgs = imgs.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # teacher forcing: input is tokens[:-1], target is tokens[1:]
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        logits = model(imgs, tgt_in)          # (B,T-1,V)
        loss = criterion(
            logits.reshape(-1, vocab_size),
            tgt_out.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def eval_one_epoch():
    model.eval()
    total_loss = 0.0
    for imgs, tgt in val_loader:
        imgs = imgs.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits = model(imgs, tgt_in)
        loss = criterion(
            logits.reshape(-1, vocab_size),
            tgt_out.reshape(-1)
        )
        total_loss += loss.item()
    return total_loss / len(val_loader)

# train_stageA.py (replace the main block at the end)

if __name__ == "__main__":
    max_epochs = 50
    best_val = float("inf")
    patience = 5
    no_improve = 0

    best_path = ROOT / "stageB_best.pt"

    for ep in range(1, max_epochs + 1):
        train_loss = train_one_epoch()
        val_loss = eval_one_epoch()
        print(f"Epoch {ep}: train loss {train_loss:.3f}, val loss {val_loss:.3f}")

        if val_loss < best_val - 1e-3:   # require small improvement
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print("  -> New best model saved.")
        else:
            no_improve += 1
            print(f"  -> No improvement ({no_improve}/{patience})")

            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    print("Best val loss:", best_val)
    print("Best model path:", best_path)
