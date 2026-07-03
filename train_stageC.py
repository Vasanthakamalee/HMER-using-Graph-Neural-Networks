# train_stageC.py
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HMERDataset, collate_fn
from model_stageC_tamer_simplified import Im2LatexTamerLike

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

ROOT = Path(r"C:\\Users\\Vasantha Kamalee\\OneDrive\\Desktop\\research project\\CHROME_merged")

train_list = ROOT / "train_list_png.txt"
val_list   = ROOT / "val_list_png.txt"
vocab_json = ROOT / "vocab.json"

print("ROOT:", ROOT)
print("train_list exists:", train_list.exists())
print("val_list exists:", val_list.exists())
print("vocab_json exists:", vocab_json.exists())

# ---------------- vocab ----------------
with open(vocab_json, "r", encoding="utf-8") as f:
    vocab = json.load(f)
    idx2tok = vocab["idx2tok"]

tok2idx = {t: i for i, t in enumerate(idx2tok)}
pad_idx = tok2idx["<pad>"]
sos_idx = tok2idx["<sos>"]
eos_idx = tok2idx["<eos>"]
vocab_size = len(idx2tok)
print(f"Vocab size: {vocab_size}")

# ---------------- data -----------------
train_ds = HMERDataset(train_list, ROOT, vocab_json)
val_ds   = HMERDataset(val_list, ROOT, vocab_json)

def collate_with_pad(batch):
    return collate_fn(batch, pad_idx)

train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_with_pad,
    num_workers=0,
    pin_memory=False,
)
val_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_with_pad,
    num_workers=0,
    pin_memory=False,
)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ---------------- model ----------------
MAX_LEN = 80           # parent index classes: 0..79
model = Im2LatexTamerLike(
    vocab_size=vocab_size,
    d_model=256,
    nhead=8,
    num_dec_layers=4,
    dim_feedforward=512,
    dropout=0.1,
    pad_idx=pad_idx,
    max_len=MAX_LEN,
).to(DEVICE)

token_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
tree_criterion  = nn.CrossEntropyLoss()  # for parent indices 0..MAX_LEN-1
lambda_tree = 0.1

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

# ------------- helpers -----------------
def build_parent_labels(tgt_in, max_len):
    """
    Simple fake tree: each token's parent is itself (clamped into 0..max_len-1).
    Guarantees all labels are < max_len, so CrossEntropyLoss is safe.
    """
    B, T = tgt_in.size()
    # parent index = current position index (0..T-1), but capped
    parent = torch.arange(T, device=tgt_in.device).unsqueeze(0).expand(B, -1)
    parent = torch.clamp(parent, max=max_len - 1)   # ensures < max_len
    return parent                                   # (B,T), int64

def train_one_epoch():
    model.train()
    total_loss = total_token_loss = total_tree_loss = 0.0

    for batch_idx, (imgs, tgt) in enumerate(train_loader):
        imgs = imgs.to(DEVICE)
        tgt  = tgt.to(DEVICE)

        tgt_in  = tgt[:, :-1]   # (B,T)
        tgt_out = tgt[:, 1:]    # (B,T)

        token_logits, tree_logits = model(imgs, tgt_in)

        # token loss
        token_loss = token_criterion(
            token_logits.reshape(-1, vocab_size),
            tgt_out.reshape(-1),
        )

        # tree loss with safe parent labels
        parent_labels = build_parent_labels(tgt_in, MAX_LEN)   # (B,T)
        B, T, M = tree_logits.shape
        tree_loss = tree_criterion(
            tree_logits.reshape(-1, M),
            parent_labels.reshape(-1),
        )

        loss = token_loss + lambda_tree * tree_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_token_loss += token_loss.item()
        total_tree_loss += tree_loss.item()

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: loss={loss.item():.3f}")

    n = len(train_loader)
    return total_loss / n, total_token_loss / n, total_tree_loss / n

@torch.no_grad()
def eval_one_epoch():
    model.eval()
    total_loss = 0.0

    for imgs, tgt in val_loader:
        imgs = imgs.to(DEVICE)
        tgt  = tgt.to(DEVICE)

        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        token_logits, tree_logits = model(imgs, tgt_in)

        token_loss = token_criterion(
            token_logits.reshape(-1, vocab_size),
            tgt_out.reshape(-1),
        )

        parent_labels = build_parent_labels(tgt_in, MAX_LEN)
        B, T, M = tree_logits.shape
        tree_loss = tree_criterion(
            tree_logits.reshape(-1, M),
            parent_labels.reshape(-1),
        )

        loss = token_loss + lambda_tree * tree_loss
        total_loss += loss.item()

    return total_loss / len(val_loader)

# ------------- training loop -----------
if __name__ == "__main__":
    max_epochs = 50
    best_val = float("inf")
    patience = 7
    no_improve = 0
    best_path = ROOT / "stageC_best.pt"

    for ep in range(1, max_epochs + 1):
        print(f"\nEpoch {ep}/{max_epochs}")
        train_loss, token_loss, tree_loss = train_one_epoch()
        val_loss = eval_one_epoch()

        print(f"  Train: total={train_loss:.3f} (token={token_loss:.3f}, tree={tree_loss:.3f})")
        print(f"  Val:   {val_loss:.3f}")

        scheduler.step(val_loss)

        if val_loss < best_val - 1e-3:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved: {best_path}")
        else:
            no_improve += 1
            print(f"  -> No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print("Early stopping.")
                break

    print(f"\nBest val loss: {best_val:.3f}")
    print(f"Best model: {best_path}")
