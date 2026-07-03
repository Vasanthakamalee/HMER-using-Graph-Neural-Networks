from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from dataset import HMERDataset, collate_fn
from model_stageB_gnn import Im2LatexGNNTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(r"C:\\Users\\Vasantha Kamalee\\OneDrive\\Desktop\\research project\\CHROME_merged")
val_list = ROOT / "val_list_png.txt"
vocab_json = ROOT / "vocab.json"
best_ckpt = ROOT / "stageB_best.pt"  # from early-stopping training

# load vocab
with open(vocab_json, 'r', encoding='utf-8') as f:
    idx2tok = json.load(f)["idx2tok"]
tok2idx = {t: i for i, t in enumerate(idx2tok)}
pad_idx = tok2idx['<pad>']
sos_idx = tok2idx['<sos>']
eos_idx = tok2idx['<eos>']
vocab_size = len(idx2tok)

# dataset/loader
val_ds = HMERDataset(val_list, ROOT, vocab_json)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, pad_idx))

# model
model = Im2LatexGNNTransformer(
    vocab_size=vocab_size,
    d_model=256,
    nhead=8,
    num_dec_layers=4,
    dim_feedforward=512,
    dropout=0.1,
    pad_idx=pad_idx,
    gnn_layers=2
).to(DEVICE)

model.load_state_dict(torch.load(ROOT / "stageB_best.pt", map_location=DEVICE))
model.eval()

@torch.no_grad()
def greedy_decode(img, max_len=80):
    # img: (1,1,H,W)
    tgt_tokens = [sos_idx]
    for _ in range(max_len):
        tgt = torch.tensor(tgt_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1,T)
        logits = model(img, tgt)          # (1,T,V)
        next_token = logits[0, -1].argmax(dim=-1).item()
        if next_token == eos_idx:
            break
        tgt_tokens.append(next_token)
    return tgt_tokens[1:]  # drop <sos>

def tokens_to_string(tokens):
    return ''.join(idx2tok[t] for t in tokens if t not in (sos_idx, eos_idx, pad_idx))

if __name__ == "__main__":
    expr_total = 0
    expr_correct = 0
    tok_total = 0
    tok_correct = 0

    for imgs, tgt in val_loader:
        imgs = imgs.to(DEVICE)           # (1,1,H,W)
        tgt = tgt.to(DEVICE)[0]          # (T,)
        # ground truth tokens without padding
        gt_tokens = [t.item() for t in tgt if t.item() not in (pad_idx,)]
        # remove leading <sos> and trailing <eos> if present
        if gt_tokens and gt_tokens[0] == sos_idx:
            gt_tokens = gt_tokens[1:]
        if gt_tokens and gt_tokens[-1] == eos_idx:
            gt_tokens = gt_tokens[:-1]

        pred_tokens = greedy_decode(imgs)

        # token accuracy
        L = max(len(gt_tokens), len(pred_tokens))
        for i in range(L):
            gt_t = gt_tokens[i] if i < len(gt_tokens) else pad_idx
            pr_t = pred_tokens[i] if i < len(pred_tokens) else pad_idx
            if gt_t == pr_t:
                tok_correct += 1
            tok_total += 1

        # expression accuracy
        if gt_tokens == pred_tokens:
            expr_correct += 1
        expr_total += 1

    expr_acc = expr_correct / expr_total * 100
    tok_acc = tok_correct / tok_total * 100
    print(f"Expression accuracy: {expr_acc:.2f}%")
    print(f"Token accuracy: {tok_acc:.2f}%")
