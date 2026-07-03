# eval_stageC.py
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader

from dataset import HMERDataset, collate_fn
from model_stageC_tamer_simplified import Im2LatexTamerLike

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# -------- paths ----------
ROOT = Path("C:/Users/Vasantha Kamalee/OneDrive/Desktop/research project/CHROME_merged")
val_list   = ROOT / "val_list_png.txt"
vocab_json = ROOT / "vocab.json"
best_ckpt  = ROOT / "stageC_best.pt"

print("ROOT:", ROOT)
print("val_list exists:", val_list.exists())
print("vocab_json exists:", vocab_json.exists())
print("ckpt exists:", best_ckpt.exists())

# -------- vocab ----------
with open(vocab_json, "r", encoding="utf-8") as f:
    vocab = json.load(f)
    idx2tok = vocab["idx2tok"]

tok2idx = {t: i for i, t in enumerate(idx2tok)}
pad_idx = tok2idx["<pad>"]
sos_idx = tok2idx["<sos>"]
eos_idx = tok2idx["<eos>"]
vocab_size = len(idx2tok)
print(f"Vocab size: {vocab_size}")

# -------- data -----------
val_ds = HMERDataset(val_list, ROOT, vocab_json)

def collate_with_pad(batch):
    return collate_fn(batch, pad_idx)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_with_pad,
    num_workers=0,
    pin_memory=False,
)
print(f"Val batches: {len(val_loader)}")

# -------- model ----------
MAX_LEN = 80
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

model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
model.eval()
print(f"Loaded checkpoint: {best_ckpt}")

# -------- decoding -------
@torch.no_grad()
def greedy_decode(img, max_len=80):
    """
    img: (1,1,H,W)
    returns: list[int] predicted token ids (without <sos>/<eos>)
    """
    tgt_tokens = [sos_idx]
    for _ in range(max_len):
        tgt = torch.tensor(tgt_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1,T)
        token_logits, _ = model(img, tgt)
        next_token = token_logits[0, -1].argmax(dim=-1).item()
        if next_token == eos_idx:
            break
        tgt_tokens.append(next_token)
    return tgt_tokens[1:]  # drop <sos>

def tokens_to_string(tokens):
    return "".join(
        idx2tok[t] for t in tokens
        if t not in (sos_idx, eos_idx, pad_idx)
    )

# -------- evaluation -----
if __name__ == "__main__":
    expr_total = 0
    expr_correct = 0
    tok_total = 0
    tok_correct = 0

    examples = []

    for batch_idx, (imgs, tgt) in enumerate(val_loader):
        imgs = imgs.to(DEVICE)       # (1,1,H,W)
        tgt  = tgt.to(DEVICE)[0]     # (T,)

        # ground-truth tokens (remove pad/sos/eos)
        gt_tokens = [t.item() for t in tgt if t.item() != pad_idx]
        if gt_tokens and gt_tokens[0] == sos_idx:
            gt_tokens = gt_tokens[1:]
        if gt_tokens and gt_tokens[-1] == eos_idx:
            gt_tokens = gt_tokens[:-1]

        pred_tokens = greedy_decode(imgs)

        # token accuracy (position-wise)
        L = max(len(gt_tokens), len(pred_tokens))
        for i in range(L):
            gt_t = gt_tokens[i] if i < len(gt_tokens) else pad_idx
            pr_t = pred_tokens[i] if i < len(pred_tokens) else pad_idx
            if gt_t == pr_t:
                tok_correct += 1
            tok_total += 1

        # expression accuracy (exact match)
        if gt_tokens == pred_tokens:
            expr_correct += 1
        expr_total += 1

        # store a few examples
        if len(examples) < 10:
            examples.append({
                "gt": tokens_to_string(gt_tokens),
                "pred": tokens_to_string(pred_tokens),
                "match": gt_tokens == pred_tokens,
            })

    expr_acc = expr_correct / expr_total * 100
    tok_acc  = tok_correct / tok_total * 100

    print("\n===============================")
    print("STAGE-C EVALUATION RESULTS")
    print("===============================")
    print(f"Expression accuracy: {expr_acc:.2f}% ({expr_correct}/{expr_total})")
    print(f"Token accuracy:      {tok_acc:.2f}% ({tok_correct}/{tok_total})")

    print("\nSample predictions:")
    for i, ex in enumerate(examples, 1):
        mark = "OK" if ex["match"] else "XX"
        print(f"{i}. [{mark}] GT: {ex['gt']}  |  PRED: {ex['pred']}")
