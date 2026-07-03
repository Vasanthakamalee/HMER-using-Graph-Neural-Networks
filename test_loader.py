# test_loader.py
from pathlib import Path
import json
from torch.utils.data import DataLoader
from dataset import HMERDataset, collate_fn

ROOT = Path(r"C:\\Users\\Vasantha Kamalee\\OneDrive\\Desktop\\research project\\CHROME_merged")
print((ROOT / "images_train/2012_training/TrainData2_15_sub_17.png").exists())

train_list = ROOT / "train_list_png.txt"
val_list = ROOT / "val_list_png.txt"
vocab_json = ROOT / "vocab.json"

train_ds = HMERDataset(train_list, ROOT, vocab_json)
val_ds = HMERDataset(val_list, ROOT, vocab_json)

with open(vocab_json, 'r', encoding='utf-8') as f:
    idx2tok = json.load(f)["idx2tok"]
pad_idx = idx2tok.index('<pad>')

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, pad_idx))

for imgs, tgts in train_loader:
    print("Images batch:", imgs.shape)  # (B,1,H,W)
    print("Targets batch:", tgts.shape) # (B,T)
    break
