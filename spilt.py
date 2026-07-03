# split_train_val_png.py
from pathlib import Path
import random

ROOT = Path(r"C:\\Users\\Vasantha Kamalee\\OneDrive\\Desktop\\research project\\CHROME_merged")
src = ROOT / "train_list_png.txt"
train_out = ROOT / "train_list_png.txt"   # overwrite as train
val_out = ROOT / "val_list_png.txt"

with open(src, "r", encoding="utf-8") as f:
    lines = [l for l in f.readlines() if l.strip()]

random.shuffle(lines)

val_ratio = 0.1  # 10% for validation
n_val = int(val_ratio * len(lines))
val_lines = lines[:n_val]
train_lines = lines[n_val:]

with open(train_out, "w", encoding="utf-8") as f:
    f.writelines(train_lines)
with open(val_out, "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print("Train:", len(train_lines), "Val:", len(val_lines))
