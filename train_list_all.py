# build_train_list_all.py

import os
from pathlib import Path
import sys

# --- CHANGE THESE TO YOUR REAL PATHS ---
ROOT = Path(r"C:\\Users\\Vasantha Kamalee\\OneDrive\\Desktop\\research project\\CHROME_merged\\training data")
INKML2LTX_DIR = Path(r"C:\\Users\\Vasantha Kamalee\\OneDrive\\Desktop\\research project")  # folder with inkml2ltx.py
# ---------------------------------------

sys.path.append(str(INKML2LTX_DIR))
from inkml2ltx import inkml2tag   # the function you tested

train_dirs = [
    "2011_training",
    "2012_training",
    "2014_training",
]

out_path = ROOT / "train_list.txt"
lines = []

for d in train_dirs:
    dir_full = ROOT / d
    for fname in os.listdir(dir_full):
        if not fname.lower().endswith(".inkml"):
            continue
        inkml_full = dir_full / fname

        try:
            latex = inkml2tag(str(inkml_full)).strip()
        except Exception as e:
            print("ERROR on", inkml_full, ":", e)
            continue

        if not latex:
            print("NO LATEX for", inkml_full)
            continue

        rel_path = Path(d) / fname
        lines.append(f"{rel_path.as_posix()}\t{latex}")

print("Total expressions:", len(lines))

with open(out_path, "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print("Wrote", out_path)
