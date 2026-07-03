# vocab.py
import re
import json
from pathlib import Path

SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '<unk>']

def tokenize_latex(s: str):
    """
    Simple LaTeX tokenizer:
    - commands like \frac, \log
    - braces {} and _ ^
    - numbers, single letters, other symbols
    """
    s = s.strip()
    tokens = re.findall(r'\\[A-Za-z]+|[{}_^]|[0-9]+|[A-Za-z]|[^ \t]', s)
    return tokens

def build_vocab(list_paths):
    token_set = set()
    for path in list_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                _, latex = line.rstrip('\n').split('\t', 1)
                token_set.update(tokenize_latex(latex))
    idx2tok = SPECIAL_TOKENS + sorted(token_set)
    tok2idx = {t: i for i, t in enumerate(idx2tok)}
    return tok2idx, idx2tok

if __name__ == "__main__":
    ROOT = Path(r"C:\\Users\\Vasantha Kamalee\\OneDrive\\Desktop\\research project\\CHROME_merged")
    train_list = ROOT / "train_list_png.txt"
    val_list = ROOT / "val_list_png.txt"

    tok2idx, idx2tok = build_vocab([train_list, val_list])

    out_path = ROOT / "vocab.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"idx2tok": idx2tok}, f, ensure_ascii=False, indent=2)

    print("Vocab size:", len(idx2tok))
    print("Saved to:", out_path)
