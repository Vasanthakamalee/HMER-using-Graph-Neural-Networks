# dataset.py
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from vocab import tokenize_latex, SPECIAL_TOKENS

class HMERDataset(Dataset):
    def __init__(self, list_path, root_dir, vocab_json, img_height=96):
        self.root_dir = Path(root_dir)
        self.samples = []
        with open(list_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                img_rel, latex = line.rstrip('\n').split('\t', 1)
                self.samples.append((img_rel, latex))

        with open(vocab_json, 'r', encoding='utf-8') as f:
            idx2tok = json.load(f)["idx2tok"]
        self.idx2tok = idx2tok
        self.tok2idx = {t: i for i, t in enumerate(idx2tok)}

        self.pad_idx = self.tok2idx['<pad>']
        self.sos_idx = self.tok2idx['<sos>']
        self.eos_idx = self.tok2idx['<eos>']

        self.img_height = img_height
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            T.ToTensor(),
            ])

    def encode_latex(self, latex):
        toks = tokenize_latex(latex)
        ids = [self.sos_idx] + [self.tok2idx.get(t, self.tok2idx['<unk>']) for t in toks] + [self.eos_idx]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel, latex = self.samples[idx]
        img_path = (self.root_dir / "images_train" / img_rel).resolve()
        img = Image.open(img_path).convert("L")

        # resize to fixed height, keep aspect
        w, h = img.size
        new_h = self.img_height
        new_w = int(w * new_h / h)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        img = self.transform(img)  # (1, H, W)
        tgt = self.encode_latex(latex)
        return img, tgt

def collate_fn(batch, pad_idx):
    imgs, tgts = zip(*batch)

    # pad images to same width in batch
    max_w = max(img.shape[-1] for img in imgs)
    padded_imgs = []
    for img in imgs:
        c, h, w = img.shape
        pad_w = max_w - w
        if pad_w > 0:
            pad = torch.zeros(c, h, pad_w)
            img = torch.cat([img, pad], dim=-1)
        padded_imgs.append(img)
    imgs = torch.stack(padded_imgs, dim=0)  # (B,1,H,W)

    # pad target sequences
    max_len = max(t.size(0) for t in tgts)
    tgt_batch = torch.full((len(tgts), max_len), pad_idx, dtype=torch.long)
    for i, t in enumerate(tgts):
        tgt_batch[i, :t.size(0)] = t

    return imgs, tgt_batch
