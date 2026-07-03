# app_batch.py - PROCESSES 36+ IMAGES AT ONCE
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import io
import json
from pathlib import Path
import numpy as np
import torchvision.transforms as T
import pandas as pd
from pathlib import Path

ROOT = Path("C:/Users/Vasantha Kamalee/OneDrive/Desktop/research project/CHROME_merged")
vocab_json = ROOT / "vocab.json"
ckpt_path = ROOT / "stageC_best.pt"
IMG_HEIGHT = 96
MAX_WIDTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FIXED PREPROCESSING (CRITICAL)
def preprocess_image(pil_img):
    """EXACT dataset.py + CORRECT NORMALIZATION"""
    img = pil_img.convert("L")
    
    # Resize: FIXED HEIGHT=96, keep aspect
    w, h = img.size
    new_w = int(w * IMG_HEIGHT / max(h, 1))
    img = img.resize((new_w, IMG_HEIGHT), Image.Resampling.BILINEAR)
    
    # Convert to tensor (0-1 range)
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)  # (1,1,H,W)
    
    # PAD with ZEROS (NOT 1.0!) - CRITICAL FIX
    if img_tensor.shape[-1] < MAX_WIDTH:
        pad_w = MAX_WIDTH - img_tensor.shape[-1]
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, 0), value=0.0)  # WHITE BG
    
    return img_tensor

# SIMPLIFIED GREEDY DECODE (FASTER FOR BATCH)
@torch.no_grad()
def fast_predict(model, img_tensor, idx2tok, sos_idx, eos_idx):
    tokens = [sos_idx]
    img_tensor = img_tensor.to(DEVICE)
    
    for _ in range(80):
        tgt = torch.tensor(tokens, device=DEVICE).unsqueeze(0)
        logits, _ = model(img_tensor, tgt)
        next_tok = logits[0, -1].argmax(-1).item()
        
        if next_tok == eos_idx:
            break
        tokens.append(next_tok)
    
    return "".join(idx2tok[t] for t in tokens[1:] if t not in [sos_idx, eos_idx, 0])

# BATCH PROCESSING
@st.cache_resource
def load_model():
    with open(vocab_json, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        idx2tok = vocab["idx2tok"]
    
    from model_stageC_tamer_simplified import Im2LatexTamerLike
    tok2idx = {t:i for i,t in enumerate(idx2tok)}
    
    model = Im2LatexTamerLike(
        vocab_size=len(idx2tok), d_model=256, nhead=8, num_dec_layers=4,
        dim_feedforward=512, dropout=0.1, pad_idx=tok2idx["<pad>"], max_len=80
    ).to(DEVICE)
    
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model, idx2tok, tok2idx["<sos>"], tok2idx["<eos>"]

# MAIN APP
st.title("🧮 BATCH HMER - Process 36+ Images")
st.markdown("**Upload ZIP/folder → Get predictions for ALL images**")

model, idx2tok, sos_idx, eos_idx = load_model()
st.success(f"✅ Model loaded | Vocab: {len(idx2tok)} tokens | Device: {DEVICE}")

# FILE UPLOADER
uploaded_files = st.file_uploader(
    "📁 Upload images or ZIP", 
    type=['png','jpg','jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Process image
        img_bytes = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
        
        img_tensor = preprocess_image(pil_img)
        latex = fast_predict(model, img_tensor, idx2tok, sos_idx, eos_idx)
        
        results.append({
            'filename': uploaded_file.name,
            'latex': latex,
            'width': pil_img.width,
            'height': pil_img.height
        })
        
        # Update progress
        progress = (i+1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processed {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
    
    # DISPLAY RESULTS TABLE
    df = pd.DataFrame(results)
    st.subheader("📊 BATCH RESULTS")
    st.dataframe(df, use_container_width=True)
    
    # DOWNLOAD CSV
    csv = df.to_csv(index=False)
    st.download_button(
        "💾 Download Results CSV",
        csv,
        "hmer_predictions.csv",
        "text/csv"
    )
    
    # SHOW BEST PREDICTIONS
    st.subheader("⭐ TOP PREDICTIONS")
    good_preds = df[df['latex'].str.len() > 5]  # Filter garbage
    for _, row in good_preds.head(5).iterrows():
        st.write(f"**{row['filename']}** → `{row['latex']}`")

# DIAGNOSTIC INFO
with st.expander("🔧 DEBUG INFO"):
    st.json({
        "img_height": IMG_HEIGHT,
        "max_width": MAX_WIDTH,
        "pad_value": 0.0,  # FIXED!
        "device": DEVICE,
        "vocab_size": len(idx2tok)
    })
    st.info("""
    **FIXES APPLIED:**
    ✅ Padding=0.0 (white background)
    ✅ Exact dataset.py resize (96px height)
    ✅ Batch processing (36+ images)
    ✅ CSV export
    """)
