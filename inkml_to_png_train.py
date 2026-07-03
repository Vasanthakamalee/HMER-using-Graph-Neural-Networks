

from pathlib import Path
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

# ---------- PATHS ----------
PROJECT_ROOT = Path(r"C:\\Users\\Vasantha Kamalee\\OneDrive\\Desktop\\research project")
# inkml_to_png_train.py
# Big, centered PNGs + train_list_png.txt

MERGED_ROOT  = PROJECT_ROOT / "CHROME_merged" / "training data"
TRAIN_LIST   = MERGED_ROOT / "train_list.txt"
OUT_IMG_ROOT = PROJECT_ROOT / "CHROME_merged" / "images_train"
# ---------------------------


def get_traces_data(inkml_file_abs_path: str):
    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"

    traces_all = []
    for trace_tag in root.findall(doc_namespace + 'trace'):
        text = trace_tag.text
        if text is None:
            continue
        coords = []
        for coord in text.replace('\n', '').split(','):
            parts = coord.strip().split(' ')
            if len(parts) != 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            coords.append([x, y])
        if coords:
            traces_all.append({
                'id': int(trace_tag.get('id')),
                'coords': np.array(coords, dtype=float)
            })

    if not traces_all:
        return []

    traces_all.sort(key=lambda t: t['id'])

    # global bbox
    all_pts = np.concatenate([t['coords'] for t in traces_all], axis=0)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    size = max((max_xy - min_xy).max(), 1e-6)

    # normalize to [0,1] keeping aspect ratio
    for t in traces_all:
        pts = (t['coords'] - min_xy) / size
        t['coords_norm'] = pts

    traceGroupWrapper = root.find(doc_namespace + 'traceGroup')
    traces_data = []

    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):
            label_el = traceGroup.find(doc_namespace + 'annotation')
            label = label_el.text if label_el is not None else None

            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):
                ref = int(traceView.get('traceDataRef'))
                if ref >= len(traces_all):
                    continue
                traces_curr.append(traces_all[ref]['coords_norm'])

            traces_data.append({'label': label, 'trace_group': traces_curr})
    else:
        for t in traces_all:
            traces_data.append({'trace_group': [t['coords_norm']]})

    return traces_data


def inkml2img(input_path: str, output_path: str):
    """Render one InkML file to a normalized PNG image, scaled and centered with thin strokes."""
    traces = get_traces_data(input_path)

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # scale + center: map [0,1] -> [0.1,0.9] and flip y
    scale = 0.8
    offset = 0.1

    for elem in traces:
        for subls in elem['trace_group']:
            data = np.array(subls)
            if data.size == 0:
                continue
            x = data[:, 0] * scale + offset
            y = (1.0 - data[:, 1]) * scale + offset
            ax.plot(x, y, linewidth=1, c='black')  # thinner line

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    plt.savefig(output_path, bbox_inches='tight', dpi=150, pad_inches=0.0)  # higher dpi for sharpness
    plt.close(fig)

def main():
    OUT_IMG_ROOT.mkdir(parents=True, exist_ok=True)
    lines_out = []

    with open(TRAIN_LIST, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            rel_inkml, latex = line.split("\t", 1)
            inkml_path = MERGED_ROOT / rel_inkml

            if not inkml_path.exists():
                print("MISSING:", inkml_path)
                continue

            png_rel_path = Path(rel_inkml).with_suffix(".png")
            png_full_path = OUT_IMG_ROOT / png_rel_path
            png_full_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                inkml2img(str(inkml_path), str(png_full_path))
            except Exception as e:
                print("ERROR converting", inkml_path, ":", e)
                continue

            lines_out.append(f"{png_rel_path.as_posix()}\t{latex}")
            if idx % 50 == 0:
                print("Converted", idx, "InkML files...")

    PNG_LIST = PROJECT_ROOT / "CHROME_merged" / "train_list_png.txt"
    with open(PNG_LIST, "w", encoding="utf-8") as f:
        for l in lines_out:
            f.write(l + "\n")

    print("Finished. Converted", len(lines_out), "InkML files to PNG")
    print("PNG list written to", PNG_LIST)


if __name__ == "__main__":
    main()
