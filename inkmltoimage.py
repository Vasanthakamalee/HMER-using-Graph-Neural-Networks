import os
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re

def parse_crome_traces(inkml_path):
    """Parse CROHME trace elements (NOT stroke elements)"""
    try:
        tree = ET.parse(inkml_path)
        root = tree.getroot()
        
        traces = []
        inkml_ns = '{http://www.w3.org/2003/InkML}'
        
        # Parse ALL trace elements (CROHME format)
        for trace_elem in root.findall('.//' + inkml_ns + 'trace'):
            trace_data = trace_elem.text
            if trace_data:
                # Parse trace data: x1 y1 x2 y2 ...
                coords = list(map(float, re.split(r'\s+|,', trace_data.strip())))
                if len(coords) % 2 == 0 and len(coords) >= 4:  # At least 2 points
                    x_coords = coords[0::2]
                    y_coords = coords[1::2]
                    traces.append((x_coords, y_coords))
        
        print(f"✅ Found {len(traces)} traces in {os.path.basename(inkml_path)}")
        return traces
        
    except Exception as e:
        print(f"❌ Parse error {os.path.basename(inkml_path)}: {e}")
        return []

def inkml_to_image(inkml_path, output_dir, width=800, height=600, stroke_width=2):
    """Convert CROHME InkML (trace-based) to image"""
    traces = parse_crome_traces(inkml_path)
    
    if not traces:
        return None
    
    # Collect all coordinates for normalization
    all_x = [x for trace in traces for x in trace[0]]
    all_y = [y for trace in traces for y in trace[1]]
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    if max_x - min_x < 1 or max_y - min_y < 1:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=150)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Top-left origin for CROHME
    ax.set_aspect('equal')
    ax.axis('off')
    ax.patch.set_facecolor('white')
    
    # Draw all traces as strokes
    for x_coords, y_coords in traces:
        # Normalize to canvas
        x_norm = [(x - min_x) / (max_x - min_x) * width * 0.9 + width * 0.05 for x in x_coords]
        y_norm = [(y - min_y) / (max_y - min_y) * height * 0.9 + height * 0.05 for y in y_coords]
        
        ax.plot(x_norm, y_norm, linewidth=stroke_width, color='black', alpha=0.9)
    
    # Save high-quality image
    filename = os.path.splitext(os.path.basename(inkml_path))[0]
    output_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, facecolor='white', dpi=150)
    plt.close()
    
    return output_path

def batch_convert_crome(input_folder, output_folder):
    """Batch convert CROHME dataset - FIXED for trace elements"""
    os.makedirs(output_folder, exist_ok=True)
    
    inkml_files = glob.glob(os.path.join(input_folder, "*.inkml"))
    print(f"🔍 Found {len(inkml_files)} CROHME .inkml files")
    
    converted = 0
    failed = 0
    
    for i, inkml_path in enumerate(inkml_files, 1):
        img_path = inkml_to_image(inkml_path, output_folder)
        if img_path:
            converted += 1
        else:
            failed += 1
        
        if i % 100 == 0 or i == len(inkml_files):
            print(f"📊 Progress: {i}/{len(inkml_files)} | ✅{converted} | ❌{failed}")
    
    print(f"\n🎉 CONVERSION COMPLETE!")
    print(f"✅ Successfully converted: {converted}/{len(inkml_files)}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Images saved: {output_folder}")

# YOUR EXACT PATHS
# YOUR PATHS (updte these)
input_folder = "C:\\Users\\Vasantha Kamalee\\OneDrive\\Desktop\\research project\\CHROME_merged\\test data\\TestEM2014"  # Change this
output_folder = "C:\\Users\\Vasantha Kamalee\\OneDrive\\Desktop\\research project\\CHROME_merged\\test data\\test images"      # Change this

batch_convert_crome(input_folder, output_folder)

