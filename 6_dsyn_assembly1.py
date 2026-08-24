import json
import os
import torch
from pathlib import Path
import glob

BASE_DIR = Path("project_root")
OUTPUT_DIR = BASE_DIR / "output"
IMAGE_ROOT = BASE_DIR / "data" / "images"

# Primary Inputs
VQA_PAIRS_FILE = OUTPUT_DIR / "final_unified_7531.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "micro_crops" / "crop_manifest.json"
CROSS_VIEW_FILE = OUTPUT_DIR / "cross_view_pairs.json"

# Final Output
FINAL_TRAINING_FILE = OUTPUT_DIR / "dsyn_training_set.jsonl"

def build_image_indices():
    """Indexed for both Global and Crop images across all folders."""
    global_index = {}
    crop_index = {}
    
    # 1. Index Global Images
    global_paths = glob.glob(str(IMAGE_ROOT / "**" / "*.png"), recursive=True) + \
                   glob.glob(str(IMAGE_ROOT / "**" / "*.jpg"), recursive=True)
    for p in global_paths:
        global_index[os.path.basename(p)] = p

    # 2. Index Crop Images
    crop_paths = glob.glob(str(OUTPUT_DIR / "micro_crops" / "*.jpg"))
    for p in crop_paths:
        crop_index[os.path.basename(p)] = p
        
    print(f"Indexed {len(global_index)} Global images and {len(crop_index)} Crops.")
    return global_index, crop_index

def assemble_dataset():
    print("Professional R2I Dataset Assembly")
    
    global_idx, crop_idx = build_image_indices()
    
    # Load manifest for offsets
    manifest = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r") as f:
            m_list = json.load(f)
            for m in m_list:
                cid = os.path.basename(m.get("crop_id", m.get("crop_path", "")))
                manifest[cid] = m

    assembled_items = []
    
    # 1. Process VQA Distillation Pairs
    print(f"Processing Oracle Distillation Pairs: {VQA_PAIRS_FILE.name}")
    if VQA_PAIRS_FILE.exists():
        with open(VQA_PAIRS_FILE, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    cid = data.get("crop_id")
                    if not cid: continue
                    
                    # RESOLUTION LOGIC:
                    # R2I requires: Global Image (Student) + Crop (Wavelet Branch)
                    full_crop_path = crop_idx.get(cid)
                    
                    # Logic to find the Global Image
                    # Usually, the VQA crop_id matches the SAM crop filename.
                    # We use the manifest to find which global image produced this crop.
                    meta = manifest.get(cid)
                    global_path = None
                    if meta:
                        global_path = meta.get("global_image")
                    
                    # If manifest fails, we attempt to find the source in global_idx
                    if not global_path or not os.path.exists(global_path):
                        # Attempt basename match or similar
                        global_path = global_idx.get(cid.split('_')[0] + ".png") # Heuristic
                    
                    if not full_crop_path or not global_path:
                        continue

                    # Bbox remapping
                    gx1, gy1, gx2, gy2 = 0, 0, 1024, 1024
                    if "bbox" in data and meta:
                        try:
                            # Remap teacher 1024-normalized box to global pixels
                            y1, x1, y2, x2 = data["bbox"]
                            ox, oy, cw, ch = meta["offset"]
                            gx1 = int(ox + (x1 * cw / 1000))
                            gy1 = int(oy + (y1 * ch / 1000))
                            gx2 = int(ox + (x2 * cw / 1000))
                            gy2 = int(oy + (y2 * ch / 1000))
                        except: pass

                    entry = {
                        "type": "vqa_distillation",
                        "global_image_path": str(global_path),
                        "crop_image_path": str(full_crop_path),
                        "question": data.get("question", ""),
                        "answer": data.get("answer", ""),
                        "global_bbox": [gy1, gx1, gy2, gx2]
                    }
                    assembled_items.append(entry)
                except Exception as e: pass

    # 2. Process Geometric Pairs (Homography)
    if CROSS_VIEW_FILE.exists():
        with open(CROSS_VIEW_FILE, "r") as f:
            cv_pairs = json.load(f)
            for p in cv_pairs:
                assembled_items.append({
                    "type": "geometric_consistency",
                    "frame_1": p["frame_1"],
                    "frame_2": p["frame_2"],
                    "homography": p["homography_matrix"]
                })

    print(f"assembled {len(assembled_items)} samples.")
    
    with open(FINAL_TRAINING_FILE, "w") as f:
        for item in assembled_items:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    assemble_dataset()
