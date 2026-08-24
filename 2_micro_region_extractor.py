import cv2
import json
import sys
import os
from collections import defaultdict
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import resnet50

# --- PATHS ---
SUBSET_JSON = "./project_root/data/annotations/active_subset_8k.json"
OUTPUT_DIR = "./project_root/output/micro_crops"
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "crop_manifest.json")
POSSIBLE_DATASET_ROOTS = [
    Path("./project_root/data/images"),
    Path("./SurgMLLMBench/SurgMLLMBench"),
    Path("./SurgMLLMBench"),
]
SAM_CHECKPOINT = "./weights/sam_vit_h_4b8939.pth"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# OPTIMIZED INDEXER
print(" Building image index")
image_index = {}
indexed_count = 0
for root in POSSIBLE_DATASET_ROOTS:
    if not root.exists(): continue
    for folder, subs, files in os.walk(root):
        for f in files:
            if f.endswith(('.jpg', '.png', '.jpeg')):
                # Index by filename and by the last two parts of the path for accuracy
                full_p = os.path.join(folder, f)
                image_index[f] = full_p
                indexed_count += 1
print(f"Indexed {indexed_count} files.")

#LOAD MODELS
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ Loading SAM on {device}...")
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

print("⚡ Loading Saliency Model...")
saliency_model = resnet50(pretrained=True).eval().to(device)

def get_saliency_map(img):
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)), 
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_t = transform(img).unsqueeze(0).to(device)
    img_t.requires_grad = True
    out = saliency_model(img_t)
    tc = out.argmax(dim=1)
    grad = torch.autograd.grad(out[:, tc], img_t)[0]
    grad = grad.abs().sum(dim=1).squeeze().detach().cpu().numpy()
    return cv2.resize(grad, (img.shape[1], img.shape[0]))

def run():
    with open(SUBSET_JSON, "r") as f:
        samples = json.load(f)

    manifest = []
    saved_total = 0

    for idx, item in enumerate(samples):
        #Resolve Path from Index
        image_field = item.get("image")
        img_name = os.path.basename(image_field)
        img_path = image_index.get(img_name)

        if img_path is None or not os.path.exists(img_path):
            continue

        #Process Image
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        s_map = get_saliency_map(img)
        masks = mask_generator.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        crop_count = 0
        for m in masks:
            x, y, w, h = map(int, m["bbox"])
            
            # FILTERS (Same as original stable run)
            if w*h > 0:
                s_score = np.mean(s_map[y:y+h, x:x+w])
                if s_score < 0.3: continue # Balanced threshold
            
            if (w*h) / (img.shape[0]*img.shape[1]) > 0.10: continue

            # Save Crop (Using EXACT Original Naming Convention)
            crop = img[y:y+h, x:x+w]
            crop = cv2.resize(crop, (1024, 1024))
            
            # IMPORTANT: Reverting to original naming to match VQA
            sample_id = item.get("id", f"sample_{idx}")
            crop_filename = f"{sample_id}_{crop_count}.jpg"
            out_path = os.path.join(OUTPUT_DIR, crop_filename)
            cv2.imwrite(out_path, crop)

            # Record in Manifest
            manifest.append({
                "crop_id": crop_filename,
                "global_image": str(os.path.abspath(img_path)),
                "offset": [x, y, w, h]
            })
            crop_count += 1
            saved_total += 1

        if idx % 50 == 0:
            print(f"[{idx}/{len(samples)}] Saved Total: {saved_total}")
            with open(MANIFEST_PATH, "w") as f_m:
                json.dump(manifest, f_m)

    print(f"Total Crops Saved: {saved_total}")

if __name__ == "__main__":
    run()
