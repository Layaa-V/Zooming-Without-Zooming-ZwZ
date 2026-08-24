import cv2
import numpy as np
import json
from pathlib import Path
import os
import glob


BASE_DIR = Path("project_root")
OUTPUT_DIR = BASE_DIR / "output"
GLOBAL_IMAGES_ROOT = BASE_DIR / "data" / "images"
MANIFEST_PATH = OUTPUT_DIR / "micro_crops" / "crop_manifest.json"
OUTPUT_PAIRS = OUTPUT_DIR / "cross_view_pairs.json"

def compute_homography(img_path1, img_path2):
    """Computes Homography between two global images."""
    img1 = cv2.imread(str(img_path1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img_path2), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None: return None
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or len(des1) < 4 or des2 is None or len(des2) < 4: return None
        
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        if len(good) >= 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H
    except: pass
    return None

def main():
        
    # 1. Index all global images to handle multiple database folders
    print(f"Indexing global images in {GLOBAL_IMAGES_ROOT}")
    all_global_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        all_global_images.extend(glob.glob(str(GLOBAL_IMAGES_ROOT / "**" / ext), recursive=True))
    
    # Group by directory to find temporal/session sequences
    sequences = {}
    for img_path in all_global_images:
        session_id = os.path.dirname(img_path)
        sequences.setdefault(session_id, []).append(img_path)
    
    print(f"Found {len(sequences)} session folders across datasets.")
    
    paired_data = []
    total_analyzed = 0
    
    for session_id, img_paths in sequences.items():
        # Sort files to ensure we compare adjacent frames in the sequence
        img_paths.sort()
        
        if len(img_paths) > 1:
            print(f" Pocessing sequence: {os.path.basename(session_id)} ({len(img_paths)} frames)")
            
            # Subsample or limit if needed, here we check adjacent frames
            for i in range(len(img_paths) - 1):
                p1, p2 = img_paths[i], img_paths[i+1]
                
                # To save time, we only analyze a subset of the global images 
                # (e.g., every 5th pair) or just a fixed amount
                if i % 10 != 0: continue 
                
                H = compute_homography(p1, p2)
                total_analyzed += 1
                
                if H is not None:
                    paired_data.append({
                        "session": session_id,
                        "frame_1": p1,
                        "frame_2": p2,
                        "homography_matrix": H.tolist()
                    })
                    if len(paired_data) % 5 == 0:
                        print(f"   Pairs found: {len(paired_data)}")
                
                if len(paired_data) >= 500: break # Safety limit for now
        
        if len(paired_data) >= 500: break

    with open(OUTPUT_PAIRS, "w") as f:
        json.dump(paired_data, f, indent=4)
        
    print(f"Analyzed {total_analyzed} pairs. Saved {len(paired_data)} Global Homography Pairs.")

if __name__ == "__main__":
    main()
