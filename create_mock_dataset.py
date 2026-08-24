import cv2
import numpy as np
import os
from pathlib import Path
import json

def generate_mock_surgical_frames(num_frames=20):
    images_dir = Path("/scratch/m25cse012/CV_major/project_root/data/images/synthetic")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # We will generate a mock `active_subset_8k.json` too so the script finds them
    subset_json_path = Path("/scratch/m25cse012/CV_major/project_root/data/annotations")
    subset_json_path.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    print(f"Generating {num_frames} synthetic surgical frames...")
    for i in range(num_frames):
        # 1920x1080 background (pinkish tissue)
        img = np.random.normal(loc=150, scale=30, size=(1080, 1920, 3)).astype(np.uint8)
        img[:,:,2] = np.clip(img[:,:,2] + 50, 0, 255) # Add red tint
        
        # Draw a small "instrument" (metallic grey)
        x = np.random.randint(100, 1800)
        y = np.random.randint(100, 900)
        w, h = np.random.randint(20, 100), np.random.randint(50, 200)
        cv2.rectangle(img, (x, y), (x+w, y+h), (120, 120, 120), -1)
        
        # Draw a small "suture/needle" (small green or blue line)
        sx = np.random.randint(100, 1800)
        sy = np.random.randint(100, 900)
        cv2.line(img, (sx, sy), (sx+20, sy+20), (255, 0, 0), 3)
        
        img_name = f"synth_{i}.jpg"
        cv2.imwrite(str(images_dir / img_name), img)
        
        metadata.append({
            "id": f"synth_{i}",
            "image": str(images_dir / img_name),
            "stage": "Mock",
            "phase": "Mock",
            "step": "Mock",
            "instrument_action": ["MockAction"]
        })
        
    with open(subset_json_path / "active_subset_8k.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Mock dataset generated.")

if __name__ == "__main__":
    generate_mock_surgical_frames()
