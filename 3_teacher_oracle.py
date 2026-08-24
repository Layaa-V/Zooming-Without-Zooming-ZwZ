import json
import time
import re
import sys
import os
from pathlib import Path
from PIL import Image
from google import genai


GENAI_KEY ="AIzaSyB_xgnduRsalmdP7uyDAIcRM_MUzxA2K7s"

# Match your specific folder structure
BASE_DIR = Path("project_root")
CROP_DIR = BASE_DIR / "output" / "micro_crops"
OUTPUT_FILE = BASE_DIR / "output" / "syn_vqa_pairs.jsonl"

# The goal for the Teacher phase
TEACHER_TARGET = 1000 
# Free Tier safety: 15 Requests Per Minute (RPM)
RPM_DELAY = 4.6 


def initialize_environment():
    """Validates paths and initializes the 2026 GenAI Client."""
    if not CROP_DIR.exists():
        print(f"err: Crops not found at {CROP_DIR}.")
        sys.exit(1)
    
    # Ensure the output folder exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the client using the new 2026 SDK
    return genai.Client(api_key=GENAI_KEY)

def get_surgical_analysis(client, image_path):
    """Sends image to Gemini for Teacher-level description.
    prompt = analyze this surgical image crop:
    1. Identify the tool and action (e.g., Hook Dissecting, Grasper Retracting).
    2. Provide tool tip coordinates in [ymin, xmin, ymax, xmax] format.
    Return strictly:
    Answer: [Tool] is [Action] on [Tissue].
    BBox: [ymin, xmin, ymax, xmax]"""

    try:
        img = Image.open(image_path)
        # Using the 2026 'Flash-Lite' for maximum speed and quota efficiency
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite-preview',
            contents=[prompt, img]
        )

        text = response.text
        # Extract BBox and Answer using Regex
        bbox_match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", text)
        ans_match = re.search(r"Answer:\s*(.*)", text)

        return {
            "crop_id": image_path.name,
            "expert_answer": ans_match.group(1).strip() if ans_match else text[:100],
            "expert_bbox": [int(x) for x in bbox_match.groups()] if bbox_match else None
        }
    except Exception as e:
        if "429" in str(e): return "RATE_LIMIT"
        print(f"⚠API Error on {image_path.name}: {e}")
        return None

def main():
    client = initialize_environment()

    # --- RESUME LOGIC: Do not repeat work ---
    processed_ids = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # We store the filename (crop_id) to check against
                    processed_ids.add(data["crop_id"])
                except json.JSONDecodeError:
                    continue

    already_done = len(processed_ids)
    print(f"Resume Sync: Found {already_done} already completed images.")

    if already_done >= TEACHER_TARGET:
        print(f"Target reached ({already_done}/{TEACHER_TARGET}).")
        return

    # Identify remaining images
    all_crops = sorted(list(CROP_DIR.glob("*.jpg")))
    # Filter: ONLY process if not already in our JSONL file
    to_process = [c for c in all_crops if c.name not in processed_ids]
    
    # Slice to finish the 1,000 target
    remaining_needed = TEACHER_TARGET - already_done
    queue = to_process[:remaining_needed]


    # Open in 'a' (append) mode so we don't overwrite previous 531 results
    with open(OUTPUT_FILE, "a") as f:
        for i, crop_path in enumerate(queue):
            result = get_surgical_analysis(client, crop_path)
            
            # Handle Quota / Rate Limit
            if result == "RATE_LIMIT":
                print("Rate Limit hit. Sleeping 60s")
                time.sleep(60)
                # One retry attempt
                result = get_surgical_analysis(client, crop_path)

            if result and result != "RATE_LIMIT":
                # Save immediately
                f.write(json.dumps(result) + "\n")
                f.flush() # Forces writing to file so progress isn't lost if you hit Ctrl+C
                
                print(f"[{already_done + i + 1}/{TEACHER_TARGET}] {crop_path.name}")
            
            # Stay under 15 RPM
            time.sleep(RPM_DELAY)

    print("\nTeacher distillation complete.")

if __name__ == "__main__":
    main()
