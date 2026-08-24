import json
from pathlib import Path

MANIFEST_PATH = "/scratch/m25cse012/CV_major/project_root/output/micro_crops/crop_manifest.json"
VQA_INPUT = "/scratch/m25cse012/CV_major/project_root/output/syn_vqa_pairs.jsonl"
FINAL_JSONL = "/scratch/m25cse012/CV_major/project_root/data/dsyn_surgical_dataset.jsonl"

def assemble():
    with open(MANIFEST_PATH, "r") as f:
        manifest = {m["crop_path"]: m for m in json.load(f)}

    with open(VQA_INPUT, "r") as fin, open(FINAL_JSONL, "w") as fout:
        for line in fin:
            vqa = json.loads(line)
            meta = manifest.get(vqa["crop_path"])
            if not meta: continue

            # Map 1024px crop coords back to original global pixels
            ox, oy, cw, ch = meta["offset"]
            t_ymin, t_xmin, _, _ = vqa["teacher_bbox_1024"]

            gx = int(ox + (t_xmin * cw / 1000))
            gy = int(oy + (t_ymin * ch / 1000))
            
            fout.write(json.dumps({
                "image": meta["global_image"],
                "conversations": [
                    {"from": "user", "value": f"<image>\nWhat is the tool state at ({gx}, {gy})?"},
                    {"from": "assistant", "value": vqa["answer"]}
                ]
            }) + "\n")
    print(f"Training Dataset Ready: {FINAL_JSONL}")

if __name__ == "__main__":
    assemble()
