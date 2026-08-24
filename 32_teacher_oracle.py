import json
import torch
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


WORKSPACE = Path("/workspace/project_root")
CROP_DIR = WORKSPACE / "output" / "micro_crops"
MANIFEST_PATH = WORKSPACE / "output" / "micro_crops" / "crop_manifest.json"

# INPUT SOURCES
PRIMARY_INPUT = Path("/workspace/project_root/data/dsyn_dataset.jsonl") 
GEMINI_FILE = WORKSPACE / "output" / "syn_vqa_pairs.jsonl" 
MODEL_PATH = Path("/workspace/Qwen-VL")
FINAL_OUTPUT = WORKSPACE / "output" / "final_unified_4000.jsonl"

def get_top_4000_crops():
    print("Filtering for the 4,000 highest quality surgical crops")
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    
    # Sort by region size (favor tools over background noise)
    # offset is [x, y, w, h]
    manifest.sort(key=lambda x: x["offset"][2] * x["offset"][3], reverse=True)
    
    top_crops_meta = manifest[:4000]
    top_ids = {m["crop_id"] for m in top_crops_meta}
    return top_ids, top_crops_meta


def load_prioritized_knowledge(valid_ids):
    master_knowledge = {}
    
    #PRIORITY 1: DSYN (GOLD)
    if PRIMARY_INPUT.exists():
        with open(PRIMARY_INPUT, "r") as f:
            for line in f:
                item = json.loads(line)
                fname = Path(item["crop_image"]).name
                if fname in valid_ids:
                    convs = item.get("conversations", [])
                    a_text = next((c["value"] for c in convs if c["from"] == "assistant"), "None")
                    if "None" not in a_text:
                        master_knowledge[fname] = {"question": "Identify surgical tool...", "answer": a_text, "source": "PRIORITY_1_DSYN"}
    
    # PRIORITY 2: Gemini
    if GEMINI_FILE.exists():
        with open(GEMINI_FILE, "r") as f:
            for line in f:
                item = json.loads(line)
                fname = item["crop_id"]
                if fname in valid_ids and fname not in master_knowledge:
                    master_knowledge[fname] = {"question": "Identify surgical tool...", "answer": item["expert_answer"], "source": "PRIORITY_2_GEMINI"}
    
    return master_knowledge


def main():
    valid_ids, _ = get_top_4000_crops()
    knowledge_base = load_prioritized_knowledge(valid_ids)
    
    print(f"Initializing Qwen-VL for {4000 - len(knowledge_base)} Gap-Fill operations...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    with open(FINAL_OUTPUT, "w") as f_out:
        for fname in tqdm(valid_ids, desc="Building 4000 Dataset"):
            img_path = CROP_DIR / fname
            
            if fname in knowledge_base:
                entry = knowledge_base[fname]
                entry["crop_id"] = fname
            else:
                # RUN QWEN STUDENT INFERENCE
                image = Image.open(img_path).convert("RGB")
                q_text = "Identify the surgical tool, action, and tissue."
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": q_text}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    gen_ids = model.generate(**inputs, max_new_tokens=128)
                
                a_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                entry = {"crop_id": fname, "question": q_text, "answer": a_text, "source": "PRIORITY_3_QWEN"}

            f_out.write(json.dumps(entry) + "\n")

    print(f"Filtered 4K Dataset Ready: {FINAL_OUTPUT}")

if __name__ == "__main__":
    main()
