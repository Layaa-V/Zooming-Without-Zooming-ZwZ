import json
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


WORKSPACE = Path("/workspace/project_root")
CROP_DIR = WORKSPACE / "output" / "micro_crops"

# INPUT SOURCES
PRIMARY_INPUT = Path("/workspace/project_root/data/dsyn_dataset.jsonl") 
GEMINI_FILE = WORKSPACE / "output" / "syn_vqa_pairs.jsonl" 

MODEL_PATH = Path("/workspace/Qwen-VL")
FINAL_OUTPUT = WORKSPACE / "output" / "final_unified_7531.jsonl"


def load_prioritized_knowledge():
    master_knowledge = {}
    
    # --- PRIORITY 1: Metadata from dsyn_dataset.jsonl (GOLD) ---
    if PRIMARY_INPUT.exists():
        with open(PRIMARY_INPUT, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    fname = Path(item["crop_image"]).name
                    
                    # Extract the original question and answer from the 'conversations' list
                    convs = item.get("conversations", [])
                    q_text = next((c["value"] for c in convs if c["from"] == "user"), "Unknown Question")
                    a_text = next((c["value"] for c in convs if c["from"] == "assistant"), "None")

                    # We only take it if it's high-quality (not "None")
                    if "None" not in a_text:
                        master_knowledge[fname] = {
                            "question": q_text,
                            "answer": a_text,
                            "source": "PRIORITY_1_DSYN",
                            "original_meta": item # Keep full original item (masks, IDs, etc.)
                        }
                except: continue
    print(f"Priority 1: Loaded {len(master_knowledge)} Gold entries from dsyn_dataset.")

    # --- PRIORITY 2: Original Gemini (Capped at 1000 Total for this Source) ---
    gem_count = 0
    if GEMINI_FILE.exists():
        with open(GEMINI_FILE, "r") as f:
            for line in f:
                if gem_count >= 1000: break # Hard bound for Gemini entries
                try:
                    item = json.loads(line)
                    fname = item["crop_id"]
                    
                    if fname not in master_knowledge and "None" not in item["expert_answer"]:
                        master_knowledge[fname] = {
                            "question": "Identify the surgical tool, action, and tissue.",
                            "answer": item["expert_answer"],
                            "source": "PRIORITY_2_GEMINI",
                            "bbox": item.get("expert_bbox")
                        }
                        gem_count += 1
                except: continue
    print(f"🤖 Priority 2: Added {gem_count} Gemini Teacher entries.")
    
    return master_knowledge


def main():
    knowledge_base = load_prioritized_knowledge()
    
    print("Loading Qwen-VL for Student Gap Filling")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    all_crops = sorted(list(CROP_DIR.glob("*.jpg")))
    
    with open(FINAL_OUTPUT, "w") as f_out:
        for img_path in tqdm(all_crops, desc="Building 7531 Dataset"):
            fname = img_path.name
            
            # CASE A: Found in Priority 1 (DSYN) or Priority 2 (GEMINI)
            if fname in knowledge_base:
                entry = knowledge_base[fname]
                entry["crop_id"] = fname # Ensure ID is present
            
            # CASE B: Gap Fill (Priority 3 - Qwen Student)
            else:
                image = Image.open(img_path).convert("RGB")
                q_text = "Identify the surgical tool, action, and tissue. Format: Answer: [Tool] is [Action] on [Tissue]."
                
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": q_text}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    gen_ids = model.generate(**inputs, max_new_tokens=128)
                
                a_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                
                entry = {
                    "crop_id": fname,
                    "question": q_text,
                    "answer": a_text,
                    "source": "PRIORITY_3_QWEN"
                }

            f_out.write(json.dumps(entry) + "\n")

    print(f"MASTER DATASET READY: {FINAL_OUTPUT}")

if __name__ == "__main__":
    main()
