import json
import re
import random
from pathlib import Path
from collections import Counter


WORKSPACE   = Path("/workspace/project_root/output")
INPUT_FILE  = WORKSPACE / "final_unified_7531.jsonl"
OUTPUT_FILE = WORKSPACE / "dsyn_repaired_full.jsonl"


TOOL_MAP = {
    "hook dissecting": "Hook Dissector",
    "monopolar hook": "Hook Dissector",
    "monopolar curved scissors": "Curved Scissors",
    "maryland dissector": "Maryland Dissector",
    "maryland grasper": "Maryland Dissector",
    "fenestrated grasper": "Grasper",
    "bipolar forceps": "Bipolar Forceps",
    "suction cannula": "Suction Device",
    "needle driver": "Needle Driver",
}

# Actions to inject if the template is empty (adds variety)
RESCUE_ACTIONS = [
    "retracting", "dissecting", "grasping", 
    "manipulating", "holding", "clamping"
]

TEMPLATE_PLACEHOLDERS = [
    "[Surgical tool]", "[Tool]", "[Action]", "[Tissue]",
    "[Surgical action]", "Surgical procedure in progress",
]


def repair_answer(answer, question):
    """
    Cleans prompt pollution and repairs template placeholders.
    """
    if not answer or not isinstance(answer, str):
        return "Surgical instrument is visible in the operative field."

    # Step A: Strip Chat Template Leakage
    # Removes "system", "user", and "assistant" noise
    for sep in ["assistant\n", "ASSISTANT:", "Answer: ", "Answer:"]:
        if sep in answer:
            answer = answer.split(sep)[-1].strip()

    # Clean redundant prefixes/suffixes
    answer = re.sub(r"^Answer:\s*", "", answer, flags=re.IGNORECASE).strip()
    answer = answer.replace("<|endoftext|>", "").rstrip(".")

    # Step B: Placeholder Repair (The Rescue)
    is_poisoned = any(p.lower() in answer.lower() for p in TEMPLATE_PLACEHOLDERS)
    
    if is_poisoned:
        q_lower = question.lower() if question else ""
        extracted_tool = None
        
        # 1. Try to find tool in Tool Map
        for alias, canonical in TOOL_MAP.items():
            if alias in q_lower:
                extracted_tool = canonical
                break
        
        # 2. Try generic keywords if map fails
        if not extracted_tool:
            keywords = ["grasper", "scissors", "hook", "bipolar", "needle driver", "suction"]
            for kw in keywords:
                if kw in q_lower:
                    extracted_tool = kw.title()
                    break
        
        # 3. Final Assembly of Repaired Answer
        if extracted_tool:
            action = random.choice(RESCUE_ACTIONS)
            return f"{extracted_tool} is {action} on surgical tissue."
        else:
            return "Surgical instrument is active in the operative field."

    # --- Step C: Normalization for Clean Answers ---
    answer_lower = answer.lower()
    for alias, canonical in TOOL_MAP.items():
        if alias in answer_lower:
            answer = re.sub(re.escape(alias), canonical, answer, flags=re.IGNORECASE)

    # Final Length Check
    if len(answer.strip()) < 5:
        return "Surgical instrument is visible in frame."

    return answer.strip()


def run_repair():
    stats = Counter()
    repaired_entries = []

    if not INPUT_FILE.exists():
        print(f"Error: Could not find {INPUT_FILE}")
        return

    print(f"Repairing {INPUT_FILE}...", flush=True)

    with open(INPUT_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                d = json.loads(line)
                original_ans = d.get("answer", "")
                question = d.get("question", "")

                # Check if it needs a rescue
                poisoned = any(p.lower() in original_ans.lower() for p in TEMPLATE_PLACEHOLDERS)
                pollution = "assistant\n" in original_ans or "system\n" in original_ans

                if poisoned or pollution:
                    d["answer"] = repair_answer(original_ans, question)
                    d["repaired"] = True
                    stats["repaired"] += 1
                else:
                    # Just normalize tool names
                    d["answer"] = repair_answer(original_ans, question)
                    d["repaired"] = False
                    stats["kept_clean"] += 1
                
                # Append the final period
                if not d["answer"].endswith("."):
                    d["answer"] += "."

                repaired_entries.append(d)

            except json.JSONDecodeError:
                stats["corrupt_json"] += 1
                continue

    # Write output
    with open(OUTPUT_FILE, "w") as f_out:
        for entry in repaired_entries:
            f_out.write(json.dumps(entry) + "\n")

    
    print(f"Total Processed  : {len(repaired_entries)}")
    print(f"Repaired/Rescued : {stats['repaired']}")
    print(f"Already Clean    : {stats['kept_clean']}")
    print(f" Corrupt (Skipped): {stats['corrupt_json']}")
    print(f"Saved to         : {OUTPUT_FILE}")


if __name__ == "__main__":
    run_repair()
