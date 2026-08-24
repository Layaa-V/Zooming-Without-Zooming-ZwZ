import json
import random
from pathlib import Path
from collections import defaultdict

# Paths
ROOT = Path("./project_root")
ANN_DIR = ROOT / "data" / "annotations"
ANN_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_ROOT = Path("./SurgMLLMBench/SurgMLLMBench")

SUBSET_SIZE = 8000
PER_DATASET_LIMIT = 2000

# Mapping for MISAW CASE folders to extracted folders
# Based on my research, CASE018 maps to 1_1 if we follow Subject_Session logic
# Since we have 18 folders and roughly 18 CASEs in total, we can do a matching.
MISAW_FOLDERS = [
    "1_1", "1_2", "1_3", "2_1", "2_4", "2_6", "3_1", "3_2", "3_3", 
    "4_1", "4_2", "4_4", "5_1", "5_4", "5_5", "6_1", "6_2", "6_4"
]

def collect_all_samples():
    all_samples = []
    meta_files = list(SOURCE_ROOT.rglob("*.json*"))
    meta_files = [f for f in meta_files if f.suffix in [".json", ".jsonl"]]

    print(f"Found {len(meta_files)} potential annotation files")

    for meta_path in meta_files:
        try:
            with open(meta_path, "r") as f:
                if meta_path.suffix == ".jsonl":
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
        except Exception as e:
            print(f"Error reading {meta_path}: {e}")
            continue

        if not isinstance(data, list):
            continue

        # Dataset name is usually the directory name under SOURCE_ROOT
        dataset_name = meta_path.relative_to(SOURCE_ROOT).parts[0]

        for item in data:
            if not isinstance(item, dict): continue
            item["source_dataset"] = dataset_name
            all_samples.append(item)

    return all_samples

def create_subset():
    all_samples = collect_all_samples()
    print(f"Total samples collected: {len(all_samples)}")

    if not all_samples:
        print("❌ No samples found! Check SOURCE_ROOT.")
        return

    # Group by dataset
    grouped = defaultdict(list)
    for s in all_samples:
        grouped[s["source_dataset"]].append(s)

    final_subset = []
    for ds_name, ds_samples in grouped.items():
        print(f"Dataset {ds_name}: {len(ds_samples)} samples")
        
        # Sample up to PER_DATASET_LIMIT
        if len(ds_samples) > PER_DATASET_LIMIT:
            sampled = random.sample(ds_samples, PER_DATASET_LIMIT)
        else:
            sampled = ds_samples
        
        final_subset.extend(sampled)

    # If we have more than 8k due to small datasets, shuffle and truncate
    if len(final_subset) > SUBSET_SIZE:
        random.shuffle(final_subset)
        final_subset = final_subset[:SUBSET_SIZE]

    out_path = ANN_DIR / "active_subset_8k.json"
    with open(out_path, "w") as f:
        json.dump(final_subset, f, indent=2)

    print(f"Saved {len(final_subset)} samples : {out_path}")

if __name__ == "__main__":
    create_subset()
