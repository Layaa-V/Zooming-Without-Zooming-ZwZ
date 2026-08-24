import torch, json, os, re
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

LORA_ADAPTER = "/workspace/project_root/weights/checkpoints1/step_6900"
BASE_MODEL   = "/workspace/Qwen-VL"
ENDOVIS_DIR  = "/workspace/SurgMLLMBench/SurgMLLMBench/EndoVis2018"
ANNOTATIONS  = f"{ENDOVIS_DIR}/test_vqa.json" 
OUTPUT_DIR   = "/workspace/project_root/output/semantic_eval"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load the Similarity Scorer
# This model converts sentences into "concept vectors"
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def run_semantic_evaluation():
    # Load your fine-tuned ZwZ model
    print("Loading ZwZ Model")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, LORA_ADAPTER).eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    with open(ANNOTATIONS, 'r') as f:
        gt_data = json.load(f)

    results = []
    total_similarity = 0

    print(f"Semantically evaluating {len(gt_data)} samples")

    for i, item in enumerate(gt_data):
        img_path = os.path.join(ENDOVIS_DIR, item['image'])
        image = Image.open(img_path).convert("RGB")
        
        # Inference
        prompt = f"<|vision_start|><|image_pad|><|vision_end|>Question: {item['question']}\nAnswer:"
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=30)
        
        pred_text = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        gt_text = item['answer']

        # --- SEMANTIC MATCHING ---
        # Convert both to vectors
        emb1 = similarity_model.encode(pred_text, convert_to_tensor=True)
        emb2 = similarity_model.encode(gt_text, convert_to_tensor=True)
        
        # Calculate Cosine Similarity (0.0 to 1.0)
        score = util.cos_sim(emb1, emb2).item()
        total_similarity += score

        # Binary "Correct" if similarity is > 0.8 (common threshold)
        is_correct = score > 0.8

        results.append({
            "gt": gt_text,
            "pred": pred_text,
            "similarity_score": round(score, 4),
            "match": is_correct
        })

        if i % 50 == 0:
            avg_so_far = total_similarity / (i + 1)
            print(f"Sample {i} | Similarity: {score:.2f} | Avg: {avg_so_far:.2f}")

    # 3. Final Metrics
    mean_similarity = total_similarity / len(gt_data)
    semantic_accuracy = (sum(1 for r in results if r['match']) / len(gt_data)) * 100

    print(f"\nSEMANTIC REPORT")
    print(f"Mean Semantic Similarity: {mean_similarity:.4f}")
    print(f"Semantic Accuracy (>0.8): {semantic_accuracy:.2f}%")

    with open(f"{OUTPUT_DIR}/semantic_report.json", "w") as f:
        json.dump({"mean_similarity": mean_similarity, "accuracy": semantic_accuracy, "data": results}, f, indent=4)

if __name__ == "__main__":
    run_semantic_evaluation()
