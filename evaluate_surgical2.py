
import torch, json, os, cv2, random, time, glob, math
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

LORA_ADAPTER = "/workspace/project_root/weights/qwen_lora_final"
BASE_MODEL   = "/workspace/Qwen-VL"
TEST_JSONL   = "/workspace/project_root/output/dsyn_repaired_full.jsonl"
OUTPUT_DIR   = "/workspace/project_root/output/inference_results"
N_SAMPLES    = 10

IMAGE_ROOTS = [
    "/workspace/project_root/data/images",
    "/workspace/SurgMLLMBench/SurgMLLMBench",
    "/workspace/SurgMLLMBench",
]


os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_valid_endo_frame(img_path: str) -> bool:
    name = os.path.basename(img_path).lower()
 #   if any(p in name for p in SKIP_PATTERNS):
  #      return False
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
        if img.shape[0] < 100 or img.shape[1] < 100:
            return False
        if img.mean() < 20:          # near-black → mask frame
            return False
        return True
    except Exception:
        return False


def get_eval_images(n: int = 10) -> list:
    all_imgs = []
    for root in IMAGE_ROOTS:
        if os.path.exists(root):
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                all_imgs.extend(glob.glob(f"{root}/**/{ext}", recursive=True))

    valid = [p for p in all_imgs if is_valid_endo_frame(p)]
    print(f" Valid endo frames found: {len(valid)}", flush=True)

    random.seed(int(time.time()))   # different images every run
    random.shuffle(valid)
    return valid


def build_gt_lookup() -> dict:
    lookup = {}
    try:
        with open(TEST_JSONL) as f:
            for line in f:
                d = json.loads(line)
                p = d.get("global_image_path", "")
                if p:
                    lookup[os.path.basename(p)] = {
                        "question": d.get("question",
                                          "Identify the surgical tool, action, and tissue."),
                        "answer":   d.get("answer", ""),
                    }
    except Exception as e:
        print(f"GT lookup error: {e}", flush=True)
    print(f"GT lookup: {len(lookup)} entries.", flush=True)
    return lookup


# VISUALISATION  — image + Q/GT/PRED text panel, no bbox
def save_viz(img_path: str, question: str, pred: str, gt: str, idx: int):
    img = cv2.imread(img_path)
    if img is None:
        return
    img = cv2.resize(img, (900, 560))
    h, w = img.shape[:2]

    panel = img.copy()
    cv2.rectangle(panel, (0, h - 120), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.75, img, 0.25, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Q:    {question[:90]}",  (10, h - 92), font, 0.44, (200, 200, 200), 1)
    cv2.putText(img, f"GT:   {gt[:90]}",         (10, h - 62), font, 0.44, (80, 255, 80),  1)
    cv2.putText(img, f"PRED: {pred[:90]}",        (10, h - 32), font, 0.44, (0, 220, 255),  1)
    cv2.rectangle(img, (0, 0), (140, 38), (0, 0, 0), -1)
    cv2.putText(img, f"Sample #{idx + 1}", (6, 26), font, 0.7, (255, 255, 0), 2)

    out = f"{OUTPUT_DIR}/result_{idx + 1:02d}.jpg"
    cv2.imwrite(out, img)
    print(f"{out}", flush=True)

# GRAD-CAM — fixed for Qwen2-VL
def get_target_layer(model):
    candidates = [
        lambda m: m.model.visual.blocks[-1].norm1,
        lambda m: m.model.visual.blocks[-1].norm2,
        lambda m: m.model.visual.blocks[-1],
        lambda m: m.visual.blocks[-1].norm1,
        lambda m: m.visual.blocks[-1],
        lambda m: [mod for name, mod in m.named_modules()
                   if "visual" in name and isinstance(mod, torch.nn.LayerNorm)][-1],
    ]
    for fn in candidates:
        try:
            layer = fn(model)
            if layer is not None:
                return layer
        except Exception:
            continue
    return None


def run_gradcam(model, inputs: dict, img_path: str, idx: int):
    target = get_target_layer(model)
    if target is None:
        print("GradCAM: no valid layer found.", flush=True)
        return

    act, grad = [None], [None]
    h1 = target.register_forward_hook(
        lambda m, i, o: act.__setitem__(0, o.detach().clone()))
    h2 = target.register_full_backward_hook(
        lambda m, gi, go: grad.__setitem__(0, go[0].detach().clone()))

    try:
        ci = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                ci[k] = v.clone().float().requires_grad_(True)
            elif isinstance(v, torch.Tensor):
                ci[k] = v.clone()
            else:
                ci[k] = v

        out   = model(**ci)
        score = out.logits[0, -1, :].max()
        model.zero_grad()
        score.backward(retain_graph=True)

        if act[0] is None or grad[0] is None:
            print(" GradCAM hooks returned None.", flush=True)
            return

        w   = grad[0].mean(dim=list(range(grad[0].dim() - 1)), keepdim=True)
        cam = torch.relu((w * act[0]).sum(-1).squeeze())
        if cam.numel() == 0:
            return

        s   = int(math.sqrt(cam.numel())) if cam.dim() == 1 else cam.shape[0]
        cam = cam.float().cpu().numpy()
        if cam.ndim == 1:
            cam = cam[:s * s].reshape(s, s)

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (900, 560))

        img   = cv2.resize(cv2.imread(img_path), (900, 560))
        heat  = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        fused = cv2.addWeighted(img, 0.5, heat, 0.5, 0)

        cv2.putText(fused, "GRAD-CAM: Model Attention", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(fused, "Red=High Attention  Blue=Low Attention", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out_p = f"{OUTPUT_DIR}/gradcam_{idx + 1:02d}.jpg"
        cv2.imwrite(out_p, fused)
        print(f" GradCAM: {out_p}", flush=True)

    except Exception as e:
        print(f" GradCAM error: {e}", flush=True)
    finally:
        h1.remove()
        h2.remove()

# MAIN
def run():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {BASE_MODEL}", flush=True)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if os.path.exists(LORA_ADAPTER) and os.path.exists(
            os.path.join(LORA_ADAPTER, "adapter_config.json")):
        print(f" Loading LoRA adapter: {LORA_ADAPTER}", flush=True)
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
    else:
        print(" No LoRA adapter found — running base model (zero-shot).", flush=True)
        model = base_model

    model.eval()
    print(" Model ready.\n", flush=True)

    img_paths = get_eval_images(N_SAMPLES)
    gt_lookup = build_gt_lookup()

    if not img_paths:
        print(" No valid images found. Check IMAGE_ROOTS.", flush=True)
        return

    Q = "Identify the surgical tool, action, and tissue."
    PROMPT_TEMPLATE = (
        "<|vision_start|><|image_pad|><|vision_end|>"
        "Question: {q}\n"
        "Answer with the tool name, action, and tissue.\n"
        "Answer:"
    )

    lats = []
    log  = []

    for i, img_path in enumerate(img_paths):
        bname   = os.path.basename(img_path)
        gt_info = gt_lookup.get(bname, {})
        question = gt_info.get("question", Q)
        gt_text  = gt_info.get("answer", "")

        print(f"\n{'='*55}", flush=True)
        print(f"[{i+1}/{N_SAMPLES}] {bname}", flush=True)
        if gt_text:
            print(f"GT:   {gt_text[:90]}", flush=True)

        image  = Image.open(img_path).convert("RGB")
        prompt = PROMPT_TEMPLATE.format(q=question)

        raw = processor(
            text=[prompt], images=[image],
            return_tensors="pt", padding=True,
        )
        inputs = {
            k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in raw.items()
        }

        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=120,
                do_sample=False, repetition_penalty=1.15,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        lat = time.time() - t0
        lats.append(lat)

        pred = processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        print(f"PRED: {pred[:90]}", flush=True)
        print(f"⏱️   {lat:.1f}s", flush=True)

        save_viz(img_path, question, pred, gt_text or "N/A", i)
        run_gradcam(model, inputs, img_path, i)

        log.append({
            "sample":       i + 1,
            "image":        bname,
            "question":     question,
            "ground_truth": gt_text,
            "prediction":   pred,
            "latency_s":    round(lat, 2),
        })
    avg_lat = round(float(np.mean(lats)), 2) if lats else 0.0

    print(f"\n{'='*55}", flush=True)
    print(f"  RESULTS  ({len(log)} samples)", flush=True)
    print(f"{'='*55}", flush=True)
    print(f"  Avg Latency : {avg_lat:.2f}s/image", flush=True)
    print(f"  Results     : {OUTPUT_DIR}/result_*.jpg", flush=True)
    print(f"  GradCAM     : {OUTPUT_DIR}/gradcam_*.jpg", flush=True)
    print(f"{'='*55}", flush=True)

    summary = {
        "avg_latency_s": avg_lat,
        "n_samples":     len(log),
        "samples":       log,
    }
    out_path = f"{OUTPUT_DIR}/summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f" Saved: {out_path}", flush=True)


if __name__ == "__main__":
    run()
