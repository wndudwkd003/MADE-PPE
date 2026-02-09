# bench_vlm/eval_vlm.py

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

from config.config import CFG
from utils.io_utils import write_json, write_jsonl, read_jsonl
from utils.metrics import safe_json_loads, eval_one, avg_metrics
from utils.train_utils import build_model_and_processor


def _resolve_image_path(p: str) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((CFG.ROOT / pp).resolve())


def _load_image(p: str, max_size: int | None) -> Image.Image:
    img = Image.open(_resolve_image_path(p)).convert("RGB")
    if max_size is not None:
        img.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
    return img


def _extract_json_text(s: str) -> str:
    s = (s or "").strip()

    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        s = s[l:r+1].strip()

    return s


def main():
    cfg = CFG
    cfg.ensure_dirs()

    ckpt_dir = cfg.CKPT_DIR / cfg.task_mode.value / "final"

    processor_path = str(ckpt_dir) if ckpt_dir.exists() else cfg.pretrained_id
    model_path = str(ckpt_dir) if ckpt_dir.exists() else cfg.pretrained_id

    model, processor = build_model_and_processor(model_path, bf16=cfg.bf16, fp16=cfg.fp16)
    if processor_path != model_path:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    model.eval()

    rows = read_jsonl(cfg.valid_jsonl)
    preds_out: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    parse_fail = 0

    import torch
    with torch.inference_mode():
        for i, r in enumerate(tqdm(rows, desc=f"eval[{cfg.task_mode.value}]")):
            img = _load_image(r["image"], cfg.image_size)
            prompt = r["prompt"]

            user_msg = {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
            prompt_text = processor.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)

            inputs = processor(images=[img], text=[prompt_text], return_tensors="pt", padding=True).to(model.device)

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.generation_max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )

            # ★ prompt 이후 생성된 토큰만 decode
            new_tokens = gen_ids[0, inputs["input_ids"].shape[1]:]
            text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            pred_obj, err = safe_json_loads(_extract_json_text(text))
            gold_obj, _ = safe_json_loads(_extract_json_text(r["target"]))

            preds_out.append({
                "image": r["image"],
                "pred_text": text,
                "pred_obj": pred_obj,
                "gold_obj": gold_obj,
                "parse_error": err,
            })

            if pred_obj is None or gold_obj is None:
                parse_fail += 1
                continue

            metric_rows.append(eval_one(pred_obj, gold_obj, cfg.task_mode.value))

            if (i + 1) % 50 == 0:
                print(f"[eval] {i+1}/{len(rows)} parse_fail={parse_fail}")

    avg = avg_metrics(metric_rows)
    summary = {
        "dataset": cfg.dataset_name,
        "label_run_tag": cfg.label_run_tag,
        "task_mode": cfg.task_mode.value,
        "num_samples": len(rows),
        "num_parse_fail": parse_fail,
        "parse_success_rate": (len(rows) - parse_fail) / max(1, len(rows)),
        "avg_metrics": avg,
        "f1_macro": avg.get("f1_macro", 0.0),
        "ckpt_dir": str(ckpt_dir),
        "pretrained_id": cfg.pretrained_id,
    }

    out_prefix = cfg.EVAL_DIR / f"eval_{cfg.task_mode.value}"
    write_json(out_prefix.with_suffix(".summary.json"), summary)
    write_jsonl(out_prefix.with_suffix(".preds.jsonl"), preds_out)

    print("[ok] eval summary saved:", out_prefix.with_suffix(".summary.json"))
    print(summary)


if __name__ == "__main__":
    main()
