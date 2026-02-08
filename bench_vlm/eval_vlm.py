from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import torch
from PIL import Image
from tqdm.auto import tqdm

from config.config import CFG
from utils.io_utils import write_json, write_jsonl, read_jsonl
from utils.metrics import safe_json_loads, eval_one, avg_metrics


# -----------------------------
# helpers
# -----------------------------
def resolve_path(p: str | Path, root: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (root / pp).resolve()


def maybe_resize(img: Image.Image, max_side: int | None) -> Image.Image:
    if not max_side:
        return img
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    return img


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_processor_for_eval(cfg, ckpt_dir: Path):
    """
    processor 파일이 ckpt_dir에 저장되어 있으면 거기서 로드,
    없으면 pretrained_id에서 로드.
    """
    from transformers import AutoProcessor

    has_proc = (ckpt_dir / "preprocessor_config.json").exists() or (ckpt_dir / "processor_config.json").exists()
    src = str(ckpt_dir) if has_proc else cfg.pretrained_id
    processor = AutoProcessor.from_pretrained(src, trust_remote_code=True)
    return processor


def load_model_for_eval(cfg, ckpt_dir: Path):
    """
    - adapter_config.json 있으면: base(pretrained_id) 로드 후 PeftModel로 adapter를 붙임
    - 아니면: ckpt_dir을 모델로 로드
    """
    from transformers import AutoConfig
    from transformers import AutoModelForVision2Seq, AutoModelForCausalLM

    dtype = torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else None)

    is_adapter = (ckpt_dir / "adapter_config.json").exists()

    if is_adapter:
        try:
            base = AutoModelForVision2Seq.from_pretrained(
                cfg.pretrained_id,
                trust_remote_code=True,
                torch_dtype=dtype if dtype is not None else "auto",
                device_map="auto",
            )
        except Exception:
            base = AutoModelForCausalLM.from_pretrained(
                cfg.pretrained_id,
                trust_remote_code=True,
                torch_dtype=dtype if dtype is not None else "auto",
                device_map="auto",
            )

        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(ckpt_dir))
        return model
    
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            str(ckpt_dir),
            trust_remote_code=True,
            torch_dtype=dtype if dtype is not None else "auto",
            device_map="auto",
        )
        return model
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir),
            trust_remote_code=True,
            torch_dtype=dtype if dtype is not None else "auto",
            device_map="auto",
        )
        return model


def build_mm_prompt(processor, user_text: str, image_token: str = "<image>") -> str:
    """
    모델별로 멀티모달 프롬프트 포맷이 달라질 수 있으니
    apply_chat_template가 있으면 그걸 우선 사용.
    없으면 가장 보편적인 "<image>\n{user_text}" fallback.
    """
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return f"{image_token}\n{user_text}"


def decode_generated(processor, gen_ids: torch.Tensor) -> str:
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        from transformers import AutoTokenizer
        raise RuntimeError("processor.tokenizer not found. Please load tokenizer explicitly.")
    return tok.decode(gen_ids, skip_special_tokens=True)


# -----------------------------
# main
# -----------------------------
def main():
    cfg = CFG
    cfg.ensure_dirs()

    ckpt_dir = cfg.CKPT_DIR / f"{cfg.model_name.value}_{cfg.dataset_name}_{cfg.label_run_tag}" / "final"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_dir}")

    processor = load_processor_for_eval(cfg, ckpt_dir)
    model = load_model_for_eval(cfg, ckpt_dir)
    model.eval()

    device = get_model_device(model)

    rows = read_jsonl(cfg.valid_jsonl)
    preds_out = []
    metric_rows = []
    parse_fail = 0

    MAX_EVAL_IMAGE_SIDE = getattr(cfg, "max_eval_image_side", 512)

    use_amp = (cfg.bf16 or cfg.fp16) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float16

    for i, r in enumerate(tqdm(rows, desc="eval")):
        img_path = resolve_path(r["image"], cfg.ROOT)
        img = Image.open(img_path).convert("RGB")
        img = maybe_resize(img, MAX_EVAL_IMAGE_SIDE)

        user_prompt = r["prompt"]
        prompt_text = build_mm_prompt(
            processor,
            user_text=user_prompt,
            image_token=getattr(cfg, "image_token", "<image>"),
        )

        inputs = processor(
            images=[img],
            text=[prompt_text],
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    gen = model.generate(**inputs, max_new_tokens=cfg.generation_max_new_tokens)
            else:
                gen = model.generate(**inputs, max_new_tokens=cfg.generation_max_new_tokens)

        text = decode_generated(processor, gen[0])

        pred_obj, err = safe_json_loads(text)
        gold_obj, _ = safe_json_loads(r["target"])

        item = {
            "image": str(img_path),
            "pred_text": text,
            "pred_obj": pred_obj,
            "gold_obj": gold_obj,
            "parse_error": err,
        }
        preds_out.append(item)

        if pred_obj is None or gold_obj is None:
            parse_fail += 1
            continue

        m = eval_one(pred_obj, gold_obj, cfg.task_scope)
        metric_rows.append(m)

        if (i + 1) % 20 == 0:
            print(f"[eval] {i+1}/{len(rows)}... parse_fail={parse_fail}")

    avg = avg_metrics(metric_rows)
    summary = {
        "dataset": cfg.dataset_name,
        "label_run_tag": cfg.label_run_tag,
        "task_scope": cfg.task_scope,
        "num_samples": len(rows),
        "num_parse_fail": parse_fail,
        "parse_success_rate": (len(rows) - parse_fail) / max(1, len(rows)),
        "avg_metrics": avg,
        "ckpt_dir": str(ckpt_dir),
        "pretrained_id": cfg.pretrained_id,
        "model_name": cfg.model_name.value,
    }

    out_prefix = cfg.EVAL_DIR / f"eval_{cfg.model_name.value}_{cfg.dataset_name}_{cfg.label_run_tag}"
    write_json(out_prefix.with_suffix(".summary.json"), summary)
    write_jsonl(out_prefix.with_suffix(".preds.jsonl"), preds_out)

    print("[ok] eval summary saved:", out_prefix.with_suffix(".summary.json"))
    print(summary)


if __name__ == "__main__":
    main()
