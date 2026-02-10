from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm
import torch  # torch import 위치 이동

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

    # 배치 사이즈 설정 (Config에 없으면 기본값 8 사용)
    # GPU 메모리에 따라 4, 8, 16 등으로 조절하세요.
    BATCH_SIZE = getattr(cfg, "eval_batch_size", 8)

    ckpt_dir = cfg.CKPT_DIR / cfg.task_mode.value / "final"

    processor_path = str(ckpt_dir) if ckpt_dir.exists() else cfg.pretrained_id
    model_path = str(ckpt_dir) if ckpt_dir.exists() else cfg.pretrained_id

    model, processor = build_model_and_processor(model_path, bf16=cfg.bf16, fp16=cfg.fp16)
    if processor_path != model_path:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    # 생성 시 왼쪽 패딩 권장 (배치 처리를 위함)
    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"

    model.eval()

    rows = read_jsonl(cfg.valid_jsonl)
    preds_out: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    parse_fail = 0

    print(f"Start evaluation with Batch Size: {BATCH_SIZE}")

    with torch.inference_mode():
        # tqdm 단위를 배치가 아닌 전체 샘플 수로 표시하기 위해 total 설정
        pbar = tqdm(total=len(rows), desc=f"eval[{cfg.task_mode.value}]")

        # ---------------------------------------------------------------------
        # Batch Loop
        # ---------------------------------------------------------------------
        for i in range(0, len(rows), BATCH_SIZE):
            # 1. 배치 데이터 준비
            batch_rows = rows[i : i + BATCH_SIZE]

            batch_imgs = []
            batch_texts = []

            for r in batch_rows:
                # 이미지 로드
                batch_imgs.append(_load_image(r["image"], cfg.image_size))

                # 텍스트 템플릿 적용
                user_msg = {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": r["prompt"]}],
                }
                # 개별 프롬프트 문자열 생성
                txt = processor.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
                batch_texts.append(txt)

            # 2. 토크나이징 및 전처리 (일괄 처리)
            # padding=True로 설정해야 배치 내 길이가 다른 문장들을 처리 가능
            inputs = processor(
                images=batch_imgs,
                text=batch_texts,
                return_tensors="pt",
                padding=True
            ).to(model.device)

            # 3. 모델 생성 (일괄 처리)
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.generation_max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )

            # 4. 입력 토큰 제외 (Output Slicing)
            # input_ids 길이만큼 잘라내야 순수 생성 텍스트만 남음
            generated_ids = gen_ids[:, inputs["input_ids"].shape[1]:]

            # 5. 디코딩 (일괄 처리)
            batch_decoded_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # 6. 결과 처리 및 메트릭 계산 (개별 처리)
            for j, text in enumerate(batch_decoded_texts):
                row = batch_rows[j]  # 원본 데이터
                text = text.strip()  # 생성된 텍스트

                pred_obj, err = safe_json_loads(_extract_json_text(text))
                gold_obj, _ = safe_json_loads(_extract_json_text(row["target"]))

                preds_out.append({
                    "image": row["image"],
                    "pred_text": text,
                    "pred_obj": pred_obj,
                    "gold_obj": gold_obj,
                    "parse_error": err,
                })

                if pred_obj is None or gold_obj is None:
                    parse_fail += 1
                else:
                    metric_rows.append(eval_one(pred_obj, gold_obj, cfg.task_mode.value))

            # 진행률 업데이트
            pbar.update(len(batch_rows))
            if (i // BATCH_SIZE + 1) % 10 == 0:  # 로그 빈도 조절
                pbar.set_postfix(parse_fail=parse_fail)

        pbar.close()

    # -------------------------------------------------------------------------
    # 결과 저장
    # -------------------------------------------------------------------------
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
        "batch_size": BATCH_SIZE, # 기록용
    }

    out_prefix = cfg.EVAL_DIR / f"eval_{cfg.task_mode.value}"
    write_json(out_prefix.with_suffix(".summary.json"), summary)
    write_jsonl(out_prefix.with_suffix(".preds.jsonl"), preds_out)

    print("[ok] eval summary saved:", out_prefix.with_suffix(".summary.json"))
    print(summary)


if __name__ == "__main__":
    main()
