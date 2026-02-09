# bench_vlm/train_vlm.py

from __future__ import annotations

from config.config import CFG
from utils.train_utils import (
    set_seed,
    load_samples,
    JsonlVlmDataset,
    VlmCollator,
    build_model_and_processor,
    maybe_apply_lora,
)


def main():
    cfg = CFG
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    train_samples = load_samples(cfg.train_jsonl, cfg.max_train_samples)
    valid_samples = load_samples(cfg.valid_jsonl, cfg.max_valid_samples)

    train_ds = JsonlVlmDataset(train_samples)
    valid_ds = JsonlVlmDataset(valid_samples)

    model, processor = build_model_and_processor(cfg.pretrained_id, bf16=cfg.bf16, fp16=cfg.fp16)
    model = maybe_apply_lora(model, cfg.use_lora, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)

    collator = VlmCollator(
        processor=processor,
        max_prompt_tokens=cfg.max_prompt_tokens,
        max_target_tokens=cfg.max_target_tokens,
        max_image_size=cfg.image_size,
    )

    from transformers import TrainingArguments, Trainer

    out_dir = cfg.CKPT_DIR / cfg.task_mode.value
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_total_limit=2,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
    )

    trainer.train()

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"[ok] saved model: {final_dir}")


if __name__ == "__main__":
    main()
