# MADE-PPE

MADE-PPE structures work situations and hazard factors based on PPE regulations, then uses multi-agent debate (proposer–rebutter–judge) to generate and refine labels so that explainable labeling and schema restructuring are possible. The project builds a regulation-grounded benchmark for PPE wearing and improper-wearing, along with a VLM evaluation pipeline.

Components
- `run.py`: main entry point (labeling/evaluation/analysis)
- `config/`: shared configuration and API key file
- `params/`: prompts, schema, enums, and parameter definitions
- `worker/`: labeling/evaluation/analysis execution logic
- `datasets/`: input datasets
- `runs/`, `runs_eval/`: run outputs and evaluation artifacts
- `bench_vlm/`: VLM benchmark data build/train/eval scripts
- `app_replay.py`: labeling log replay (Streamlit)
- `scripts/`: data/analysis utilities

How to run
1. Main pipeline
```bash
python run.py
```

2. VLM benchmark pipeline
```bash
python bench_vlm/build_train_jsonl.py
python bench_vlm/train_vlm.py
python bench_vlm/eval_vlm.py
```

3. Log replay (optional)
```bash
streamlit run app_replay.py
```

Config locations
- `config/config.py`: shared runtime settings (mode, dataset, model, etc.)
- `config/api_keys.json`: API key file path
- `bench_vlm/config/config.py`: VLM benchmark configuration
