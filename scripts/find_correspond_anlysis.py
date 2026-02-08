import os
from pathlib import Path
from collections import defaultdict

def extract_core_image_id(filename: str):
    """
    파일명(split_tag_imageid_proposed.json)에서 순수 image_id만 추출합니다.
    예: train_SINGLE_STEP1_pexels-photo-3811818_proposed -> pexels-photo-3811818
    """
    # '_proposed' 접미사 제거
    name = filename.replace("_proposed", "").replace(".json", "")
    # '_'로 분할: [split, tag, image_id]
    # tag나 image_id에 '_'가 포함될 수 있으므로 pexels 단어를 기준으로 파싱하거나
    # 데이터 구조에 맞춰 슬라이싱합니다.
    parts = name.split('_')

    # pexels로 시작하는 부분이 image_id라고 가정 (사용자 예시 기반)
    for i, part in enumerate(parts):
        if "pexels" in part:
            return "_".join(parts[i:])
    return name

def main():
    base_dir = Path("runs_eval/analysis/SH17")
    models = ["MADE", "SINGLE_ONESHOT", "SINGLE_STEP"]

    # 모델별 샘플 맵 저장 {model_name: {core_id: (json_path, img_path)}}
    model_samples = {}

    print(f"[*] Scanning models in {base_dir}...")

    for model in models:
        model_dir = base_dir / model / "gen_proposed"
        img_dir = model_dir / "images"
        model_samples[model] = {}

        if not model_dir.exists():
            print(f"[!] Directory not found: {model_dir}")
            continue

        # JSON 파일 탐색
        for json_file in model_dir.glob("*_proposed.json"):
            core_id = extract_core_image_id(json_file.name)

            # 대응하는 이미지 파일 찾기 (확장자 무관하게 탐색)
            img_path = "Image Not Found"
            if img_dir.exists():
                # 파일명 prefix가 core_id와 일치하는 이미지 탐색
                # {split}_{tag}_{image_id} 형태이므로 stem 전체를 활용해 찾음
                base_stem = json_file.name.replace("_proposed.json", "")
                for img_file in img_dir.iterdir():
                    if img_file.stem == base_stem:
                        img_path = str(img_file)
                        break

            model_samples[model][core_id] = (str(json_path := json_file), img_path)

    # 3개 모델 모두에 존재하는 core_id 찾기
    common_ids = set(model_samples[models[0]].keys())
    for model in models[1:]:
        common_ids &= set(model_samples[model].keys())

    # 결과 출력
    print(f"\n[Result] Found {len(common_ids)} common samples across all models.\n")

    if not common_ids:
        print("No common samples found.")
        return

    for i, cid in enumerate(sorted(common_ids), 1):
        print(f"{'='*100}")
        print(f"Sample #{i} | Core Image ID: {cid}")
        print(f"{'='*100}")

        for model in models:
            json_p, img_p = model_samples[model][cid]
            print(f"[{model:15}]")
            print(f"  - JSON:  {json_p}")
            print(f"  - IMAGE: {img_p}")
        print()

if __name__ == "__main__":
    main()
