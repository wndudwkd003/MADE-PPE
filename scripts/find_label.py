# scripts/find_label.py


from config.config import Config
from utils.clip_utils import clip_image_text_sims
from pathlib import Path
import os
import json


IMAGE_PATH = "runs/_cache_resized/pexels-photo-8961394__7913x5275__pad__512.jpg"

LABEL_DIRS = [
    Path("runs/SH17/MADE1/train/outputs/labels"),
    Path("runs/SH17/MADE1/valid/outputs/labels"),
]


def get_image_id(image_path: str):
    filename = os.path.basename(image_path)
    base, _ = os.path.splitext(filename)
    # 캐시 파일명은 "pexels-photo-8961394__...__pad__512" 형태라서 '__' 앞까지만 사용
    return base.split("__")[0]


def find_label_files_by_filename(image_id: str, label_dirs: list[Path]):
    hits = []
    for d in label_dirs:
        fp = d / f"{image_id}.json"
        if fp.exists():
            hits.append(fp)
    return hits


def find_label_files_by_content(image_id: str, label_dirs: list[Path]):
    hits = []
    for d in label_dirs:
        if not d.exists():
            continue

        for fp in d.glob("*.json"):
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)

            sample_image_path = data["image"]
            sample_id = get_image_id(sample_image_path)
            if sample_id == image_id:
                hits.append(fp)

    return hits


def main():
    image_id = get_image_id(IMAGE_PATH)
    print(f"[QUERY] image_id = {image_id}")
    print(f"[QUERY] image_path = {IMAGE_PATH}")
    print("-" * 80)

    hits = find_label_files_by_filename(image_id, LABEL_DIRS)
    if len(hits) > 0:
        print("[HIT] matched by filename:")
        for fp in hits:
            print(str(fp))
        return

    hits = find_label_files_by_content(image_id, LABEL_DIRS)
    if len(hits) > 0:
        print("[HIT] matched by json content (data['image']):")
        for fp in hits:
            print(str(fp))
        return

    print("[MISS] no matching label json found in:")
    for d in LABEL_DIRS:
        print(str(d))

    print(IMAGE_PATH)


if __name__ == "__main__":
    main()
