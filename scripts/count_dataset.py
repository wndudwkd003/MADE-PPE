import pathlib
from collections import Counter

def count_by_split(base_path):
    base_dir = pathlib.Path(base_path)
    splits = ['train', 'valid']
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for split in splits:
        split_path = base_dir / split
        if not split_path.exists():
            print(f"[{split}] 디렉토리를 찾을 수 없습니다.")
            continue

        # 해당 디렉토리 내 파일만 탐색
        files = split_path.rglob('*')
        counts = Counter(
            f.suffix.lower() for f in files
            if f.is_file() and f.suffix.lower() in extensions
        )

        print(f"[{split}]")
        if not counts:
            print("  파일 없음")
        for ext, count in counts.items():
            print(f"  {ext}: {count}")

if __name__ == "__main__":
    count_by_split('datasets/sh17-100')
