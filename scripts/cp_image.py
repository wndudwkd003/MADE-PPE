import sys
import shutil
import pathlib

def copy_and_count():
    if len(sys.argv) < 3:
        print("Usage: python scripts/cp_image.py <split_name> <image_path>")
        return

    split_name = sys.argv[1]
    img_path = pathlib.Path(sys.argv[2])

    base_dest_dir = pathlib.Path('datasets/sh17-100')
    train_dir = base_dest_dir / 'train'
    valid_dir = base_dest_dir / 'valid'
    target_dir = base_dest_dir / split_name

    if not img_path.exists():
        print(f"Error: {img_path} 파일을 찾을 수 없습니다.")
        return

    # 1. 교차 중복 체크
    exists_in_train = (train_dir / img_path.name).exists()
    exists_in_valid = (valid_dir / img_path.name).exists()

    if exists_in_train or exists_in_valid:
        location = "train" if exists_in_train else "valid"
        print(f"Warning: '{img_path.name}' 파일이 이미 {location} 폴더에 존재합니다. 복사를 중단합니다.")
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, target_dir / img_path.name)
        print(f"Copied {img_path.name} to {target_dir}")

    # 2. 현재 split 통합 카운트 출력
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    # 조건에 맞는 파일만 리스트로 담아 길이를 측정
    total_count = len([
        f for f in target_dir.rglob('*')
        if f.is_file() and f.suffix.lower() in extensions
    ])

    print(f"\n[{split_name}] current count: {total_count}")

if __name__ == "__main__":
    copy_and_count()
