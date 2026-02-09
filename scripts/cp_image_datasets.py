import shutil
from pathlib import Path

def merge_datasets():
    # 설정
    source_names = ['SH17-100', 'cppe-5-100', 'PDY-100']
    base_dir = Path('datasets')
    target_root = base_dir / 'SCP300'
    splits = ['train', 'valid']

    for name in source_names:
        src_dataset_path = base_dir / name

        for split in splits:
            src_split_dir = src_dataset_path / split
            target_split_dir = target_root / split

            if not src_split_dir.exists():
                print(f"Warning: {src_split_dir} 경로가 존재하지 않습니다.")
                continue

            # 대상 폴더 생성
            target_split_dir.mkdir(parents=True, exist_ok=True)

            # 파일 복사 및 이름 변경
            for img_path in src_split_dir.iterdir():
                if img_path.is_file():
                    # 새 이름: 데이터셋이름_원래이름.확장자
                    new_filename = f"{name}_{img_path.name}"
                    target_path = target_split_dir / new_filename

                    # 복사 (메타데이터 유지)
                    shutil.copy2(img_path, target_path)

    print(f"작업 완료: {target_root} 경로를 확인하세요.")

if __name__ == "__main__":
    merge_datasets()
