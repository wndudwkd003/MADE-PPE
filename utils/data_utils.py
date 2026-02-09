# utils/data_utils.py

from pathlib import Path
from params.params import DatasetEnum


def sh17_process(dataset_dir: Path) -> dict:

    train_files_txt = dataset_dir / "train_files.txt"
    val_files_txt = dataset_dir / "val_files.txt"
    images_dir = dataset_dir / "images"

    def _read_list(txt_path: Path) -> list[str]:
        with txt_path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    train_names = _read_list(train_files_txt)
    valid_names = _read_list(val_files_txt)

    train_paths = [images_dir / name for name in train_names]
    valid_paths = [images_dir / name for name in valid_names]

    missing = [p for p in (train_paths + valid_paths) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"[SH17] Listed image(s) not found under {images_dir}. "
            f"Missing examples: {missing[:10]}"
        )

    return {"train": train_paths, "valid": valid_paths, "test": []}


def scp300_process(dataset_dir: Path) -> dict:

    train = "train"
    valid = "valid"

    able_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def read_images(split_dir: Path):
        image_paths = []

        for ext in able_exts:
            image_paths.extend(split_dir.glob(f"*{ext}"))

        return image_paths

    train_paths = read_images(dataset_dir / train)
    valid_paths = read_images(dataset_dir / valid)

    return {"train": train_paths, "valid": valid_paths, "test": []}






def get_data(datasets_dir: str, dataset: DatasetEnum) -> dict:

    datasets_dir = Path(datasets_dir)
    dataset_dir = datasets_dir / dataset.value

    if dataset == DatasetEnum.SH17:
        return sh17_process(dataset_dir)

    if dataset == DatasetEnum.SCP300:
        return scp300_process(dataset_dir)

    raise NotImplementedError(f"get_data is not implemented for dataset={dataset}")
