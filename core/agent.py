# core/agent.py


from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from config.config import Config


class Agent(ABC):
    def __init__(self, config: Config):
        self.config = config

    def labeling(self, split: str, paths: list[Path]) -> None:
        cfg = self.config

        out_dir = self.get_out_dir(split)
        error_dir = self.get_error_dir(split)
        todo = self.get_todo(out_dir, paths)

        print(
            f"[{self.__class__.__name__}] {split} - Total: {len(paths)}, To do: {len(todo)}"
        )

        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            futures = {ex.submit(self.process_file, split, p): p for p in todo}

            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    result = fut.result()
                    self.write_output(out_dir, result, p)
                except Exception as e:
                    self.write_error(error_dir, e, p)

    @abstractmethod
    def process_file(self, split: str, path: Path) -> dict:
        """이미지 1장 처리(파이프라인 로직은 여기서 결정)"""
        raise NotImplementedError

    # ---- 공통 IO ----
    def get_out_dir(self, split: str) -> Path:
        cfg = self.config
        out_dir = (
            Path(cfg.runs) / cfg.dataset.value / cfg.agent.value / split / "outputs"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def get_error_dir(self, split: str) -> Path:
        cfg = self.config
        error_dir = (
            Path(cfg.runs) / cfg.dataset.value / cfg.agent.value / split / "errors"
        )
        error_dir.mkdir(parents=True, exist_ok=True)
        return error_dir

    def already_done(self, out_dir: Path, filename: str) -> bool:
        return (out_dir / filename).exists()

    def get_todo(self, out_dir: Path, paths: list[Path]) -> list[Path]:
        return [p for p in paths if not self.already_done(out_dir, p.name + ".json")]

    def write_output(self, out_dir: Path, result: dict, image_path: Path) -> None:
        out_path = out_dir / f"{image_path.name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def write_error(self, error_dir: Path, error: Exception, image_path: Path) -> None:
        error_path = error_dir / f"{image_path.stem}.error.json"
        payload = {"image_path": str(image_path), "error_msg": repr(error)}
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
