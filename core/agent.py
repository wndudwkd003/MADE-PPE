# core/agent.py

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import threading
import time
from tqdm.auto import tqdm

from config.config import Config
from utils.api_utils import OpenAIAPI


class Agent(ABC):
    def __init__(self, config: Config):
        self.config = config
        self._tls = threading.local()
        self._err_lock = threading.Lock()

    @property
    def api(self):
        if not hasattr(self._tls, "api"):
            self._tls.api = OpenAIAPI()
        return self._tls.api


    def get_agent_dir_name(self) -> str:
        cfg = self.config
        base_name = cfg.agent.value
        if cfg.explicit_target_tag != -1:
            return f"{base_name}{cfg.explicit_target_tag}"
        return base_name

    def labeling(self, split, paths):
        all_dir = self.get_outputs_all_dir(split)
        labels_dir = self.get_outputs_labels_dir(split)
        errors_jsonl = self.get_errors_jsonl_path(split)

        todo = self.get_todo(labels_dir, paths)
        print(f"[{self.__class__.__name__}] {split} - Total: {len(paths)}, To do: {len(todo)}")
        if not todo:
            return

        workers = self.config.workers
        print(f"[Parallel] workers={workers}")

        def worker(img_path: Path):
            file_id = None
            try:
                file_id = self.api.upload_image(img_path)
                return self.run_one_image(file_id, img_path, split, all_dir, labels_dir, errors_jsonl)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.append_error(errors_jsonl, {
                    "ts": time.time(),
                    "split": split,
                    "image_path": str(img_path),
                    "image_file_id": file_id,
                    "where": "Agent.labeling/worker",
                    "error": repr(e),
                })
                return None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(worker, p) for p in todo]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{split}", ncols=100):
                fut.result()  # 예외 전파

    @abstractmethod
    def run_one_image(self, image_file_id, image_path, split, all_dir, labels_dir, errors_jsonl):
        raise NotImplementedError

    # -----------------
    # dirs
    # -----------------
    def get_outputs_all_dir(self, split):
        cfg = self.config
        agent_dir = self.get_agent_dir_name()
        d = Path(cfg.runs) / cfg.dataset.value / agent_dir / split / "outputs" / "all"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_outputs_labels_dir(self, split):
        cfg = self.config
        agent_dir = self.get_agent_dir_name()
        d = Path(cfg.runs) / cfg.dataset.value / agent_dir / split / "outputs" / "labels"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_errors_jsonl_path(self, split):
        cfg = self.config
        agent_dir = self.get_agent_dir_name()
        d = Path(cfg.runs) / cfg.dataset.value / agent_dir / split / "errors"
        d.mkdir(parents=True, exist_ok=True)
        return d / "errors.jsonl"

    # -----------------
    # todo
    # -----------------
    def get_todo(self, labels_dir, paths):
        todo = [p for p in paths if not (labels_dir / f"{p.stem}.json").exists()]
        max_todo = self.config.max_todo
        if max_todo != -1:
            todo = todo[:max_todo]
        return todo

    # -----------------
    # errors.jsonl
    # -----------------
    def append_error(self, errors_jsonl, payload):
        line = json.dumps(payload, ensure_ascii=False)
        with self._err_lock:
            with open(errors_jsonl, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # -----------------
    # plain json write (atomic 제거)
    # -----------------
    def write_json(self, path: Path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
