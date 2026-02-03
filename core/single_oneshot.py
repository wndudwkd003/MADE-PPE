# core/single_oneshot.py
from pathlib import Path
from core.agent import Agent


class SingleOneShot(Agent):
    def process_file(self, split: str, path: Path) -> dict:
        # 프롬프트 로딩(이 클래스에서만)
        # 단일 호출로 work_situation/hazard/compliance/wearing/improper_wearing 전부 출력
        out = self.call_model_oneshot(path)
        return self.pack_result(path, split, out)

    def call_model_oneshot(self, path: Path) -> dict: ...
    def pack_result(self, path: Path, split: str, out: dict) -> dict: ...
