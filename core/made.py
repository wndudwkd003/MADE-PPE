# core/made.py
from pathlib import Path
from core.agent import Agent


class MADE(Agent):
    def process_file(self, split: str, path: Path) -> dict:
        # 1) 프롬프트 로딩(이 클래스에서만)
        # 2) 단계별 3역할 실행
        ws = self.step_work_situation(path)  # proposer->rebutter->judge
        hz = self.step_hazard(path, ws)
        cp = self.step_compliance(path, ws, hz)
        wd = self.step_wearing(path, ws, hz, cp)
        iw = self.step_improper_wearing(path, ws, hz, cp, wd)

        # 3) (선택) schema proposal 집계 등
        return self.pack_result(path, split, ws, hz, cp, wd, iw)

    # 아래 step_* 들은 "3역할"을 내부에서 수행하도록 구현
    def step_work_situation(self, path: Path) -> dict: ...
    def step_hazard(self, path: Path, ws: dict) -> dict: ...
    def step_compliance(self, path: Path, ws: dict, hz: dict) -> dict: ...
    def step_wearing(self, path: Path, ws: dict, hz: dict, cp: dict) -> dict: ...
    def step_improper_wearing(
        self, path: Path, ws: dict, hz: dict, cp: dict, wd: dict
    ) -> dict: ...
    def pack_result(self, path: Path, split: str, *steps) -> dict: ...
