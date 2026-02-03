# core/agent.py

from abc import ABC, abstractmethod
from pathlib import Path
import json

from config.config import Config
from utils.api_utils import OpenAIAPI


class Agent(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.api = OpenAIAPI()  # 공통 API 유틸

    def labeling(self, split: str, paths: list[Path]) -> None:
        cfg = self.config

        out_dir = self.get_out_dir(split)
        error_dir = self.get_error_dir(split)
        todo = self.get_todo(out_dir, paths)

        print(
            f"[{self.__class__.__name__}] {split} - Total: {len(paths)}, To do: {len(todo)}"
        )

        if len(todo) == 0:
            return

        # 1) 이미지 업로드(file_id 확보) + batch line 생성
        lines = []
        for p in todo:
            file_id = self.api.upload_image(p)
            custom_id = self.make_custom_id(split, p)
            line = self.build_line(custom_id, file_id, p, split)
            lines.append(line)

        # 2) jsonl 저장 → 업로드 → batch 생성
        batch_input_path = (
            Path(cfg.runs)
            / cfg.dataset.value
            / cfg.agent.value
            / split
            / "batchinput.jsonl"
        )
        self.api.write_jsonl(lines, batch_input_path)
        input_file_id = self.api.upload_batch_jsonl(batch_input_path)

        batch = self.api.create_batch(
            input_file_id=input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "dataset": cfg.dataset.value,
                "agent": cfg.agent.value,
                "split": split,
            },
        )

        # 3) 완료 대기
        batch = self.api.wait_batch(batch.id, poll_sec=5.0)

        # 4) 결과 저장
        if batch.output_file_id:
            out_text = self.api.download_text(batch.output_file_id)
            rows = self.api.parse_jsonl(out_text)
            self.handle_batch_outputs(rows, out_dir, error_dir)

        # 5) 에러 파일도 있으면 저장(선택)
        if batch.error_file_id:
            err_text = self.api.download_text(batch.error_file_id)
            err_rows = self.api.parse_jsonl(err_text)
            self.handle_batch_errors(err_rows, error_dir)

    @abstractmethod
    def build_line(
        self, custom_id: str, image_file_id: str, image_path: Path, split: str
    ) -> dict:
        """이미지 1장에 대한 batch jsonl 라인 생성"""
        raise NotImplementedError

    def make_custom_id(self, split: str, image_path: Path) -> str:
        # 결과 매핑용: 충돌만 안 나면 됩니다.
        # split + stem 정도면 충분
        return f"{split}|{image_path.stem}"

    def handle_batch_outputs(
        self, rows: list[dict], out_dir: Path, error_dir: Path
    ) -> None:
        """
        rows: batch output jsonl의 각 라인
        - custom_id로 어떤 이미지인지 찾아서 저장해야 함
        - 여기서는 custom_id에서 stem을 꺼내서 저장한다고 가정
        """
        for r in rows:
            custom_id = r.get("custom_id", "")
            resp = r.get("response")
            err = r.get("error")

            # 응답이 있는데 status_code가 200이 아니면 에러 취급
            if err is not None or resp is None or resp.get("status_code") != 200:
                self.write_error(
                    error_dir,
                    Exception(json.dumps(r, ensure_ascii=False)),
                    Path(custom_id.split("|")[-1] + ".jpg"),
                )
                continue

            # assistant content 추출(/v1/responses라 가정)
            body = resp.get("body", {})
            # Responses API는 output_text가 있거나 output[*].content[*].text가 있을 수 있어 형태가 달라질 수 있음
            # -> 일단 raw로 저장하고 나중에 파싱
            result = {"custom_id": custom_id, "raw": body}

            # 파일명은 stem 기반으로
            stem = custom_id.split("|")[-1]
            out_path = out_dir / f"{stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    def handle_batch_errors(self, rows: list[dict], error_dir: Path) -> None:
        for r in rows:
            custom_id = r.get("custom_id", "unknown")
            error_path = error_dir / f"{custom_id.replace('|','_')}.batch_error.json"
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2)

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
        todo = [p for p in paths if not self.already_done(out_dir, p.name + ".json")]

        max_todo = self.config.max_todo

        if max_todo != -1:
            todo = todo[:max_todo]

        return todo

    def write_error(self, error_dir: Path, error: Exception, image_path: Path) -> None:
        error_path = error_dir / f"{image_path.stem}.error.json"
        payload = {"image_path": str(image_path), "error_msg": repr(error)}
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
