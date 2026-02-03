# utils/api_utils.py

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI


def set_api_key(api_json: str) -> None:
    api_json = str(api_json)
    with open(api_json, "r", encoding="utf-8") as f:
        api_keys = json.load(f)

    for i, (key, value) in enumerate(api_keys.items()):
        os.environ[key] = value
        print(f"[{i+1}/{len(api_keys)}] Set env: {key}")

    print("All API keys have been set.")


class OpenAIAPI:

    def __init__(self):
        self.client = OpenAI()

    # 이미지 파일 업로드 하는거라고 함
    def upload_image_for_vision(self, image_path: str | Path) -> str:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")

        with image_path.open("rb") as f:
            uploaded = self.client.files.create(file=f, purpose="vision")

        return uploaded.id

    def upload_batch_input_jsonl(self, jsonl_path: str | Path) -> str:
        """
        배치 입력 .jsonl 파일을 업로드하고 input_file_id(file_...)를 반환합니다.
        purpose="batch" 사용. :contentReference[oaicite:2]{index=2}
        """
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"jsonl not found: {jsonl_path}")
        if jsonl_path.suffix.lower() != ".jsonl":
            raise ValueError(f"batch input must be .jsonl: {jsonl_path}")

        with jsonl_path.open("rb") as f:
            uploaded = self.client.files.create(file=f, purpose="batch")

        return uploaded.id

    def download_file_bytes(self, file_id: str) -> bytes:
        """
        Files API content 다운로드(바이너리).
        문서 예시: client.files.content(file_id).read() :contentReference[oaicite:3]{index=3}
        """
        resp = self.client.files.content(file_id)
        return resp.read()

    def download_file_text(self, file_id: str, encoding: str = "utf-8") -> str:
        return self.download_file_bytes(file_id).decode(encoding, errors="replace")

    # -------------------------
    # Batch API
    # -------------------------
    def create_batch(
        self,
        input_file_id: str,
        endpoint: str = "/v1/responses",
        completion_window: str = "24h",
        metadata: dict[str, Any] | None = None,
    ):
        """
        배치 생성.
        endpoint는 문서에 나온 지원 목록 중 하나여야 합니다. :contentReference[oaicite:4]{index=4}
        """
        if endpoint not in {
            "/v1/responses",
            "/v1/chat/completions",
            "/v1/embeddings",
            "/v1/completions",
            "/v1/moderations",
        }:
            raise ValueError(f"unsupported endpoint for batch: {endpoint}")

        return self.client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
        )

    def retrieve_batch(self, batch_id: str):
        return self.client.batches.retrieve(batch_id)

    def wait_batch(
        self,
        batch_id: str,
        poll_sec: float = 5.0,
    ):
        """
        배치가 완료/실패/만료/취소 상태가 될 때까지 폴링합니다.
        """
        terminal = {"completed", "failed", "expired", "cancelled"}

        while True:
            b = self.retrieve_batch(batch_id)
            status = getattr(b, "status", None)
            if status in terminal:
                return b
            time.sleep(poll_sec)

    # -------------------------
    # JSONL helpers
    # -------------------------
    @staticmethod
    def write_jsonl(lines: Iterable[dict[str, Any]], out_path: str | Path) -> Path:
        """
        dict 라인들을 .jsonl로 저장합니다.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
            for obj in lines:
                f.write(json.dumps(obj, ensure_ascii=False))
                f.write("\n")
        return out_path

    @staticmethod
    def parse_jsonl(text: str) -> list[dict[str, Any]]:
        """
        output/error jsonl 텍스트를 라인별 JSON으로 파싱합니다.
        """
        out: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out

    # -------------------------
    # Batch request line builders
    # -------------------------
    @staticmethod
    def build_responses_batch_line(
        custom_id: str,
        model: str,
        prompt_text: str,
        image_file_id: str,
        detail: str = "low",
        max_output_tokens: int = 800,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Batch 입력 jsonl의 1줄(요청 1개)을 /v1/responses 용으로 구성합니다.
        body는 Responses API 파라미터와 동일 구조를 사용합니다. :contentReference[oaicite:5]{index=5}
        """
        body: dict[str, Any] = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {
                            "type": "input_image",
                            "file_id": image_file_id,
                            "detail": detail,
                        },
                    ],
                }
            ],
            "max_output_tokens": max_output_tokens,
        }

        if extra_body:
            # 예: {"temperature": 0, "response_format": {...}} 등을 추가로 주입
            body.update(extra_body)

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
