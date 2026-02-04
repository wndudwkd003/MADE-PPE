# utils/api_utils.py

import os
import json
from pathlib import Path
from typing import Type, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel
from PIL import Image, ImageOps


def set_api_key(api_json: str) -> None:
    api_json = str(api_json)

    with open(api_json, "r", encoding="utf-8") as f:
        api_keys = json.load(f)

    for i, (key, value) in enumerate(api_keys.items()):
        os.environ[key] = value
        print(f"[{i + 1}/{len(api_keys)}] Set env: {key}")

    print("All API keys have been set.")


class OpenAIAPI:
    def __init__(self):
        self.client = OpenAI()

    def _resize_to_512(
        self,
        src_path: Path,
        out_path: Path,
        mode: str = "pad",   # "pad" | "crop" | "stretch"
        size: int = 512,
    ) -> Path:

        img = Image.open(src_path).convert("RGB")
        w, h = img.size

        # 512 이하이면 원본 그대로 사용(원하시면 여기서도 512로 맞추도록 바꿀 수 있음)
        if max(w, h) <= size and (w == size and h == size):
            img.save(out_path, format="JPEG", quality=95)
            return out_path

        if max(w, h) <= size and mode != "stretch":
            # 작은 이미지는 그대로 업로드해도 되지만,
            # "무조건 512x512" 원칙이면 아래처럼 처리하세요.
            pass

        if mode == "stretch":
            # 강제 512x512 (왜곡 가능)
            img = img.resize((size, size))
        elif mode == "crop":
            # 비율 유지 + 중앙 크롭
            img = ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        else:
            # mode == "pad": 비율 유지 + 패딩(왜곡 없음)
            img = ImageOps.pad(img, (size, size), method=Image.Resampling.LANCZOS, color=(0, 0, 0), centering=(0.5, 0.5))

        img.save(out_path, format="JPEG", quality=95)
        return out_path

    def upload_image(
        self,
        image_path: str | Path,
        resize_if_over: int = 512,
        resize_mode: str = "pad",
        cache_dir: str | Path = "runs/_cache_resized",
    ) -> str:

        p = Path(image_path)

        with Image.open(p) as img:
            w, h = img.size

        upload_path = p

        if max(w, h) > resize_if_over:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            # 파일명 충돌 방지(원본 stem + 원본 크기 + 모드)
            resized_path = cache_dir / f"{p.stem}__{w}x{h}__{resize_mode}__512.jpg"
            if not resized_path.exists():
                self._resize_to_512(p, resized_path, mode=resize_mode, size=512)

            upload_path = resized_path

        with upload_path.open("rb") as f:
            uploaded = self.client.files.create(file=f, purpose="vision")

        return uploaded.id

    def call_responses(
        self,
        model: str,
        system_text: str,
        user_text: str,
        image_file_id: str,
        text_format: Type[BaseModel],
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        retry_times: int,
    ):
        last_err = None
        for attempt in range(1, retry_times + 1):
            try:
                resp = self.client.responses.parse(
                    model=model,
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user_text},
                                {"type": "input_image", "file_id": image_file_id},
                            ],
                        },
                    ],
                    text_format=text_format,
                    max_output_tokens=max_output_tokens,
                    # temperature=temperature, gpt-5.1-mini 에서는 지원 안함
                    # top_p=top_p,
                )
                parsed_obj = resp.output_parsed
                if parsed_obj is None:
                    last_err = f"output_parsed is None (attempt {attempt}/{retry_times})"
                    continue
                return parsed_obj, None
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                continue
        return None, last_err
