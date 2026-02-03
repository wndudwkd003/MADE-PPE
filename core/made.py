# core/made.py

from pathlib import Path
from core.agent import Agent


class MADE(Agent):
    def build_line(
        self, custom_id: str, image_file_id: str, image_path: Path, split: str
    ) -> dict:
        cfg = self.config

        prompt_text = self.build_prompt(image_path)  # 이 클래스에서 프롬프트 구성

        return self.api.build_responses_line(
            custom_id=custom_id,
            model=cfg.model.value,
            prompt_text=prompt_text,
            image_file_id=image_file_id,
            detail="low",
            max_output_tokens=800,
            extra_body=None,
        )

    def build_prompt(self, image_path: Path) -> str:
        # MADE 전체 프롬프트(5단계 + proposer/rebutter/judge를 한 요청에 다 담을지)
        # 일단은 placeholder
        return "..."
