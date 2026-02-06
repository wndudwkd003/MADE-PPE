import sys

import torch
import open_clip
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download


def clip_image_text_sims(
    image: Image.Image,
    texts: list[str],
    model_name: str,
    pretrained: str,
    device: str,
):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device).eval()

    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feat = model.encode_image(image_input)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

    text_tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    sims = (image_feat @ text_feat.T).squeeze(0)
    return sims.detach().cpu().tolist()


def blip_scores(
    image: Image.Image,
    texts: list[str],
    model_name: str,
    device: str,
    use_itm: bool = True,
):
    """
    BLIP 기반 스코어
    - ITC cosine similarity: retrieval projection 임베딩 코사인
    - use_itm=True: ITM match 확률(p_match)과 (cos + p)/2로 결합

    반환:
      - use_itm=False: list[float]  # cosine
      - use_itm=True: (list[float], list[float], list[float])
          (combined, cosine, p_match)
    """
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForImageTextRetrieval.from_pretrained(model_name).to(device).eval()

    inputs = processor(
        images=image,
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        # ---- ITC 임베딩 (retrieval projection space) ----
        vision_outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
        image_pool = vision_outputs.pooler_output
        image_feat = model.vision_proj(image_pool)

        text_outputs = model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_pool = text_outputs.pooler_output
        if text_pool is None:
            text_pool = text_outputs.last_hidden_state[:, 0, :]
        text_feat = model.text_proj(text_pool)

        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        cosine_scores = (text_feat @ image_feat.T).squeeze(-1)
        cosine_list = cosine_scores.detach().cpu().tolist()

        if not use_itm:
            return cosine_list

        # ---- ITM 확률 ----
        out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        if "itm_score" in out:
            itm_score = out["itm_score"]
        else:
            itm_score = out["logits"]

        p_match = torch.softmax(itm_score, dim=-1)[:, 1]
        p_match_list = p_match.detach().cpu().tolist()

        combined = (cosine_scores + p_match) * 0.5
        combined_list = combined.detach().cpu().tolist()

    return combined_list, cosine_list, p_match_list


gme_model_cache: dict[str, SentenceTransformer] = {}


def gme_image_text_sims(
    image_path: str,
    texts: list[str],
    model_name: str,
    device: str,
    prompt: str | None = None,
):
    """
    GME-Qwen2-VL 계열로 이미지-텍스트 임베딩 유사도(코사인)를 계산합니다.

    - image_path: 로컬 이미지 경로
    - texts: 후보 텍스트 리스트
    - prompt: (선택) query instruction. 예: 'Find an image that matches the given text.'
      prompt를 주면 텍스트 임베딩이 query 스타일로 생성됩니다.

    주의:
    - GME 리포지토리는 Sentence-Transformers 로딩 시 custom_st.py 같은 커스텀 모듈을 참조합니다.
      따라서 snapshot_download로 받은 로컬 디렉토리를 sys.path에 추가해 import 에러를 방지합니다.
    """
    if model_name in gme_model_cache:
        model = gme_model_cache[model_name]
    else:
        local_dir = snapshot_download(repo_id=model_name)
        if local_dir not in sys.path:
            sys.path.insert(0, local_dir)

        model = SentenceTransformer(model_name, device=device)
        gme_model_cache[model_name] = model

    if prompt is None:
        e_text = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    else:
        e_text = model.encode(
            [dict(text=t, prompt=prompt) for t in texts],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    e_img = model.encode(
        [dict(image=image_path)],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    sims = (e_text @ e_img.T).squeeze(1)
    return sims.detach().cpu().tolist()
