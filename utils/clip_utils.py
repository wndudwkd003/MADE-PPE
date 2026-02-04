import torch
import open_clip
from PIL import Image

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

    # --- encode image ---
    image_input = preprocess(image).unsqueeze(0).to(device)  # (1,3,H,W)
    with torch.no_grad():
        image_feat = model.encode_image(image_input)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)  # normalize

    # --- encode text ---
    text_tokens = tokenizer(texts).to(device)  # (N, ctx_len)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    # cosine similarity (N,)
    sims = (image_feat @ text_feat.T).squeeze(0)  # (N,)
    return sims.detach().cpu().tolist()
