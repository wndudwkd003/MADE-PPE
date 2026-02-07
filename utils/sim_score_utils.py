# utils/sim_score_utils.py

import torch
import open_clip
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval

from transformers import AutoModel


class ImageTextScorer:
    def __init__(
        self,
        clip_model_name: str,
        clip_pretrained: str,
        blip_model_name: str,
        gme_model_name: str,
        device: str,
    ):
        self.device = device

        # CLIP
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            model_name=clip_model_name,
            pretrained=clip_pretrained,
        )
        clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        self.clip_model = clip_model.to(device).eval()
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer

        # BLIP
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
        self.blip_model = BlipForImageTextRetrieval.from_pretrained(blip_model_name).to(device).eval()

        # GME
        self.gme_t2i_prompt = "Find an image that matches the given text."
        self.gme_model = AutoModel.from_pretrained(
            gme_model_name,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        self.gme_model.eval()

    def get_clip_score(self, image: Image.Image, texts: list[str],):
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feat = self.clip_model.encode_image(image_input)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        text_tokens = self.clip_tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_feat = self.clip_model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        sims = (image_feat @ text_feat.T).squeeze(0)
        return sims.detach().cpu().tolist()

    def get_blip_score(self, image: Image.Image, texts: list[str],):
        inputs = self.blip_processor(
            images=image,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            itm_out = self.blip_model(**inputs, return_dict=True)
            itm_score = itm_out["itm_score"]
            probs = torch.softmax(itm_score, dim=-1)
            p_match = probs[:, 1]

            itc_out = self.blip_model(**inputs, return_dict=True, use_itm_head=False)
            itc_score = itc_out["itm_score"][0]

        return p_match.detach().cpu().tolist(), itc_score.detach().cpu().tolist()

    def get_gme_score(self, image: Image.Image, texts: list[str],):
        images = [image]

        e_text = self.gme_model.get_text_embeddings(texts=texts)
        e_image = self.gme_model.get_image_embeddings(images=images)

        e_query = self.gme_model.get_text_embeddings(texts=texts, instruction=self.gme_t2i_prompt)
        e_corpus = self.gme_model.get_image_embeddings(images=images, is_query=False)

        gme_score = (e_text @ e_image.T).view(-1).tolist() # 항상 리스트
        gme_inst_score = (e_query @ e_corpus.T).view(-1).tolist()

        return gme_score, gme_inst_score
