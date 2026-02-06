from config.config import Config
from utils.clip_utils import blip_scores, clip_image_text_sims, gme_image_text_sims
from PIL import Image
from utils.seeds import set_seed

config = Config()

IMAGE_PATH = "runs/_cache_resized/pexels-photo-8961394__7913x5275__pad__512.jpg"

set_seed(42)

TEST = [
    "W05_POWDER_CEMENT_WOODWORKING_ABRASIVE_DUST",

    "this image contains the hazard of DUST",
    "this image contains the hazard of IMPACT",
    "this image contains the hazard of CUTTING",
    "this image contains the hazard of NOISE",

    "this image requires DUST_MASK for safety compliance",
    "this image requires A_TYPE_SAFETY_HELMET for safety compliance",
    "this image requires SAFETY_GOGGLES for safety compliance",
    "this image requires CUT_RESISTANT_GLOVES for safety compliance",
    "this image requires SAFETY_SHOES for safety compliance",
    "this image requires HEARING_PROTECTION for safety compliance",

    "a person is not wearing DUST_MASK",
    "a person is wearing A_TYPE_SAFETY_HELMET",
    "a person is not wearing SAFETY_GOGGLES",
    "a person is wearing CUT_RESISTANT_GLOVES",
    "a person is not wearing SAFETY_SHOES",
    "a person is not wearing HEARING_PROTECTION",

    "a person is wearing A_TYPE_SAFETY_HELMET properly",
    "a person is wearing SAFETY_GOGGLES properly",
    "a person is wearing CUT_RESISTANT_GLOVES properly",
]

image = Image.open(IMAGE_PATH).convert("RGB")

# 1) CLIP (open_clip)
clip_sims = clip_image_text_sims(
    image=image,
    texts=TEST,
    model_name=config.clip_vision,
    pretrained=config.clip_pretrained,
    device=config.device,
)

# 2) BLIP
combined, cos_s, p_match = blip_scores(
    image=image,
    texts=TEST,
    model_name="Salesforce/blip-itm-base-coco",
    device=config.device,
    use_itm=True,
)

# 3) GME (Qwen2-VL 기반 멀티모달 임베딩)
gme_model = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
gme_prompt = None  # 필요하면 "Find an image that matches the given text." 같은 프롬프트를 지정

gme_sims = gme_image_text_sims(
    image_path=IMAGE_PATH,
    texts=TEST,
    model_name=gme_model,
    device=config.device,
    prompt=gme_prompt,
)

print("idx\tCLIP\tBLIP(combined)\tBLIP(cos)\tBLIP(p_match)\tGME\ttext")
for i, (t, a_clip, a_blip, b_cos, c_itm, a_gme) in enumerate(
    zip(TEST, clip_sims, combined, cos_s, p_match, gme_sims)
):
    print(f"{i}\t{a_clip:.6f}\t{a_blip:.6f}\t{b_cos:.6f}\t{c_itm:.6f}\t{a_gme:.6f}\t{t}")

# ---- 지표별 평균 ----
clip_avg = sum(clip_sims) / len(clip_sims)
blip_combined_avg = sum(combined) / len(combined)
blip_cos_avg = sum(cos_s) / len(cos_s)
blip_p_match_avg = sum(p_match) / len(p_match)
gme_avg = sum(gme_sims) / len(gme_sims)

print()
print("=== Averages over TEST ===")
print(f"CLIP avg           : {clip_avg:.6f}")
print(f"BLIP combined avg  : {blip_combined_avg:.6f}")
print(f"BLIP cosine avg    : {blip_cos_avg:.6f}")
print(f"BLIP p_match avg   : {blip_p_match_avg:.6f}")
print(f"GME avg            : {gme_avg:.6f}")
