from config.config import Config
from utils.sim_score_utils import ImageTextScorer
from PIL import Image
from utils.seeds import set_seed

config = Config()

IMAGE_PATH = "/workspace/MADE-PPE/runs/_cache_resized/pexels-photo-13944023__4160x2765__pad__512.jpg"

set_seed(42)

TEST = [
    "This work is welding work",
    "This work is a crushing work",
    "This work is cement work",
    "This work is a sewing work",

    "There requires a hard hat for this work",
    "There requires safety glasses for this work",
    "There requires a labtop for this work",

    "a person is wearing a safety glasses properly",
    "a person is not wearing a hard hat properly",

    "there is a hazard of falling objects",

    "there requires welding mask for this work",

    "there is a welder",
    "there is a labtop",
]

image = Image.open(IMAGE_PATH).convert("RGB")

scoerer = ImageTextScorer(
    device=config.device,
    clip_model_name=config.clip_model,
    clip_pretrained=config.clip_pretrained,
    blip_model_name=config.blip_model,
    gme_model_name=config.gme_model,
)

clip_score = scoerer.get_clip_score(image, TEST)
blip_itm_score, blip_itc_score = scoerer.get_blip_score(image, TEST)
gme_score, gme_inst_score = scoerer.get_gme_score(image, TEST)

print("CLIP Scores:", clip_score)
print("BLIP ITM Scores:", blip_itm_score)
print("BLIP ITC Scores:", blip_itc_score)
print("GME Scores:", gme_score)
print("GME Inst Scores:", gme_inst_score)


