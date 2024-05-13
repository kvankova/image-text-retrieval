from typing import Any, List, Tuple

import numpy as np
from transformers import CLIPModel, CLIPProcessor  # type: ignore

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
amenities = [
    "pool",
    "balcony",
    "microwave",
    "TV",
    "shower",
    "bathtub",
    "coffee machine",
    "washing machine",
]


def create_prompts(amenities: List[str]) -> Tuple[List[str], List[str]]:
    amenities_pos_prompts = ["there is " + i for i in amenities]
    amenities_neg_prompts = ["there is no " + i for i in amenities]

    return amenities_pos_prompts, amenities_neg_prompts


def detect_amenity(
    image: Any, pos_prompt: str, neg_prompt: str
) -> Tuple[str, np.ndarray]:

    inputs = processor(
        text=[pos_prompt] + [neg_prompt],
        images=image,
        return_tensors="pt",
        padding=True,
    )

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    probs = probs.detach().numpy().squeeze()

    predicted_prompt = pos_prompt if probs[0] > probs[1] else neg_prompt

    return predicted_prompt, probs
