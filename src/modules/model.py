from typing import Any, List, Protocol

import numpy as np
from transformers import CLIPModel, CLIPProcessor  # type: ignore


class ModelInterface(Protocol):
    def predict(self, image: Any, prompts: List[str]) -> np.ndarray:
        pass


class CLIP:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def predict(self, image: Any, prompts: List[str]) -> np.ndarray:
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        probs = probs.detach().numpy().squeeze()

        return probs
