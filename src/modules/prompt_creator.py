from typing import List, Protocol, Tuple


class PromptCreatorInterface(Protocol):
    def create_prompts(
        self, amenities: List[str], pos_prefix: str
    ) -> Tuple[List[str], List[str]]:
        pass


class PromptCreator:
    def create_prompts(
        self, amenities: List[str], pos_prefix: str
    ) -> Tuple[List[str], List[str]]:
        pos_prompts = [pos_prefix + i for i in amenities]
        neg_prompts = [pos_prefix + " no " + i for i in amenities]

        return pos_prompts, neg_prompts
