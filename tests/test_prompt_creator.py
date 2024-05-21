from src.modules.prompt_creator import PromptCreator


def test_create_prompts():
    prompt_creator = PromptCreator()
    amenities = ["pool", "gym", "wifi"]
    pos_prefix = "there is "
    pos_prompts, neg_prompts = prompt_creator.create_prompts(
        amenities=amenities, pos_prefix=pos_prefix
    )

    assert pos_prompts == ["there is pool", "there is gym", "there is wifi"]
    assert neg_prompts == ["there is no pool", "there is no gym", "there is no wifi"]
