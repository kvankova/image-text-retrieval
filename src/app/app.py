import streamlit as st
from box import Box
from PIL import Image

from src.modules.model import CLIP
from src.modules.prompt_creator import PromptCreator

config = Box.from_yaml(filename="config.yaml")

amenities = config.amenities

st.title("Amenity Detection App")
amenities = st.multiselect("Select Amenities", amenities)
uploaded_file = st.file_uploader(":red[Choose an image...]")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    if st.button("Detect"):
        model = CLIP()
        prompt_creator = PromptCreator()
        pos_prompt, neg_prompt = prompt_creator.create_prompts(
            amenities=amenities, pos_prefix="there is"
        )
        st.header("Results")

        for pos, neg in zip(pos_prompt, neg_prompt):
            probs = model.predict(image=image, prompts=[pos, neg])

            if probs[0] > probs[1]:
                predicted_prompt = pos
            else:
                predicted_prompt = neg

            st.write(
                f":red[**{predicted_prompt}**] with probability :red[{round(100*probs.max(), 2)} %]"
            )
