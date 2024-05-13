import streamlit as st
from PIL import Image

from src.utils import amenities, create_prompts, detect_amenity

st.title("CLIP model")
st.write("This is a simple app to detect amenities in an image.")
st.write("**Amenities to detect:** ")

amenities = st.multiselect("Select amenities to detect", amenities)

uploaded_file = st.file_uploader(":red[Choose an image...]")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    if st.button("Detect amenities"):
        pos_prompts, neg_prompts = create_prompts(amenities)

        for pos, neg in zip(pos_prompts, neg_prompts):
            predicted_prompt, probs = detect_amenity(
                image=image, pos_prompt=pos, neg_prompt=neg
            )
            st.write(
                f":red[**{predicted_prompt}**] with probability :red[{round(100*probs.max(), 2)} %]"
            )
