import streamlit as st
from PIL import Image
from modeltest.test import ModelStore, test_image

models = ModelStore()
# models.load_model(1.08, "models/v1.08.ckpt", mask_limit=2)
models.load_model(1.31, "models/v1.31.ckpt")

st.title("CAD Segmentation")

st.write("Upload An Image to test")


file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.write(img)

    pred = test_image(models.get_model(1.31), file) # type: ignore
    
    st.write("Predicts")
    st.write(pred)

