import streamlit as st
from PIL import Image
from modeltest.test import predict


st.title("CAD Segmentation")

st.write("Upload An Image")


file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.write(img)
    try:
        pred = predict("models/v1.08.ckpt", file)
    except RuntimeError:
        pred = predict("models/v1.08.ckpt", file, max_masks=2)
    st.write("Predicts")
    st.write(pred)

