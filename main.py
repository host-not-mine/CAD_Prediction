import os
import random
import streamlit as st
from PIL import Image
from modeltest.test import ModelStore, test_image

@st.cache_resource
def load_once():
    models = ModelStore()
    models.load_model(1.08, "models/v1.08.ckpt", mask_limit=2)
    models.load_model(1.31, "models/v1.31.ckpt")
    models.set_default(1.31)
    print("Model loaded, Default set")
    return models
models = load_once()
examples_dir = "./examples"

st.title("CAD Segmentation Tool")
st.markdown(
    """
    **Upload an image** to test segmentation for Coronary Artery Disease (CAD). 
    This tool uses pre-trained model to generate segmentation mask.
    """
)
st.markdown(
    """
    You can find images [here](https://github.com/Sankalp-eldar/CAD_Prediction/tree/0139caf3b172c19a3589857aa36b5af542cedf63/examples)
    """
)

file = st.file_uploader("Select an image to upload (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

# "Load Demo Images" button
if st.button("Load Random Demo Images"):
    example_images = [img for img in os.listdir(examples_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(example_images)
    example_images = example_images[:9]

    # Display images in a grid with checkboxes for selection
    demo_col, close_col = st.columns(2, gap="large")
    demo_col.write("### Demo Images",)
    close_col.button("Close")

    columns = st.columns(3)
    for idx, image_name in enumerate(example_images):
        img_path = os.path.join(examples_dir, image_name)
        img = Image.open(img_path)
        with columns[idx % 3]:
            st.image(img, caption=image_name, use_column_width=True)

    # Run predictions on selected demo images
    st.write("Processing the images, please wait...")
    with st.spinner("Running segmentation model..."):
        st.write("### Predicted Segmentations")
        result_columns = st.columns(3)

        for idx, img_path in enumerate(example_images):
            img_path = os.path.join(examples_dir, img_path)
            pred = test_image(models.get_model(), img_path)  # Run model prediction

            img = Image.open(img_path).convert("RGBA")
            # Create a red overlay where the mask is present
            color_overlay = Image.new("RGBA", pred.size, (255, 0, 0, 128))  # Semi-transparent red
            mask_overlay = Image.composite(color_overlay, Image.new("RGBA", pred.size), pred)

            # Overlay mask on original image
            overlay_image = Image.alpha_composite(img, mask_overlay)

            with result_columns[idx % 3]:
                st.image(
                    overlay_image,
                    caption=f"Predicted {os.path.basename(img_path)}",
                    use_column_width=True
                    )

if file:
    try:
        st.write("Processing uploaded image, please wait...")
        with st.spinner("Running segmentation model..."):
            pred = test_image(models.get_model(), file)  # type: ignore
        st.success("Segmentation completed!")

        img_col, pred_col = st.columns(2)
        img = Image.open(file).convert("RGBA")
        img_col.write("Original image")
        img_col.image(img, caption="Uploaded Image")

        color_overlay = Image.new("RGBA", pred.size, (255, 0, 0, 128))  # Semi-transparent red
        mask_overlay = Image.composite(color_overlay, Image.new("RGBA", pred.size), pred)
        overlay_image = Image.alpha_composite(img, mask_overlay)

        pred_col.write("Predicted Segmentation")
        pred_col.image(overlay_image, "Predicted stenosis")

        st.button("Reload")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
else:
    st.info("Please upload an image to get started.")

# Optional: Provide options for users to select a model (if multiple models are available)
model_version = st.selectbox("Select model version", ["1.08", "1.31"], index=1)
if model_version == "1.08":
    models.set_default(1.08)
elif model_version == "1.31":
    models.set_default(1.31)

