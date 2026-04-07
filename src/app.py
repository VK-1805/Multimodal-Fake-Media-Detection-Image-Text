import streamlit as st
import torch
import open_clip
from PIL import Image
import numpy as np

# ------------------------
# SETUP
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("🧠 Fake News Detection (Multimodal)")
st.write("Upload an image + enter text → AI will predict FAKE or REAL")

# ------------------------
# LOAD MODEL
# ------------------------
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    return model.to(device), preprocess

model_clip, preprocess = load_model()

# ------------------------
# INPUTS
# ------------------------
uploaded_image = st.file_uploader("📷 Upload Image", type=["jpg", "png", "jpeg"])
text_input = st.text_area("📝 Enter News Text")

# ------------------------
# FEATURE FUNCTIONS
# ------------------------
def get_image_feature(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model_clip.encode_image(image)
    return feat.cpu().numpy()[0]

def get_text_feature(text):
    tokens = open_clip.tokenize([text]).to(device)
    with torch.no_grad():
        feat = model_clip.encode_text(tokens)
    return feat.cpu().numpy()[0]

# ------------------------
# PREDICTION
# ------------------------
if st.button("🔍 Predict"):

    if uploaded_image is None or text_input.strip() == "":
        st.warning("Please provide both image and text")
    else:
        image = Image.open(uploaded_image).convert("RGB")

        img_feat = get_image_feature(image)
        txt_feat = get_text_feature(text_input)

        # cosine similarity
        sim = np.dot(img_feat, txt_feat) / (
            np.linalg.norm(img_feat) * np.linalg.norm(txt_feat) + 1e-8
        )

        # simple decision rule
        if sim > 0.2:
            result = "REAL ✅"
        else:
            result = "FAKE 🚨"

        # ------------------------
        # OUTPUT UI
        # ------------------------
        st.image(image, caption="Uploaded Image", use_container_width=True)

        st.success(f"Prediction: {result}")
        st.metric("Similarity Score", f"{sim:.4f}")