import torch
import open_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Loading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model = model.to(device)

def extract_image_features(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image)

    return features


# ✅ TEST RUN
if __name__ == "__main__":
    test_image = "C:\\Users\\koushik\\OneDrive\\Desktop\\hakathon ML\\src\\sample.jpg"  # put any image here

    try:
        feats = extract_image_features(test_image)
        print("✅ Feature shape:", feats.shape)
    except Exception as e:
        print("❌ Error:", e)