import torch
import torch.nn as nn
import open_clip
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# LOAD DATA
# ------------------------
print("📂 Loading data...")
df = pd.read_csv("../data/Fakeddit/train.tsv", sep="\t")

# keep useful columns
df = df[['clean_title', '2_way_label', 'id']].dropna()

# 🔥 limit for speed
df = df.head(1000)

texts = df['clean_title'].values
labels = df['2_way_label'].values
ids = df['id'].astype(str).values

# ------------------------
# LOAD CLIP
# ------------------------
print("🖼 Loading CLIP...")
model_clip, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model_clip = model_clip.to(device)

# ------------------------
# FEATURE FUNCTIONS
# ------------------------
def get_image_feature(img_id):
    path = f"../data/Fakeddit/images/{img_id}.jpg"

    if not os.path.exists(path):
        return None

    try:
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model_clip.encode_image(image)

        return feat.cpu().numpy()[0]

    except:
        return None


def get_text_feature(text):
    tokens = open_clip.tokenize([text]).to(device)

    with torch.no_grad():
        feat = model_clip.encode_text(tokens)

    return feat.cpu().numpy()[0]


# ------------------------
# EXTRACT FEATURES
# ------------------------
print("⚡ Extracting multimodal features...")

X_img, X_txt, X_sim, y_clean = [], [], [], []

for i in tqdm(range(len(texts))):
    img_feat = get_image_feature(ids[i])

    if img_feat is None:
        continue  # skip missing images

    txt_feat = get_text_feature(texts[i])

    # cosine similarity
    sim = np.dot(img_feat, txt_feat) / (
        np.linalg.norm(img_feat) * np.linalg.norm(txt_feat) + 1e-8
    )

    X_img.append(img_feat)
    X_txt.append(txt_feat)
    X_sim.append([sim])
    y_clean.append(labels[i])

# convert to tensors
X_img = torch.tensor(np.array(X_img), dtype=torch.float32)
X_txt = torch.tensor(np.array(X_txt), dtype=torch.float32)
X_sim = torch.tensor(np.array(X_sim), dtype=torch.float32)
y = torch.tensor(np.array(y_clean))

# combine features
X = torch.cat([X_img, X_txt, X_sim], dim=1)

# ------------------------
# TRAIN TEST SPLIT
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# MODEL
# ------------------------
class MultimodalNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

model = MultimodalNet(X.shape[1]).to(device)

# ------------------------
# TRAIN
# ------------------------
print("🚀 Training...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

X_train, y_train = X_train.to(device), y_train.to(device)

for epoch in range(5):
    out = model(X_train)
    loss = criterion(out, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# ------------------------
# EVALUATION
# ------------------------
print("📊 Evaluating...")
X_test, y_test = X_test.to(device), y_test.to(device)

with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    acc = (preds == y_test).float().mean()

print("🔥 FINAL Multimodal Accuracy:", acc.item())