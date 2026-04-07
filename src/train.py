import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# load data
path = os.path.join("..", "data", "Fakeddit", "train.tsv")
df = pd.read_csv(path, sep="\t")

# keep useful columns
df = df[["clean_title", "2_way_label"]]
df = df.dropna()

X = df["clean_title"]
y = df["2_way_label"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# convert text → numbers
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# predict
y_pred = model.predict(X_test_vec)

# evaluate
acc = accuracy_score(y_test, y_pred)

print("✅ Model Trained")
print("Accuracy:", acc)