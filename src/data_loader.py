import pandas as pd
import os

def load_data():
    path = os.path.join("..", "data", "Fakeddit", "train.tsv")

    df = pd.read_csv(path, sep="\t")

    print("✅ Data Loaded")

    # keep only useful columns
    df = df[["clean_title", "2_way_label"]]

    # remove null values
    df = df.dropna()

    print("\nCleaned Data:")
    print(df.head())

    return df


if __name__ == "__main__":
    load_data()