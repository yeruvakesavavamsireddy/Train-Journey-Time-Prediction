import pandas as pd

def load_data():

    df = pd.read_csv("data/raw/train_routes.csv")

    print("Dataset Shape:", df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nMissing Values:")
    print(df.isnull().sum())

    return df