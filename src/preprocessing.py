import pandas as pd

def clean_data(df):

    df = df.drop_duplicates()

    df = df.dropna()

    df.to_csv("data/processed/cleaned_data.csv", index=False)

    return df