def create_features(df):

    df["distance_per_stop"] = df["distance"] / (df["stops"] + 1)

    return df