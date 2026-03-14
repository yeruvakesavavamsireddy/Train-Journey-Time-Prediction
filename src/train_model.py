import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from data_loader import load_data
from preprocessing import clean_data
from feature_engineering import create_features

df = load_data()

df = clean_data(df)

df = create_features(df)

X = df[["distance", "stops"]]
y = df["journey_duration"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

joblib.dump(model, "models/linear_regression_model.pkl")

print("Model trained and saved.")