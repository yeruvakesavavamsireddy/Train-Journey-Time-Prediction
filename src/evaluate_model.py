import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from data_loader import load_data

df = load_data()

X = df[["distance","stops"]]
y = df["journey_duration"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = joblib.load("models/linear_regression_model.pkl")

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test,predictions)

rmse = np.sqrt(mean_squared_error(y_test,predictions))

print("MAE:",mae)
print("RMSE:",rmse)