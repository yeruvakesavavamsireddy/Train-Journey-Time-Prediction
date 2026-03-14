import joblib

model = joblib.load("models/linear_regression_model.pkl")

distance = float(input("Enter distance (km): "))
stops = float(input("Enter number of stops: "))

prediction = model.predict([[distance,stops]])

print("Predicted Journey Duration:",prediction[0],"hours")