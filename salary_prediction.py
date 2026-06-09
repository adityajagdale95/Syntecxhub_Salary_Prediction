
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("salary_data.csv")

# Features and Target
X = df[["Experience", "TestScore", "InterviewScore"]]
y = df["Salary"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Single Feature Model
single_X = df[["Experience"]]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    single_X, y,
    test_size=0.2,
    random_state=42
)

single_model = LinearRegression()
single_model.fit(X_train_s, y_train_s)

single_pred = single_model.predict(X_test_s)

single_r2 = r2_score(y_test_s, single_pred)

print("Single Feature R2 Score:", single_r2)

# Save Model
joblib.dump(model, "salary_model.pkl")

print("Model Saved Successfully")

# User Prediction
exp = int(input("Enter Experience: "))
test = int(input("Enter Test Score: "))
interview = int(input("Enter Interview Score: "))

predicted_salary = model.predict([[exp, test, interview]])

print("\n" + "="*40)
print("      SALARY PREDICTION")
print("="*40)
print(f"Predicted Salary : ₹{predicted_salary[0]:,.2f}")
print("="*40)


