# ==========================================
# SCENARIO 2: POLYNOMIAL REGRESSION
# Roll No: <YOUR_ROLL_NUMBER>
# ==========================================

# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load dataset
df = pd.read_csv(r"C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp3\Dataset-2\auto-mpg.csv")

# 3. Clean data
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# 4. Handle missing values
imputer = SimpleImputer(strategy='mean')
df[['horsepower']] = imputer.fit_transform(df[['horsepower']])

# 5. Select features
X = df[['horsepower']]
y = df['mpg']

# 6. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Store errors
degrees = [2, 3, 4]
results = {}

# 8–11. Train Polynomial Models
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[d] = (mse, rmse, r2)

    print(f"Degree {d} -> MSE: {mse}, RMSE: {rmse}, R2: {r2}")

# 12. Ridge Regression to control overfitting
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_poly, y_train)

# ====================
# Visualization
# ====================

# Polynomial curve fitting
X_range = np.linspace(X.min(), X.max(), 100)
X_range_scaled = scaler.transform(X_range)

plt.scatter(X, y, label="Actual Data")

for d in degrees:
    poly = PolynomialFeatures(d)
    X_poly_range = poly.fit_transform(X_range_scaled)
    model = LinearRegression()
    model.fit(poly.fit_transform(X_train), y_train)
    y_curve = model.predict(X_poly_range)
    plt.plot(X_range, y_curve, label=f"Degree {d}")

plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.title("Polynomial Regression Curve Fitting")
plt.show()
