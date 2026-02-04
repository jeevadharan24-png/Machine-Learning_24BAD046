import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.ioff()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(
    r"C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp2\Dataset-1\bottle.csv",
    low_memory=False
)

print("Dataset Columns:\n", df.columns)

features = ['Depthm', 'Salnty', 'O2ml_L']
target = 'T_degC'
data = df[features + [target]].copy()
data = data.apply(pd.to_numeric, errors='coerce')
data.fillna(data.mean(), inplace=True)

X = data[features]
y = data[target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Performance")
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)

plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.tight_layout()
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(6,5))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Residual Errors")
plt.tight_layout()
plt.show()

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("Ridge R²:", r2_score(y_test, ridge_pred))

lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
print("Lasso R²:", r2_score(y_test, lasso_pred))
