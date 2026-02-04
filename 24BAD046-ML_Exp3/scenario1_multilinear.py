import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp3\Dataset-1\StudentsPerformance.csv")

label_encoder = LabelEncoder()
df['parental level of education'] = label_encoder.fit_transform(
    df['parental level of education']
)
df['test preparation course'] = label_encoder.fit_transform(
    df['test preparation course']
)
df['final_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
X = df[['parental level of education', 'test preparation course']]
y = df['final_score']
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
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
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

coeff_df = pd.DataFrame({
    'Feature': ['Parental Education', 'Test Preparation'],
    'Coefficient': lr.coef_
})
print(coeff_df)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Predicted vs Actual Scores")
plt.show()

coeff_df.set_index("Feature").plot(kind='bar')
plt.title("Regression Coefficient Magnitudes")
plt.show()

residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()
