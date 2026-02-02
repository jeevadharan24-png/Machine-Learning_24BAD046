import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
df = pd.read_csv(
    r"C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp2\Dataset-2\LICI - 10 minute data.csv"
)
print("Dataset Columns:\n", df.columns)

open_col = next(col for col in df.columns if 'open' in col.lower())
close_col = next(col for col in df.columns if 'close' in col.lower())

print("Using Open :", open_col)
print("Using Close:", close_col)
df['Price_Movement'] = np.where(df[close_col] > df[open_col], 1, 0)
feature_candidates = ['open', 'high', 'low', 'volume']
features = [
    col for col in df.columns
    if any(key in col.lower() for key in feature_candidates)
]
print("Selected Features:", features)
data = df[features + ['Price_Movement']].copy()
data = data.apply(pd.to_numeric, errors='coerce')
data.fillna(data.mean(), inplace=True)
X = data[features]
y = data['Price_Movement']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Performance")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-Score :", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()