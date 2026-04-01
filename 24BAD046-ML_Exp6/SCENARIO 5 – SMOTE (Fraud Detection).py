
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv(r"C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp6\fraud_smote.csv")

# Clean column names
df.columns = df.columns.str.strip()

#  FIXED TARGET COLUMN
X = df.drop("Fraud", axis=1)
y = df["Fraud"]

# 🔹 BEFORE SMOTE
print("Before SMOTE:\n", y.value_counts())

plt.figure()
y.value_counts().plot(kind='bar')
plt.title("Before SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Split (important)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔹 MODEL BEFORE SMOTE
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
pred_before = model.predict(X_test)

print("\nBefore SMOTE Report:\n", classification_report(y_test, pred_before))

# 🔹 APPLY SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# AFTER SMOTE
print("After SMOTE:\n", pd.Series(y_res).value_counts())

plt.figure()
pd.Series(y_res).value_counts().plot(kind='bar')
plt.title("After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# 🔹 MODEL AFTER SMOTE
model.fit(X_res, y_res)
pred_after = model.predict(X_test)

print("\nAfter SMOTE Report:\n", classification_report(y_test, pred_after))

# 🔹 PRECISION-RECALL CURVE
prob = model.predict_proba(X_test)[:,1]
precision, recall, _ = precision_recall_curve(y_test, prob)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()