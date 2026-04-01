import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier

# Load dataset
df = pd.read_csv(r"C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp6\heart_stacking.csv")

# Clean column names
df.columns = df.columns.str.strip()

# j FIXED TARGET
target_col = "HeartDisease"

# Features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

print("Class Distribution:\n", y.value_counts())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
lr = LogisticRegression(max_iter=1000)
svm = SVC(probability=True)
dt = DecisionTreeClassifier()

stack = StackingClassifier(
    estimators=[('lr', lr), ('svm', svm), ('dt', dt)],
    final_estimator=LogisticRegression()
)

# Train
lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
dt.fit(X_train, y_train)
stack.fit(X_train, y_train)

# Accuracy
acc_lr = accuracy_score(y_test, lr.predict(X_test))
acc_svm = accuracy_score(y_test, svm.predict(X_test))
acc_dt = accuracy_score(y_test, dt.predict(X_test))
acc_stack = accuracy_score(y_test, stack.predict(X_test))

print("Logistic Regression:", acc_lr)
print("SVM:", acc_svm)
print("Decision Tree:", acc_dt)
print("Stacking:", acc_stack)

# Plot
plt.figure()
plt.bar(["LR","SVM","DT","Stack"], [acc_lr, acc_svm, acc_dt, acc_stack])
plt.title("Model Comparison")
plt.show()