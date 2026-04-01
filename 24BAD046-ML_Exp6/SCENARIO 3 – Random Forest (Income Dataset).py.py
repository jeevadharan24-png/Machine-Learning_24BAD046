import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder  

df = pd.read_csv(r"C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp6\income_random_forest.csv")

# Encode categorical data
df = df.apply(LabelEncoder().fit_transform)
X = df.drop("Income", axis=1)
y = df["Income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scores = []
trees = [10, 50, 100]

for n in trees:
    rf = RandomForestClassifier(n_estimators=n)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    scores.append(accuracy_score(y_test, pred))

# Graph
plt.plot(trees, scores, marker='o')
plt.title("Accuracy vs Number of Trees")
plt.xlabel("Trees")
plt.ylabel("Accuracy")
plt.show()

# Feature importance
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

plt.bar(X.columns, rf.feature_importances_)
plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.show()