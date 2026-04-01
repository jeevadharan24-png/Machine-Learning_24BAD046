import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv(r"C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp6\diabetes_bagging.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Bagging
bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50)
bag.fit(X_train, y_train)
bag_pred = bag.predict(X_test)

# Accuracy
dt_acc = accuracy_score(y_test, dt_pred)
bag_acc = accuracy_score(y_test, bag_pred)

print("Decision Tree Accuracy:", dt_acc)
print("Bagging Accuracy:", bag_acc)

# Bar graph
plt.bar(["Decision Tree", "Bagging"], [dt_acc, bag_acc])
plt.title("Accuracy Comparison")
plt.show()

# Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, bag_pred)).plot()
plt.show()