import pandas as pd   
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv(r"C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp6\churn_boosting.csv")

# Encode categorical
df = df.apply(LabelEncoder().fit_transform)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# AdaBoost
ada = AdaBoostClassifier()
ada.fit(X_train, y_train)

# Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# ROC Curve
for model, name in [(ada, "AdaBoost"), (gb, "Gradient Boost")]:
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=name)

plt.legend()
plt.title("ROC Curve")
plt.show()

# Feature Importance (Gradient Boost)
plt.bar(X.columns, gb.feature_importances_)
plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.show()