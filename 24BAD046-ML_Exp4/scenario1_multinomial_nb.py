import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

nltk.download('stopwords')
from nltk.corpus import stopwords

df = pd.read_csv(r'C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp4\scenario1\spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['message'] = df['message'].str.lower()
df['message'] = df['message'].str.translate(str.maketrans('', '', string.punctuation))

stop_words = set(stopwords.words('english'))
df['message'] = df['message'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])

le = LabelEncoder()
y = le.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

feature_names = vectorizer.get_feature_names_out()
spam_prob = model.feature_log_prob_[1]
top_spam_indices = np.argsort(spam_prob)[-10:]
top_spam_words = feature_names[top_spam_indices]

plt.barh(top_spam_words, spam_prob[top_spam_indices])
plt.title("Top Words Influencing Spam")
plt.show()

spam_words = df[df['label'] == 'spam']['message'].str.split(expand=True).stack()
ham_words = df[df['label'] == 'ham']['message'].str.split(expand=True).stack()

print("Top Spam Words:")
print(spam_words.value_counts().head(10))

print("Top Ham Words:")
print(ham_words.value_counts().head(10))

misclassified = X_test[y_test != y_pred]
print("Misclassified Samples:", misclassified.shape[0])

model_smooth = MultinomialNB(alpha=0.1)
model_smooth.fit(X_train, y_train)
y_pred_smooth = model_smooth.predict(X_test)

print("Accuracy after Laplace Smoothing:", accuracy_score(y_test, y_pred_smooth))