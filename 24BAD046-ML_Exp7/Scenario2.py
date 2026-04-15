
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans   # ✅ FIXED (import added)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 2. Load dataset
df = pd.read_csv(r'C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp7\Mall_Customers.csv')

# 3. Data preprocessing
df = df.dropna()

# 4. Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 5. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------ GMM ------------------ #

# 6. Apply GMM
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(X_scaled)

# 7. Predict probabilities
probs = gmm.predict_proba(X_scaled)

# 8. Assign clusters
gmm_labels = gmm.predict(X_scaled)
df['GMM_Cluster'] = gmm_labels

# 9. Visualization (GMM)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels)
plt.title('GMM Clustering')
plt.xlabel('Scaled Income')
plt.ylabel('Scaled Spending Score')
plt.show()

# 10. Evaluation Metrics
print("Log-Likelihood:", gmm.score(X_scaled))
print("AIC:", gmm.aic(X_scaled))
print("BIC:", gmm.bic(X_scaled))
print("Silhouette Score (GMM):", silhouette_score(X_scaled, gmm_labels))

# ------------------ K-MEANS ------------------ #

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)

# ------------------ COMPARISON ------------------ #

plt.figure(figsize=(10, 5))

# K-Means plot
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_kmeans)
plt.title('K-Means')

# GMM plot
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels)
plt.title('GMM')

plt.show()

# 11. Cluster interpretation
print("\nCluster Means (GMM):\n")
print(df.groupby('GMM_Cluster').mean())

# 12. Show probabilities
print("\nCluster Probabilities (first 5 rows):\n")
print(probs[:5])