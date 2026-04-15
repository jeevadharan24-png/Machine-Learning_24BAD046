
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

# 6. Elbow Method
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 7. Apply K-Means (choose K = 5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 8. Assign cluster labels
df['Cluster'] = labels

# 9. Visualization (Clusters)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200, marker='X')
plt.title('K-Means Clustering')
plt.xlabel('Scaled Income')
plt.ylabel('Scaled Spending Score')
plt.show()

# 10. Visualization with centroids (fixed)
centroids = kmeans.cluster_centers_

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)
plt.title('Clusters with Centroids')
plt.show()

# Evaluation
print("Inertia:", kmeans.inertia_)
print("Silhouette Score:", silhouette_score(X_scaled, labels))

# Cluster interpretation
print(df.groupby('Cluster').mean())