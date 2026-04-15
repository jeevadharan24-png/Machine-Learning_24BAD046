
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 2. Load dataset (Iris)
data = load_iris()
X = data.data

# 3. Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 5. Explained variance
explained_variance = pca.explained_variance_ratio_

print("Explained Variance Ratio:\n", explained_variance)

# ------------------ VISUALIZATION ------------------ #

# 1. Scree Plot
plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
plt.xlabel('Principal Components')
plt.ylabel('Variance')
plt.title('Scree Plot')
plt.show()

# 2. Cumulative Variance Plot
cumulative_variance = explained_variance.cumsum()

plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
plt.xlabel('Components')
plt.ylabel('Cumulative Variance')
plt.title('Cumulative Variance')
plt.show()

# 3. 2D Scatter Plot (First 2 components)
pca_2 = PCA(n_components=2)
X_2D = pca_2.fit_transform(X_scaled)

plt.scatter(X_2D[:, 0], X_2D[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('2D PCA Projection')
plt.show()