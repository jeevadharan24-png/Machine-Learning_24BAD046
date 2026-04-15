
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

# 2. Load dataset
df = pd.read_csv(r'C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp10\ratings.csv')

# 3. Create User-Item matrix
user_item = df.pivot(index='userId', columns='movieId', values='rating')

# 4. Handle missing values (NMF requires non-negative values → use 0)
R = user_item.fillna(0).values

# 5. Apply NMF
nmf = NMF(n_components=20, random_state=42, max_iter=200)
W = nmf.fit_transform(R)   # User-feature matrix
H = nmf.components_        # Item-feature matrix

# 6. Reconstruct matrix
R_nmf = np.dot(W, H)

# 7. Convert to DataFrame
R_nmf_df = pd.DataFrame(R_nmf, index=user_item.index, columns=user_item.columns)

# ------------------ EVALUATION (FIXED) ------------------ #

# Only evaluate on known ratings (ignore missing values)
actual = user_item.values[user_item.notna().values]
predicted = R_nmf[user_item.notna().values]

rmse_nmf = np.sqrt(mean_squared_error(actual, predicted))
print("RMSE (NMF):", rmse_nmf)

# ------------------ VISUALIZATION ------------------ #

# 1. Latent feature visualization (User features)
plt.imshow(W[:20, :10])
plt.title("User Latent Features")
plt.colorbar()
plt.show()

# 2. Reconstruction comparison (sample)
plt.imshow(R_nmf[:20, :20])
plt.title("NMF Reconstructed Matrix")
plt.colorbar()
plt.show()

# 3. Recommendation ranking chart
user_id = 1

user_ratings = user_item.loc[user_id]
predictions = R_nmf_df.loc[user_id]

# Recommend only unseen movies
unseen_movies = user_ratings[user_ratings.isna()].index

top_items = predictions[unseen_movies].sort_values(ascending=False).head(10)

top_items.plot(kind='bar')
plt.title("Top Recommendations (NMF)")
plt.xlabel("Movie ID")
plt.ylabel("Predicted Rating")
plt.show()

# Print recommendations
print("\nTop 10 Recommended Movies for User 1:\n")
print(top_items)