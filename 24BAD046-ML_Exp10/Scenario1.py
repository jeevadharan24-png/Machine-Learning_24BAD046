
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 2. Load dataset
df = pd.read_csv(r'C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp10\ratings.csv')

# 3. Create User-Item matrix
user_item = df.pivot(index='userId', columns='movieId', values='rating')

# 4. Fill missing values (mean filling - better than 0)
user_item_filled = user_item.fillna(user_item.mean())

# 5. Convert to numpy
R = user_item_filled.values

# 6. Apply SVD
U, sigma, Vt = np.linalg.svd(R, full_matrices=False)

# 7. Reduce dimensions
k = 20
U_k = U[:, :k]
sigma_k = np.diag(sigma[:k])
Vt_k = Vt[:k, :]

# 8. Reconstruct matrix
R_pred = np.dot(np.dot(U_k, sigma_k), Vt_k)

# 9. Convert to DataFrame
R_pred_df = pd.DataFrame(R_pred, index=user_item.index, columns=user_item.columns)

# ------------------ EVALUATION (FIXED) ------------------ #

# Only compare known ratings (no NaN)
actual = user_item.values[user_item.notna().values]
predicted = R_pred_df.values[user_item.notna().values]

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)

print("RMSE:", rmse)
print("MAE:", mae)

# ------------------ VISUALIZATION ------------------ #

# 1. Heatmap (Original vs Reconstructed)
plt.imshow(R[:20, :20])
plt.title("Original Matrix (Sample)")
plt.colorbar()
plt.show()

plt.imshow(R_pred[:20, :20])
plt.title("Reconstructed Matrix (Sample)")
plt.colorbar()
plt.show()

# 2. Error vs Latent Factors
errors = []
k_values = [5, 10, 20, 30, 40]

for k in k_values:
    U_k = U[:, :k]
    sigma_k = np.diag(sigma[:k])
    Vt_k = Vt[:k, :]
    R_temp = np.dot(np.dot(U_k, sigma_k), Vt_k)

    actual_temp = user_item.values[user_item.notna().values]
    pred_temp = R_temp[user_item.notna().values]

    error = np.sqrt(mean_squared_error(actual_temp, pred_temp))
    errors.append(error)

plt.plot(k_values, errors, marker='o')
plt.xlabel("Latent Factors (k)")
plt.ylabel("RMSE")
plt.title("Error vs Latent Factors")
plt.show()

# 3. Top-N Recommendations
user_id = 1

user_ratings = user_item.loc[user_id]
predictions = R_pred_df.loc[user_id]

# Get only unseen movies
unseen_movies = user_ratings[user_ratings.isna()].index

# Recommend top 10
top_recommendations = predictions[unseen_movies].sort_values(ascending=False).head(10)

print("\nTop 10 Recommended Movies for User 1:\n")
print(top_recommendations)