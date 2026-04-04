# ===== 24BAD046 – Scenario 1 =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. Load dataset
ratings = pd.read_csv(r'C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp9\ratings.csv')
movies = pd.read_csv(r'C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp9\movies.csv')

# 2. Merge
data = ratings.merge(movies[['movieId', 'title']], on='movieId')

# 3. Create User-Item Matrix
user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# 4. Handle missing values
user_item_filled = user_item_matrix.fillna(0)

# 5. Compute similarity
user_similarity = cosine_similarity(user_item_filled)
user_similarity_df = pd.DataFrame(user_similarity,
                                 index=user_item_filled.index,
                                 columns=user_item_filled.index)

# 6. Top similar users
def get_top_users(user_id, n=5):
    return user_similarity_df[user_id].drop(user_id).nlargest(n)

# 7. Predict rating
def predict_rating(user_id, movie, n=5):
    similar_users = get_top_users(user_id, n)
    ratings = user_item_filled.loc[similar_users.index, movie]
    
    mask = ratings > 0
    if mask.sum() == 0:
        return 0
    
    return np.dot(similar_users[mask], ratings[mask]) / similar_users[mask].sum()

# 8. Recommend movies
def recommend_movies(user_id, n=5):
    user_data = user_item_filled.loc[user_id]
    unrated = user_data[user_data == 0].index
    
    scores = {movie: predict_rating(user_id, movie) for movie in unrated}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

print("Recommendations:", recommend_movies(1))

# 9. Evaluation
train, test = train_test_split(data, test_size=0.2, random_state=42)

actual, pred = [], []
for _, row in test.sample(200).iterrows():
    try:
        p = predict_rating(row['userId'], row['title'])
        actual.append(row['rating'])
        pred.append(p)
    except:
        continue

print("RMSE:", np.sqrt(mean_squared_error(actual, pred)))
print("MAE :", mean_absolute_error(actual, pred))

# 10. Visualizations

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(user_item_matrix.iloc[:20,:20], cmap='viridis')
plt.title("User-Item Matrix")
plt.show()

# Similarity matrix
plt.figure(figsize=(10,6))
sns.heatmap(user_similarity_df.iloc[:20,:20], cmap='coolwarm')
plt.title("User Similarity Matrix")
plt.show()

# Recommendation chart
recs = recommend_movies(1)
movies_list = [m[0] for m in recs]
scores = [m[1] for m in recs]

plt.barh(movies_list[::-1], scores[::-1])
plt.title("Top Recommendations")
plt.xlabel("Predicted Rating")
plt.show()