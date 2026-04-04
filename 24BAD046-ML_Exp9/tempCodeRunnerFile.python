# ===== 24BAD046 – Scenario 2 (Item-Based + Comparison FINAL) =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Load dataset
ratings = pd.read_csv(r'C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp9\ratings.csv')
movies = pd.read_csv(r'C:\Users\Jeevadharan\Documents\ML\24BAD046-ML_Exp9\movies.csv')

# 2. Merge
data = ratings.merge(movies[['movieId', 'title']], on='movieId')

# ================= USER-BASED =================
user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

user_sim = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_top_users(user_id, n=5):
    return user_sim_df[user_id].drop(user_id).nlargest(n)

def predict_rating(user_id, movie, n=5):
    similar_users = get_top_users(user_id, n)
    ratings = user_item_matrix.loc[similar_users.index, movie]
    
    mask = ratings > 0
    if mask.sum() == 0:
        return 0
    
    return np.dot(similar_users[mask], ratings[mask]) / similar_users[mask].sum()

def recommend_movies(user_id, n=5):
    user_data = user_item_matrix.loc[user_id]
    unrated = user_data[user_data == 0].index
    
    scores = {movie: predict_rating(user_id, movie) for movie in unrated}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

# ================= ITEM-BASED =================

item_user_matrix = data.pivot_table(index='title', columns='userId', values='rating').fillna(0)

item_similarity = cosine_similarity(item_user_matrix)
item_similarity_df = pd.DataFrame(item_similarity,
                                 index=item_user_matrix.index,
                                 columns=item_user_matrix.index)

def get_similar_items(movie, n=5):
    return item_similarity_df[movie].drop(movie).nlargest(n)

def recommend_items(user_id, n=5):
    user_data = data[data['userId'] == user_id]
    liked = user_data[user_data['rating'] >= 4]['title']
    
    scores = {}
    for movie in liked:
        similar = get_similar_items(movie)
        for m, score in similar.items():
            if m not in liked.values:
                scores[m] = scores.get(m, 0) + score
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

print("Item-Based Recommendations:", recommend_items(1))

# ================= EVALUATION =================

# Train-Test Split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# RMSE
actual = []
pred = []

for _, row in test_data.sample(200).iterrows():
    user = row['userId']
    movie = row['title']
    
    try:
        prediction = predict_rating(user, movie)
        actual.append(row['rating'])
        pred.append(prediction)
    except:
        continue

rmse = np.sqrt(mean_squared_error(actual, pred))
print("RMSE:", rmse)

# Precision@K (FIXED)
def precision_at_k(user_id, k=5):
    test_user = test_data[test_data['userId'] == user_id]
    relevant = set(test_user[test_user['rating'] >= 3]['title'])
    
    if len(relevant) == 0:
        return 0

    recs = recommend_items(user_id, n=20)
    rec_movies = [m[0] for m in recs]

    rec_movies = [m for m in rec_movies if m in relevant]

    if len(rec_movies) == 0:
        return 0

    rec_movies = rec_movies[:k]

    hits = len(set(rec_movies) & relevant)
    return hits / k

print("Precision@K:", precision_at_k(1, k=5))

# ================= VISUALIZATIONS =================

# 1. Item Similarity Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(item_similarity_df.iloc[:20,:20], cmap='magma')
plt.title("Item Similarity Matrix")
plt.show()

# 2. Top Similar Items Graph
sample_movie = item_user_matrix.index[0]
sim_items = get_similar_items(sample_movie)

plt.figure(figsize=(8,5))
plt.barh(sim_items.index[::-1], sim_items.values[::-1])
plt.title(f"Top Similar Items for: {sample_movie}")
plt.xlabel("Similarity Score")
plt.show()

# 3. Recommendation Comparison Chart

user_recs = recommend_movies(1, n=5)
item_recs = recommend_items(1, n=5)

df_user = pd.DataFrame(user_recs, columns=['Movie', 'Score'])
df_user['Method'] = 'User-Based'

df_item = pd.DataFrame(item_recs, columns=['Movie', 'Score'])
df_item['Method'] = 'Item-Based'

df_compare = pd.concat([df_user, df_item])

plt.figure(figsize=(10,6))
sns.barplot(data=df_compare, x='Score', y='Movie', hue='Method')

plt.title("Recommendation Comparison (User vs Item)")
plt.xlabel("Score")
plt.ylabel("Movies")
plt.legend()
plt.show()