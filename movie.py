

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title=" Movie Recommender", layout="wide")
st.title("🎬  Movie Recommendation System")

# -----------------------------
# 1️ Load Data
# -----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv(r"C:\Users\User\Desktop\movie\ml-32m\movies.csv"")      # movieId, title, genres, poster_url (new column)
    ratings = pd.read_csv(r"C:\Users\User\Desktop\movie\ml-32m\ratings.csv"")    # userId, movieId, rating
    return movies, ratings
movies_df, ratings_df = load_data()

top_users = ratings_df['userId'].value_counts().head(2000).index
top_movies = ratings_df['movieId'].value_counts().head(500).index
ratings_small = ratings_df[ratings_df['userId'].isin(top_users) & ratings_df['movieId'].isin(top_movies)]

st.subheader("Sample Ratings Data")
st.dataframe(ratings_df.head())

# -----------------------------
# 2 User-Movie Matrix
# -----------------------------
user_movie_matrix = ratings_small.pivot(index='userId', columns='movieId', values='rating').fillna(0)
movie_user_matrix = user_movie_matrix.T
# -----------------------------
# 3️ Item-Based Similarity
# -----------------------------
item_similarity = cosine_similarity(user_movie_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# -----------------------------
# 4️ Train SVD for AI Prediction
# -----------------------------
@st.cache_data
def train_svd(matrix, n_components=20):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    matrix_filled = svd.fit_transform(matrix)
    matrix_reconstructed = np.dot(matrix_filled, svd.components_)
    predicted_ratings = pd.DataFrame(matrix_reconstructed, index=matrix.index, columns=matrix.columns)
    return predicted_ratings

predicted_ratings_df = train_svd(user_movie_matrix)

# -----------------------------
# 5️ Hybrid Recommendation Function
# -----------------------------
def hybrid_recommend(user_id, top_n=5, genre_filter=None, rating_range=None, movie_search=None):
    user_pred = predicted_ratings_df.loc[user_id].sort_values(ascending=False)
    
    # Exclude already rated
    already_rated = user_movie_matrix.loc[user_id]
    unrated_movies = already_rated[already_rated == 0].index
    
    top_movies_ids = user_pred[unrated_movies].head(100).index  # take top 100 for filtering
    
    recommended = pd.DataFrame({
        'movieId': top_movies_ids,
        'predicted_rating': user_pred[top_movies_ids].values
    })
    recommended = recommended.merge(movies_df, on='movieId')
    
    # Apply filters
    if genre_filter:
        recommended = recommended[recommended['genres'].str.contains(genre_filter)]
    if rating_range:
        min_r, max_r = rating_range
        recommended = recommended[(recommended['predicted_rating'] >= min_r) & (recommended['predicted_rating'] <= max_r)]
    if movie_search:
        recommended = recommended[recommended['title'].str.contains(movie_search, case=False)]
    
    return recommended.head(top_n)

# -----------------------------
# 6️ Streamlit UI
# -----------------------------
st.sidebar.header("User Options")
user_list = list(user_movie_matrix.index)
selected_user = st.sidebar.selectbox("Choose User ID:", user_list)
top_n = st.sidebar.slider("Number of Recommendations:", min_value=1, max_value=20, value=5)
genre_filter = st.sidebar.selectbox("Filter by Genre (Optional):", options=[""] + sorted(list(set('|'.join(movies_df['genres'].unique()).split('|')))))
rating_range = st.sidebar.slider("Predicted Rating Range:", 0.0, 5.0, (0.0,5.0), 0.1)
movie_search = st.sidebar.text_input("Search Movie Title (Optional)")

# Generate recommendations
user_recommendations = hybrid_recommend(
    selected_user, top_n=top_n, 
    genre_filter=genre_filter if genre_filter != "" else None, 
    rating_range=rating_range,
    movie_search=movie_search
)

st.subheader(f"Top {top_n} Recommendations for User {selected_user}")

# -----------------------------
# 7️ Display Recommendations with Posters
# -----------------------------
for _, row in user_recommendations.iterrows():
    st.markdown(f"**{row['title']}** ({row['predicted_rating']:.2f})")
    if 'poster_url' in row and pd.notna(row['poster_url']):
        try:
            response = requests.get(row['poster_url'])
            img = Image.open(BytesIO(response.content))
            st.image(img, width=150)
        except:
            st.text("Poster not available")

# -----------------------------
# 8️ Visualization: Recommended Movies Ratings
# -----------------------------
st.subheader("📊 Recommended Movies Ratings")
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x='predicted_rating', y='title', data=user_recommendations, palette='magma', ax=ax)
ax.set_xlabel("Predicted Rating")
ax.set_ylabel("Movie")
st.pyplot(fig)

# -----------------------------
# 9️ Visualization: User Ratings Distribution
# -----------------------------
st.subheader(f"📈 Ratings Distribution of User {selected_user}")
user_ratings = user_movie_matrix.loc[selected_user]
fig2, ax2 = plt.subplots(figsize=(8,4))
sns.histplot(user_ratings[user_ratings > 0], bins=10, kde=True, color='skyblue', ax=ax2)
ax2.set_xlabel("Rating")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# -----------------------------
# 10 Most Popular Movies
# -----------------------------
st.subheader("🏆 Most Popular Movies")
movie_counts = ratings_df['movieId'].value_counts().head(10)
popular_movies = movies_df[movies_df['movieId'].isin(movie_counts.index)]
popular_movies = popular_movies.merge(movie_counts.rename('count'), left_on='movieId', right_index=True)

fig3, ax3 = plt.subplots(figsize=(10,5))
sns.barplot(x='title', y='count', data=popular_movies, palette='coolwarm', ax=ax3)
ax3.set_ylabel("Number of Ratings")
ax3.set_xlabel("Movie")
plt.xticks(rotation=45)
st.pyplot(fig3)

st.success("  movie recommendation is fully interactive with posters & filters!")
