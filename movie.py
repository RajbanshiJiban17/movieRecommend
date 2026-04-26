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

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# -----------------------------
# 1️ Load Data (Optimized)
# -----------------------------
@st.cache_data
def load_data():
    # ३ करोड डाटा एकैपटक लोड नगर्न nrows=2000000 (२० लाख) राख्नुहोस्
    # यदि RAM धेरै छ भने nrows हटाउन सक्नुहुन्छ
    movies = pd.read_csv(r"C:\Users\User\Desktop\movie\ml-32m\movies.csv")
    
    # Memory जोगाउन dtype तोक्ने
    ratings = pd.read_csv(
        r"C:\Users\User\Desktop\movie\ml-32m\ratings.csv",
        dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'},
        usecols=['userId', 'movieId', 'rating'],
        nrows=2000000  # पहिलो २० लाख डाटा मात्र लोड गर्ने (Memory जोगाउन)
    )
    return movies, ratings

movies_df, ratings_df = load_data()

# धेरै एक्टिभ युजर र चर्चित फिल्म मात्र छान्ने (Matrix सानो बनाउन)
top_users = ratings_df['userId'].value_counts().head(1000).index
top_movies = ratings_df['movieId'].value_counts().head(500).index
ratings_small = ratings_df[ratings_df['userId'].isin(top_users) & ratings_df['movieId'].isin(top_movies)]

st.subheader("Sample Ratings Data")
st.dataframe(ratings_df.head())

# -----------------------------
# 2️ User-Movie Matrix
# -----------------------------
# Matrix बनाउँदा sparse matrix प्रयोग गर्नु राम्रो हुन्छ, तर अहिलेलाई सानो डेटामा pivot चल्छ
user_movie_matrix = ratings_small.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# -----------------------------
# 3️ Train SVD for AI Prediction
# -----------------------------
#

# -----------------------------
# 3️ Train SVD for AI Prediction (Updated)
# -----------------------------
@st.cache_data
# index र columns को अगाडि '_' थप्नुहोस् ताकि Streamlit ले यसलाई ह्यास नगर्नुहोस्
def train_svd(matrix_values, _index, _columns, n_components=20):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    matrix_filled = svd.fit_transform(matrix_values)
    matrix_reconstructed = np.dot(matrix_filled, svd.components_)
    
    # यहाँ भित्र पनि _index र _columns नै प्रयोग गर्नुहोस्
    predicted_ratings = pd.DataFrame(matrix_reconstructed, index=_index, columns=_columns)
    return predicted_ratings

# फङ्सन कल गर्दा पनि आर्गुमेन्टको नाम मिलाउनुहोस् (optional तर राम्रो अभ्यास)
predicted_ratings_df = train_svd(
    user_movie_matrix.values, 
    _index=user_movie_matrix.index, 
    _columns=user_movie_matrix.columns
)
# -----------------------------
# 4️ Hybrid Recommendation Function
# -----------------------------
def hybrid_recommend(user_id, top_n=5, genre_filter=None, rating_range=None, movie_search=None):
    if user_id not in predicted_ratings_df.index:
        return pd.DataFrame() # यदि युजर डेटामा छैन भने खाली दिने

    user_pred = predicted_ratings_df.loc[user_id].sort_values(ascending=False)
    
    # युजरले नहेरेका फिल्म मात्र छान्ने
    already_rated = user_movie_matrix.loc[user_id]
    unrated_movies = already_rated[already_rated == 0].index
    
    available_unrated = [m for m in unrated_movies if m in user_pred.index]
    top_movies_ids = user_pred[available_unrated].head(100).index
    
    recommended = pd.DataFrame({
        'movieId': top_movies_ids,
        'predicted_rating': user_pred[top_movies_ids].values
    })
    recommended = recommended.merge(movies_df, on='movieId')
    
    # Filter: Genre
    if genre_filter:
        recommended = recommended[recommended['genres'].str.contains(genre_filter, case=False)]
    
    # Filter: Rating Range
    if rating_range:
        min_r, max_r = rating_range
        recommended = recommended[(recommended['predicted_rating'] >= min_r) & (recommended['predicted_rating'] <= max_r)]
    
    # Filter: Search
    if movie_search:
        recommended = recommended[recommended['title'].str.contains(movie_search, case=False)]
    
    return recommended.head(top_n)

# -----------------------------
# 5️ Streamlit UI
# -----------------------------
st.sidebar.header("User Options")
user_list = list(user_movie_matrix.index)
selected_user = st.sidebar.selectbox("Choose User ID:", user_list)
top_n = st.sidebar.slider("Number of Recommendations:", 1, 20, 5)

# सबै Genre हरू निकाल्ने
all_genres = set()
movies_df['genres'].str.split('|').apply(lambda x: all_genres.update(x))
genre_filter = st.sidebar.selectbox("Filter by Genre:", ["All"] + sorted(list(all_genres)))

rating_range = st.sidebar.slider("Predicted Rating Range:", 0.0, 5.0, (0.0, 5.0), 0.1)
movie_search = st.sidebar.text_input("Search Movie Title")

# Generate recommendations
user_recommendations = hybrid_recommend(
    selected_user, top_n=top_n, 
    genre_filter=None if genre_filter == "All" else genre_filter, 
    rating_range=rating_range,
    movie_search=movie_search
)

if not user_recommendations.empty:
    st.subheader(f"Top {len(user_recommendations)} Recommendations for User {selected_user}")
    
    # Display posters in columns
    cols = st.columns(min(len(user_recommendations), 5))
    for i, (_, row) in enumerate(user_recommendations.iterrows()):
        with cols[i % 5]:
            st.write(f"**{row['title']}**")
            st.caption(f"Score: {row['predicted_rating']:.2f}")
            if 'poster_url' in row and pd.notna(row['poster_url']):
                st.image(row['poster_url'], use_container_width=True)
            else:
                st.write("🖼️ No Poster")

    # Visualizations
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Recommendation Scores")
        fig, ax = plt.subplots()
        sns.barplot(x='predicted_rating', y='title', data=user_recommendations, palette='viridis', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("🏆 Popularity")
        movie_counts = ratings_df['movieId'].value_counts().head(10)
        pop_movies = movies_df[movies_df['movieId'].isin(movie_counts.index)].copy()
        pop_movies['counts'] = pop_movies['movieId'].map(movie_counts)
        fig2, ax2 = plt.subplots()
        sns.barplot(x='counts', y='title', data=pop_movies, palette='rocket', ax=ax2)
        st.pyplot(fig2)
else:
    st.warning("No movies found matching your filters.")