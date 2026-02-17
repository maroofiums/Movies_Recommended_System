import streamlit as st
import pandas as pd
import pickle
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Movie Recommender", layout="wide")

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    
    # Load pickles
    df_path = os.path.join(base_dir, "df.pkl")
    tfidf_path = os.path.join(base_dir, "tfidf_matrix.pkl")
    indices_path = os.path.join(base_dir, "indices.pkl")

    df = pd.read_pickle(df_path)
    with open(tfidf_path, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(indices_path, "rb") as f:
        indices = pickle.load(f)

    # Ensure numeric columns
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0)

    # Normalize safely
    max_pop = df["popularity"].max()
    max_vote = df["vote_average"].max()

    df["norm_popularity"] = df["popularity"] / max_pop if max_pop != 0 else 0
    df["norm_vote"] = df["vote_average"] / max_vote if max_vote != 0 else 0

    return df, tfidf_matrix, indices

df, tfidf_matrix, indices = load_data()

# =====================================
# TMDB CONFIG
# =====================================
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_IMAGE = os.path.join(os.path.dirname(__file__), "../screenshots/notfound.png")

@st.cache_data
def get_movie_data(title):
    if not TMDB_API_KEY:
        return PLACEHOLDER_IMAGE, None
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        response = requests.get(url)
        if response.status_code != 200:
            return PLACEHOLDER_IMAGE, None
        data = response.json()
        if data.get("results"):
            movie = data["results"][0]
            poster = movie.get("poster_path")
            backdrop = movie.get("backdrop_path")
            poster_url = TMDB_IMAGE_URL + poster if poster else PLACEHOLDER_IMAGE
            backdrop_url = TMDB_IMAGE_URL + backdrop if backdrop else None
            return poster_url, backdrop_url
    except Exception as e:
        print("TMDB API Error:", e)
    return PLACEHOLDER_IMAGE, None

# =====================================
# RECOMMENDATION FUNCTION
# =====================================
def recommend(title, n=10):
    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]

    # Content similarity
    sim_score = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()

    # Hybrid score (content + popularity + vote_average)
    hybrid_score = (
        sim_score * 0.6 +
        df["norm_popularity"].values * 0.2 +
        df["norm_vote"].values * 0.2
    )

    similar_idx = np.argsort(hybrid_score)[::-1][1:n+1]

    results = df.iloc[similar_idx][["title"]].copy()
    return results

# =====================================
# SESSION STATE
# =====================================
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = df["title"].iloc[0]

# =====================================
# UI - Dropdown
# =====================================
st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox(
    "Select a movie:",
    df["title"].sort_values(),
    index=int(df[df["title"] == st.session_state.selected_movie].index[0])
)

st.session_state.selected_movie = selected_movie

# =====================================
# SHOW SELECTED MOVIE POSTER / BACKDROP
# =====================================
poster, backdrop = get_movie_data(selected_movie)

if backdrop:
    st.image(backdrop, use_container_width=True)

st.header(selected_movie)

# =====================================
# SHOW RECOMMENDATIONS
# =====================================
recommendations = recommend(selected_movie, 10)

st.subheader("Recommended Movies")
cols = st.columns(5)

for i, row in enumerate(recommendations.itertuples()):
    with cols[i % 5]:
        poster_url, _ = get_movie_data(row.title)
        poster_url = poster_url or PLACEHOLDER_IMAGE

        if st.button(row.title, key=row.title):
            st.session_state.selected_movie = row.title

        st.image(poster_url, use_container_width=True)
