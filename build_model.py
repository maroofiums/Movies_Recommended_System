import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================
# LOAD RAW DATA
# =====================================
df = pd.read_csv("movies.csv")   # apna original dataset

# =====================================
# CLEAN DATA
# =====================================
df = df[["title", "overview", "popularity", "vote_average"]]

df = df.dropna(subset=["overview"])
df = df.reset_index(drop=True)

# Convert numeric safely
df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0)

# =====================================
# NORMALIZE SCORES
# =====================================
max_pop = df["popularity"].max()
max_vote = df["vote_average"].max()

df["norm_popularity"] = df["popularity"] / max_pop if max_pop != 0 else 0
df["norm_vote"] = df["vote_average"] / max_vote if max_vote != 0 else 0

# =====================================
# TFIDF
# =====================================
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["overview"])

# =====================================
# SAVE FILES
# =====================================
df.to_pickle("df.pkl")

pickle.dump(tfidf_matrix, open("tfidf_matrix.pkl", "wb"))

indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
pickle.dump(indices, open("indices.pkl", "wb"))

print("✅ Model Build Complete")
print("DF shape:", df.shape)
print("TFIDF shape:", tfidf_matrix.shape)
