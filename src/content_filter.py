import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import TOP_K


class ContentFilter:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.vectorizer = TfidfVectorizer()
        self.matrix: csr_matrix = self.vectorizer.fit_transform(df["metadata_string"])

    def get_top_k(self, user_input: str, k: int = TOP_K) -> pd.DataFrame:
        user_vec = self.vectorizer.transform([user_input])
        scores = cosine_similarity(user_vec, self.matrix)[0]
        top_indices = np.argsort(scores)[::-1][:k]
        out = self.df.iloc[top_indices].copy()
        out["score"] = scores[top_indices]
        return out.reset_index(drop=True)
