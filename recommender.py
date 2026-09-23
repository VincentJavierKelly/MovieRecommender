"""User-based collaborative filtering with cosine similarity on MovieLens."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"


class Recommender:
    def __init__(self, min_ratings=20):
        try:
            movies = pd.read_csv(DATA / "movies.csv")
            ratings = pd.read_csv(DATA / "ratings.csv")
        except FileNotFoundError:
            raise SystemExit("Missing data. Put MovieLens movies.csv and ratings.csv in data/ (see README).")

        # Keep movies with enough ratings so predictions are not driven by noise
        counts = ratings["movieId"].value_counts()
        keep = counts[counts >= min_ratings].index
        ratings = ratings[ratings["movieId"].isin(keep)]
        self.movies = movies.set_index("movieId").loc[keep]
        self.popular = counts.loc[keep].sort_values(ascending=False).index

        # User x Movie matrix (NaN = not rated)
        self.R = ratings.pivot_table(index="userId", columns="movieId", values="rating")
        self.movie_ids = self.R.columns.to_numpy()
        self.col = {m: i for i, m in enumerate(self.movie_ids)}
        self.rated = self.R.notna().to_numpy()
        self.raw = self.R.to_numpy()
        # Mean-center each user so "generous" and "harsh" raters are comparable; unrated -> 0
        self.means = np.nanmean(self.raw, axis=1)
        self.C = np.nan_to_num(self.raw - self.means[:, None])

    def title(self, mid):
        return self.movies.loc[mid, "title"]

    def recommend(self, my_ratings, k=30, n=10, min_neighbors=3):
        """my_ratings: {movieId: rating}. Returns a list of dicts with an explanation."""
        my_ratings = {m: r for m, r in my_ratings.items() if m in self.col}
        if len(my_ratings) < 3:
            return []

        # 1. Build my rating vector (mean-centered, same layout as a matrix row)
        u = np.full(len(self.movie_ids), np.nan)
        for m, r in my_ratings.items():
            u[self.col[m]] = r
        seen = ~np.isnan(u)
        my_mean = np.nanmean(u)
        uc = np.nan_to_num(u - my_mean)

        # 2. Cosine similarity between me and every user
        sims = self.C @ uc / (np.linalg.norm(self.C, axis=1) * np.linalg.norm(uc) + 1e-9)

        # 3. Keep the k most similar users with positive similarity
        nb = np.argsort(sims)[::-1][:k]
        nb = nb[sims[nb] > 0]
        if len(nb) == 0:
            return []
        s = sims[nb]

        # 4. Predict: my mean + similarity-weighted average of neighbours' deviations
        num = s @ self.C[nb]
        den = (np.abs(s)[:, None] * self.rated[nb]).sum(axis=0)
        support = self.rated[nb].sum(axis=0)
        pred = my_mean + num / np.maximum(den, 1e-9)

        # 5. Drop rated movies and movies too few neighbours rated, then rank
        pred[seen | (support < min_neighbors)] = -np.inf
        top = np.argsort(pred)[::-1][:n]

        liked_by_me = seen & (np.nan_to_num(u) >= 4)
        out = []
        for j in top:
            if np.isneginf(pred[j]):
                break
            fans = nb[self.raw[nb, j] >= 4]  # similar users who loved it
            if len(fans):
                # Which of MY liked movies do those fans also like?
                overlap = (self.raw[fans][:, liked_by_me] >= 4).sum(axis=0)
                liked_ids = self.movie_ids[liked_by_me]
                best = liked_ids[np.argsort(overlap)[::-1][:2]] if overlap.size else []
                because = " and ".join(self.title(b) for b in best)
                why = f"{len(fans)} of your {len(nb)} most similar users rated it 4+ stars."
                if because:
                    why += f" They also liked {because}, which you liked too."
            else:
                why = "Users with similar taste rated it above their own average."
            mid = self.movie_ids[j]
            out.append({
                "title": self.title(mid),
                "genres": self.movies.loc[mid, "genres"].replace("|", ", "),
                "score": round(float(min(max(pred[j], 0.5), 5.0)), 2),
                "why": why,
            })
        return out
