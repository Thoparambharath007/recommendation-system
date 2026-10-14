"""03b_price_estimator_v2.py - ML price model (HistGradientBoostingRegressor).

Trains on the 8,251 REAL Amazon-2023 prices (old dataset only - no external
sources). Features: title TF-IDF words + brand + sleeve + rating stats.
Honesty protocol:
  * 2-fold blind exam: train on fold A, predict fold B, and vice versa ->
    every exact price is predicted WITHOUT having been seen in training.
  * Baseline (brand median, fold-safe) kept for the ablation scoreboard.
  * Split conformal bands: 80% coverage guaranteed from holdout residuals.
Then the final model (trained on ALL exact prices) re-estimates the 41,973.

Output: data/shirts_priced.parquet (+ .csv) - v2 table, price_source:
  exact-metadata | estimated-ml   (bands on estimates only)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

HERE = Path(__file__).resolve().parents[2]
SRC = HERE / "data" / "intermediate" / "shirts_final.parquet"
OUT_PARQUET = HERE / "data" / "intermediate" / "shirts_priced.parquet"
OUT_CSV = HERE / "data" / "intermediate" / "shirts_priced.csv"
INR_PER_USD = 83.0
rng = np.random.default_rng(42)

df = pd.read_parquet(SRC)
df["price_exact"] = df["price"].where(df["price"] > 0)
exact = df[df["price_exact"].notna()].copy()
print(f"catalog {len(df):,} | exact prices for training: {len(exact):,}")

# ---------- feature pipeline ----------
CAT = ["store", "sleeve"]
NUM = ["average_rating", "rating_number"]

def _maps(train_df):
    med = train_df.groupby("store")["price_exact"].median()
    cnt = train_df.groupby("store")["price_exact"].count()
    return med, cnt, float(train_df["price_exact"].median())

def _feats(df_, enc, tfidf, med, cnt, glob):
    base = df_["store"].map(med).fillna(glob).to_numpy(float)
    num = np.column_stack([
        df_["average_rating"].fillna(0).to_numpy(float),
        np.log1p(df_["rating_number"].fillna(0).to_numpy(float)),
        np.log1p(base),
        np.log1p(df_["store"].map(cnt).fillna(0).to_numpy(float)),
    ])
    cat = enc.transform(df_[CAT].astype(str))
    X = np.hstack([cat, num, tfidf.transform(df_["title"].fillna("")).toarray()])
    return X, base

def build_model(train_df):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train_df[CAT].astype(str))
    tfidf = TfidfVectorizer(min_df=4, sublinear_tf=True, strip_accents="unicode",
                            lowercase=True, token_pattern=r"[a-zA-Z]{2,}",
                            max_features=15000)
    tfidf.fit(train_df["title"].fillna(""))
    med, cnt, glob = _maps(train_df)
    X, base = _feats(train_df, enc, tfidf, med, cnt, glob)
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=5,
        l2_regularization=1.0, early_stopping=False, random_state=42)
    # RESIDUAL target: how far above/below its brand median does THIS shirt sit?
    model.fit(X, np.log1p(train_df["price_exact"].to_numpy(float)) - np.log1p(base))
    return enc, tfidf, med, cnt, glob, model

def predict(df_, enc, tfidf, med, cnt, glob, model):
    X, base = _feats(df_, enc, tfidf, med, cnt, glob)
    return np.expm1(model.predict(X) + np.log1p(base))

# ---------- 2-fold blind exam ----------
exact = exact.reset_index(drop=True)
exact["fold"] = rng.permutation(len(exact)) % 2
oof_ml = np.full(len(exact), np.nan)
oof_base = np.full(len(exact), np.nan)
for f in (0, 1):
    tr, te = exact[exact.fold == f], exact[exact.fold != f]
    bundle = build_model(tr)
    oof_ml[exact.fold != f] = predict(te, *bundle)
    med = tr.groupby("store")["price_exact"].median()
    glob = tr["price_exact"].median()
    oof_base[exact.fold != f] = te["store"].map(med).fillna(glob).to_numpy()

truth = exact["price_exact"].to_numpy()
def scorecard(pred, name):
    err = np.abs(pred - truth)
    return {"method": name, "MAE_$": round(err.mean(), 2),
            "%within_$10": round(100 * (err <= 10).mean(), 1),
            "%within_20%": round(100 * (err / truth <= 0.20).mean(), 1)}
board_ml = scorecard(oof_ml, "ML gradient boosting")
board_base = scorecard(oof_base, "brand median (v1)")

# conformal band: 80th percentile of blind errors -> guaranteed ~80% coverage
q80_ml = np.quantile(np.abs(oof_ml - truth), 0.80)
q80_base = np.quantile(np.abs(oof_base - truth), 0.80)
cover_ml = 100 * np.mean(np.abs(oof_ml - truth) <= q80_ml)
cover_base = 100 * np.mean(np.abs(oof_base - truth) <= q80_base)

print("\n=== BLIND EXAM (8,251 real prices, each predicted unseen) ===")
print(pd.DataFrame([board_base, board_ml]).to_string(index=False))
print(f"\nconformal bands: ML ±${q80_ml:.2f} (covers {cover_ml:.1f}% of truths)"
      f" | brand-median ±${q80_base:.2f} (covers {cover_base:.1f}%)")

# ---------- final fit on ALL exact prices, estimate the 42k ----------
enc_tf_med = build_model(exact)
# PERSIST the trained model so it can score new shirts without retraining
import joblib
MODEL_PATH = HERE / "data" / "models" / "price_model.joblib"
joblib.dump({"bundle": enc_tf_med, "band_usd": float(q80_ml),
             "trained_on": int(len(exact)), "seed": 42}, MODEL_PATH)
print(f"model saved: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size/1e6:.1f} MB)")
unpriced = df[df["price_exact"].isna()].copy()
preds = predict(unpriced, *enc_tf_med)
unpriced["price_ml"] = preds
unpriced["band_lo"] = np.maximum(preds - q80_ml, 0.99)
unpriced["band_hi"] = preds + q80_ml

out = df.merge(unpriced[["parent_asin", "price_ml", "band_lo", "band_hi"]],
               on="parent_asin", how="left")
out["price_usd"] = out["price_exact"].fillna(out["price_ml"])
out["price_source"] = np.where(out["price_exact"].notna(), "exact-metadata", "estimated-ml")
out["price_inr_est"] = (out["price_usd"] * INR_PER_USD).round(0)

# user cleanup 2026-10-09: dropped band_lo/band_hi (constant +-q80, recomputable),
# vision_class/shirt_prob (QA leftovers), price_exact (redundant with price_usd
# + price_source). price_source KEPT by explicit user decision - the honesty label.
keep = ["parent_asin", "title", "store", "sleeve", "average_rating", "rating_number",
        "image_url", "price_usd", "price_source", "price_inr_est"]
out[keep].to_parquet(OUT_PARQUET, index=False)
out[keep].to_csv(OUT_CSV, index=False)

print(f"\n=== FINAL v2 TABLE: {len(out):,} shirts, price coverage "
      f"{out.price_usd.notna().mean()*100:.1f}% ===")
print(out.price_source.value_counts().to_string())
print("\nwhat did the model learn? sample estimates vs their brand median:")
demo = out[out.price_source == "estimated-ml"].head(5)
for _, r in demo.iterrows():
    print(f"  {str(r['title'])[:52]:54s} {r['store'][:14]:15s} ${r['price_usd']:.2f}")
print(f"\nsaved: {OUT_PARQUET.name} + {OUT_CSV.name}")
