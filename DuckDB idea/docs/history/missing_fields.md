# FINALISED FIELDS — the master table spec (locked 2026-10-09)

> Old drafts removed. This file now holds ONLY the final agreed field set.
> Verified by probe: JUNK created by qoder/probe_verify_final_fields.py
> (12-query battery: core fields answer 7/12 with real columns; the 5 gaps
>  are exactly Zone 2 below — that's why Zone 2 exists).

# ═══════════════════════════════════════════
## CORE 7 — exist today in shirts_final.parquet (verified 100% filled)
# ═══════════════════════════════════════════

1. title          → every descriptive word a human says
2. brand          → from store column (89 blanks → "Unknown")
3. sleeve         → half / full
4. rating         → average_rating, out of 5 stars (display)
5. image_url      → HIGH-RES: chop "._AC_UL1500_" suffix → original photo
6. product_id     → parent_asin, unique per row (verified), the join key
7. rating_count   → rating_number, backstage trust gate (never displayed)

# ═══════════════════════════════════════════
## PRICE — DONE ✅ (ML estimation, validated)
# ═══════════════════════════════════════════

8.  price_inr     → VISIBLE, rupees only (user is Indian, ₹ is the language)
9.  price_usd     → backstage native number (Amazon priced in $)
10. price_source  → backstage honesty label: exact-metadata | estimated-ml
11. band_lo_inr / band_hi_inr → backstage 80%-guaranteed bracket

   model: HistGradientBoostingRegressor (brand prior + title correction)
   blind exam: 70.5% within ±$10 · bands cover exactly 80.0% of truths
   why estimated at all: original Amazon dataset has prices for only
   8,251 of 50,224 shirts — the rest genuinely don't exist at source

# ═══════════════════════════════════════════
## ZONE 2 — the new attributes (5 stored; button_color discarded)
# ═══════════════════════════════════════════

12. color        → CLIP vision, Job B (Kaggle GPU, one trip)
13. pattern      → CLIP vision + title regex, Job B + lie detector
                   (titles saying striped/checked = ground truth to grade CLIP)
14. fabric       → details{} FIRST ("Fabric Type"), then features[],
                   description, title — local Job A (file already on disk:
                   men_shirts_metadata.parquet holds description+features)
15. fit / size   → frontend size chips S/M/L/XL, same results all sizes
                   (we have no per-size data — parent-asin families)
                   UI must say "sizes: see product page", never fake a filter
                   NOTE: size ≠ fit. Real fit type (slim/regular) mined in
                   Job A from details{} when present
16. pocket       → triple vote: CLIP + details{} + title mention
                   (CLIP alone is not trusted here)
17. button_color → ❌ DISCARDED 2026-10-09 (user: queries on it are rare).
                   Not built, not stored. Kept here only as history.

# ═══════════════════════════════════════════
## FUTURE (not in v1)
# ═══════════════════════════════════════════

- sentiment_score   → join from product_sentiment.parquet (free when wanted)
- co-purchase badge → from copurchase_edges.parquet (free when wanted)

# ═══════════════════════════════════════════
## THE RULE
# ═══════════════════════════════════════════

Display speaks the user's language: ₹, stars out of 5, big photos.
Engine speaks the truth language: counts, sources, bands, backstage keys.
Neither ever deletes the other.

# ═══════════════════════════════════════════
## BUILD ORDER
# ═══════════════════════════════════════════

04_shirts_master.py → CORE 7 + price fields → shirts_master.parquet (+CSV)
                      Zone 2 columns added as EMPTY reserved slots
Job A (local)       → fabric, fit-type (+pocket vote part)
Job B (Kaggle)      → color, pattern (+image fingerprints for similarity)
Job C (local)       → review-mined extras when future fields arrive
(button pipeline)   → DISCARDED with button_color — out of scope
