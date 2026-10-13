# Verification status

## Sleeve search repair (2026-10-13)

Reproduced `plain pink shirt under 1200` returning five records while adding
`full sleeve` returned zero. The generated SQL correctly used `sleeve = 'full'`.
Three visually long-sleeved results had `unknown` in the original sleeve field;
the old CV ledger also called them `unclear`. Their local Amazon descriptions
explicitly describe sleeves that can be raised and worn short.

`enrich_sleeves.py` backfilled 163 missing sleeves using explicit length wording
in titles/features/descriptions or specific convertible-long-sleeve wording.
94 became full, 69 became half; 12,408 remain unknown. Conflicting or absent text
stays unresolved, and existing full/half values were retained. The prior catalog
is backed up in `data/backups/shirts_master_before_sleeves_e409beb07a48.parquet`.
All non-sleeve fields and all 50,224 IDs were compared unchanged. Added provenance:
`sleeve_original`, `sleeve_source`, `sleeve_evidence`. CSV was refreshed as well.

The UI now shows `sleeve unknown` instead of omitting the badge. Vision status
badges explicitly concern color/pattern, since those checks did not verify sleeves.
23 automated tests pass, including the pink/full/budget regression, negative and
conflicting metadata wording, ID alignment, and repeat enrichment. This repairs
missing text-supported sleeve data; it is not a validation of all sleeve/image tags.
See `docs/sleeve-search-check.json` for live NVIDIA query checks.

2026-10-13: master merge has 50,224 unique IDs; all match Kaggle results by ID and
image URL. All original columns except color, pattern and color_source compare equal
to the preserved pre-merge master. Backup SHA256 and merged master SHA256 are recorded
in `data/vision_merge_summary.json`. CSV copy generated. Source archive intact.

Colors present: 49,658 products. Known patterns: 41,752 products.
27,411 heuristic-ok predictions; 22,253 needs_review predictions included as requested.
433 mask_failed, 92 multiple_shirts and 35 shirt_not_detected cases remain unknown.
Coverage is not an accuracy score. Some needs_review patterns also remain unknown.

The query-layer tests cover merge alignment, SQL restrictions, real Parquet retrieval,
ranking, endpoint-specific credentials, provider adapter fixtures and FastAPI routes.
The user supplied a NVIDIA key, stored only in the app's ignored `.env` configuration.
The selected hosted model is `nvidia/nemotron-3.5-lightning-30b-a3b`, tested with
thinking disabled to obtain final SQL JSON directly. NVIDIA model availability was
checked at its live model listing, followed by real successful inference requests.
16 tests pass. Exact color token and pattern vocabulary normalization is enforced
after model generation, before restricted SQL execution.

Real API endpoint tests (not fixtures) passed:
- Blue and white striped shirts under 2,000 INR: 10 matches; 5 returned and checked
  for both exact colors, a stripe pattern and the budget; 1.535 seconds.
- Maroon plain full sleeve shirts: 287 matches; 5 returned and checked for exact
  color token, plain pattern and full sleeves; 1.29 seconds.

The public, credential-free report is `docs/live-provider-check.json`. These checks
verify the query flow and returned data constraints, not image-label accuracy.
NVIDIA's free hosted endpoints are for prototyping and have account/model rate limits.
OpenRouter is configurable but was not used in the live tests.

The local app was started on port 8010. Browser access was denied by the browser
permission layer, so visual QA was not performed. API tests inspect routes and HTML
responses, not rendered browser appearance.

31,793 saved processing dimensions: 30,451 have longest side >=1000 pixels;
22,233 have longest side >=1500 pixels; 454 have longest side <500 pixels.
The pipeline capped images at 1500; these are processed sizes, not proof of original
full-resolution availability. URLs are unchanged from the master.
