# Verification status

## Predominant colors and outerwear exclusions (2026-10-14)

Single-positive-shade queries and color selections require the requested shade
to be the unique largest saved palette share and cover at least 50% of shirt
pixels. This runs before count, ranking and limit and independently of LLM SQL.
Multi-color queries keep exact token matching. Sole-color/plain rules still
apply. Outerwear title rules exclude clear non-shirt listings while preserving
all raw master rows; ambiguous line names and hybrids are retained.

44 automated tests pass. They include the reported navy/black example, blazer
title, majority/minority colors, malformed evidence, distinct gold/yellow shades,
multi-color requests, and incidental jacket wording. Live provider results are
in `docs/verification/dominant-color-live-check.json`; title exclusion evidence
is in `data/reports/non_shirt_title_exclusions.csv`.

## Folder cleanup (2026-10-14)

The project was organized without changing the live master, product-detail
sidecar or API settings. File moves were hash-checked, generated caches and empty
placeholders were removed, and maintenance/document paths were updated. The
current suite has 39 checks, including office/comfort ranking; earlier sections
below retain their original test counts and dates. Move records are in
`docs/cleanup_manifest.json`.


## Optional brand and attribute filters (2026-10-13)

Added a full searchable 3,171-name brand/store selector and optional color, pattern,
size, sleeve, fabric, fit, pocket and maximum-INR-price controls. They default to
unselected. Multiple brands use OR; different selections and text-query constraints
use AND. Backend validation checks choices against the catalog, and safely builds
the additional predicates independently of generated SQL. Empty optional selections
add no constraints. Single-color/plain selections preserve the strict mono-color
rule. Filter-only searches bypass the hosted model. Pending changes are labeled;
older search responses cannot overwrite a newer request. Clearing selections
preserves the typed query. A request with no text or selections browses via the UI's
existing browse endpoint, respecting the uncertainty checkbox.

34 automated tests pass, including optional-vs-unrestricted results, combined
brand/size/color/sleeve constraints, query conflicts returning empty matches,
multi-brand selection, apostrophes in a brand, invalid choices, and no-LLM filter-only
requests. JavaScript syntax and unique HTML IDs checked; no browser visual QA.
Live checks: JMIERR+M alone returned 31 matches in 0.205 seconds; adding text budget
under 2000 returned 13 with brand, size and budget verified. Query-only pink/plain
under 1200 remained five matches. Black/plain/max1500 selections returned 20 sole
black/plain matches in 0.168 seconds. See `docs/verification/selected-filters-live-check.json`.

## Product details and size enrichment (2026-10-13)

Saved source audit covers all 50,224 IDs. `product_enrichment.py` joins it with
local brand metadata, adds six brand/size fields and produces a product-detail
sidecar. All pre-existing master columns compare unchanged. Backup:
`data/backups/shirts_master_before_details_a3f82d328962.parquet`.
Named sizes were extracted for 8,821 products; 46 only have another recorded size
format; 41,357 have no usable size evidence. Sources and exact evidence are saved.
Model-worn sizes, L/S and S/S sleeve abbreviations, package dimensions and fit
warnings are excluded from size extraction. Range endpoints are retained without
fabricating intermediate options. These tags remain historical listing mentions,
not verified stock, image-specific variants or normalized body measurements.

Images/titles and a details button now open a native dialog with gallery, brand,
sizes, prices, ratings, product attributes, features, description and source details.
Unknown sizes are explicit. Source strings are rendered with `textContent`; no
source HTML is injected. Product lookup is parameterized with validated ASIN IDs;
invalid IDs return 422 and missing products return 404. The public static asset
mount exposes only the web folder. SQL remains restricted to the shirts view.

30 automated tests passed. JavaScript syntax, HTML IDs and image-click handler
were checked. Rendered browser visual QA was not performed. Real HTTP checks
verified three detail records (including a size chart, a structured XL and an
unknown size), static assets, and NVIDIA-generated exact-size SQL: white/plain/M
under 2000 returned 12 matches; XXL under 1500 returned 278 matches. Returned size
tokens were checked. Evidence: `docs/verification/product-details-live-check.json`.

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
See `docs/verification/sleeve-search-check.json` for live NVIDIA query checks.

2026-10-13: master merge has 50,224 unique IDs; all match Kaggle results by ID and
image URL. All original columns except color, pattern and color_source compare equal
to the preserved pre-merge master. Backup SHA256 and merged master SHA256 are recorded
in `data/reports/vision_merge_summary.json`. CSV copy generated. Source archive intact.

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

The public, credential-free report is `docs/verification/live-provider-check.json`. These checks
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
