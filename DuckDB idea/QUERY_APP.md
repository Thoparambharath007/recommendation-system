# Shirt query application

The merged 50,224-row master powers this flow:
query -> OpenAI-compatible LLM -> validated SQL -> DuckDB scans Parquet -> ranked products.
This is a working query application, not a claim of evaluated recommendation accuracy.

## Run

Double-click `start.bat` in this folder, then open http://127.0.0.1:8010.
Alternatively run `python "DuckDB idea/launch.py"` from the project root.
On another computer install `requirements-query.txt`. SQLGlot is also available in
this machine's project-local `.vendor` directory.

The prepared `.env` selects NVIDIA's hosted endpoint with
`nvidia/nemotron-3.5-lightning-30b-a3b`; your key is stored only in local `.env`.
`NVIDIA_API_KEY` in the environment also works. For OpenRouter, change the URL to
`https://openrouter.ai/api/v1`, model to `openai/gpt-4.1-mini` (or your supported model)
and use the OpenRouter key. No keys are sent to the browser. Settings require a restart.
The adapter never falls back to an unrelated provider's key.
NVIDIA defaults to prompted JSON and strict local parsing, avoiding an unsupported
response-format assumption. Nemotron Lightning uses `enable_thinking=false` to
return the SQL JSON directly within the output token budget.
OpenRouter/OpenAI use `response_format: json_object`;
choose a compatible model, or set `LLM_JSON_MODE=prompt` and keep local validation.
Only the request and catalog schema/vocabulary are sent to the LLM, not the catalog.

## Catalog update

`merge_vision.py` reads `shirt_results_backup final.zip`, validates an exact ID and
image-URL match, backs up the original Parquet, and replaces `data/shirts_master.parquet`.
A CSV copy and `vision_merge_summary.json` accompany it. Existing IDs, images, price,
reviews, fabric and fit fields stay unchanged. Previous colors/pattern/source remain
in *_original columns as well as in the byte-for-byte backup.

Both `ok` and `needs_review` color/pattern predictions are included, as requested.
No failed mask is filled using an old, potentially incorrect tag. Unknown colors
remain empty; uncertain/unknown patterns retain `pattern_candidate` as an audit field.
The search uses the actual pattern, not the uncertain highest-score candidate.

## Query behavior

The **Choose brands** button opens the complete searchable list of 3,171 distinct
recorded brand/store names, with full-catalog product counts. Brand case variants
are grouped. Users can select several brands, then press **Apply & search**.
Optional color, pattern, size and sleeve selectors are beside it; fabric, fit,
pockets and an INR maximum price are under **More filters**. Defaults are all
unselected. **Clear filters** removes optional selections while preserving the
typed query; apply/search runs the query without those extra restrictions.

Every selected field is enforced by the backend in addition to the validated LLM
SQL. Multiple brands match any selected brand, while different fields are combined
with AND. Contradictory query/selections return zero matches, with no relaxation.
Unknown/unselected fields do not add constraints. Plain plus a single selected
color requires that sole color, as with a plain-color text query. Selection-only
searches skip the LLM and execute local filters directly. UI changes are marked
pending until applied; superseded searches cannot overwrite newer responses.
`GET /api/filters` supplies the full catalog vocabulary. `POST /api/search` accepts
an optional `filters` object; legacy requests without it retain query-only behavior.

Color filtering uses exact pipe-delimited tokens: maroon is distinct from red,
gold from yellow, light blue from white. A green-and-white request requires both.
If the model uses a simple color LIKE/contains/equality/IN predicate, the validator
normalizes it to exact tokens. Pattern aliases such as striped and plaid map to the
dataset's actual pattern vocabulary; invalid names are rejected.
For a plain/solid shirt request naming one color, the app additionally requires
the complete stored color list to equal that single color and pattern=plain.
Thus plain black excludes navy|black, burgundy|black and charcoal|black.
Explicit 'only black' also excludes mixed tags. Multi-color requests continue to
match multiple tokens. This can omit real black shirts tagged with shading colors;
it changes filtering and does not re-verify the catalog images.
Pattern names include plain, three stripe directions, checked_plaid, polka_dot,
floral, graphic_print, camouflage, color_block, other and unknown.

The model returns only `SELECT parent_asin FROM shirts WHERE ...`. AST validation
rejects multiple statements, other tables, joins, subqueries, mutations, arbitrary
functions and unknown columns. DuckDB disables external access except the configured
master and local product-details Parquet files and locks its settings. Generated
SQL may still access only the shirts view. An execution timer bounds each search.

All matches are considered before the app's top-k limit. Price provenance tiers
put exact historical prices first; within a tier a Bayesian rating average with
20 prior reviews ranks results. Rating count breaks ties. Requests for cheapest,
highest-rated or most reviewed select separate
deterministic sorts; price estimates remain labeled. This is a transparent
baseline, not the original paper's cascaded review-embedding recommender. No live
stock, personalized learning or ranking evaluation is implemented yet.

Estimated INR prices use the existing project's conversion (83 INR/USD).
Recorded size labels can be searched using exact tokens. Button color, live
inventory, delivery and personalized sizing advice are unavailable and explicitly
unsupported. Empty matches stay empty rather than silently relaxing user filters.
The default includes uncertain attributes; the UI lets users exclude them.

## Product details and sizes

Click a product image, title or **View shirt details** to open its details dialog.
It shows listing images, recorded brand (or store fallback), image color/pattern,
historical/estimated price, ratings, sleeves, fabric, fit, pockets, original features,
description and listing details. Escape or Close dismisses it.

`product_enrichment.py` joins the saved original metadata audit by `parent_asin`.
It adds `brand`, `brand_source`, `sizes`, `size_source`, `size_evidence`, `size_note`
to the master, preserving the existing columns and IDs. Full listing details are
stored separately in `data/product_details.parquet`; lookup uses a parameterized
product-ID query at `/api/products/{parent_asin}`. A pre-enrichment master backup
is kept in `data/backups`. To rebuild from source, run
`python "DuckDB idea/product_enrichment.py" --apply` from REAL SYSTEM.

8,821 products have named size labels and 46 have other recorded size formats;
41,357 have no usable size evidence. Labels are historical listing mentions, not
proof of current availability or a specific image/size/color combination. Chart
ranges are not expanded to invent intermediate options. US/Asian labels may differ;
the dialog shows the original size text so users can check the chart. Unknown sizes
are displayed honestly and excluded from specific-size searches. For example,
`plain white shirt size M under 2000` searches exact M tokens, not XL or XXL.

## Checks

Sleeve filtering uses `full` and `half`. Missing sleeve tags stay `unknown` and
do not match a request for a specific length. `enrich_sleeves.py` repairs missing
tags from local source metadata without replacing existing full/half values;
conflicting text stays unknown. It records original values, text evidence and
source, and backs up the master before writing. Run it after rebuilding the master:
`python "DuckDB idea/enrich_sleeves.py" --apply` from REAL SYSTEM.
Without `--apply` it only reports the proposed changes. Current repair: 163 tags
filled, 12,408 still unknown. The website displays missing sleeve tags explicitly.

`python -m unittest discover -s "DuckDB idea/tests" -v`
Tests cover merge identity, missing/duplicate IDs, exact shade matching, unsafe SQL,
real Parquet ranking, the compatible API adapter, product details, size extraction
and exact size filtering, optional selections, quote-containing brands, invalid
choices, filter-only searches, office/comfort preference ranking, predominant
color evidence, conservative non-shirt exclusions, and image grouping. All 46 pass.
Live provider checks are separate from fixtures: NVIDIA completed two real search
queries with returned filter constraints checked. See `docs/verification/live-provider-check.json`.
Product details and size searches were also tested live; see
`docs/verification/product-details-live-check.json`.
Optional-filter live checks are in `docs/verification/selected-filters-live-check.json`.

Sources used for implementation:
- https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat
- https://duckdb.org/docs/stable/operations_manual/securing_duckdb/overview
- https://docs.api.nvidia.com/nim/reference/meta-llama-3_3-70b-instruct
- https://openrouter.ai/docs/guides/features/structured-outputs
- https://docs.nvidia.com/nim/large-language-models/2.0.10/get-started/advanced/get-started-nemotron-3.5-lightning.html
- https://docs.api.nvidia.com/nim/docs/run-anywhere

Saved image dimensions reflect the images processed at max-side 1,500, not a
guarantee that every remote URL will remain reachable or have an original larger image.

## Office and comfort queries

`comfortable black shirt for office` and `a white shrt for officie work` are
supported. Common shirt/office typos are normalized. Office and comfort are
ranking preferences, while color, budget, sleeve, size, pattern and selected
filters remain required. The LLM receives the remaining required query;
application-owned ranking uses title, style, fabric, fit, original features and
description. Office signals include dress/oxford/formal/button-down wording;
comfort signals include soft/breathable/stretch wording, cotton/linen and
regular/relaxed fit. These are heuristic listing signals, not verified comfort
or a dress-code guarantee. Missing evidence does not remove matching products.
Explicit cheapest/highest-rating/popularity sorting still takes priority.
Unavailable required attributes such as button color remain unsupported.

## Predominant color and product-type filtering

`black shirt under 1200 rupees` requires black to be the unique largest saved
palette share and account for at least 50% of analyzed shirt pixels. It no longer
matches a mostly white or navy shirt merely because black appears among the tags.
Recommended results prioritize stronger requested-color share. Explicit price or
rating sorting still applies within the matching set. Single-color dropdown
choices use the same rule. Gold/yellow, maroon/red and other shades remain distinct.

`black and white striped shirt` retains both-color token matching. Plain/only
color requests also retain the earlier restriction against additional color tags.
Malformed or tied palette evidence cannot establish a predominant shade. Legacy
catalogs with no palette column can use a sole stored label without claiming a
measured percentage; the current catalog uses its saved Kaggle palette evidence.

`catalog_quality.py` detects clear primary outerwear titles. These are excluded
from recommendations and counts using a reversible application-owned filter;
the original master rows are preserved. Ambiguous shirt-jacket descriptions,
shirt line names such as Arrow Blazer, and incidental styling mentions are not
blanket-excluded. This is a conservative text rule, not a new image model or proof
that every remaining item is a shirt. Exclusion evidence is saved in
`data/reports/non_shirt_title_exclusions.csv`. The original 50,224-row count and
searchable count are reported separately. Live checks are recorded in
`docs/verification/dominant-color-live-check.json`.

## One image card and size/listing selection

Recommendations now group exact primary image URLs after all query/selected
filters and before the result limit. Only one matching listing represents each
image group; the displayed price, rating and attributes belong to that listing.
Results and counts refer to image groups, not repeated variant rows. Missing
image URLs fall back to individual product IDs so unrelated items are not merged.

The underlying 50,224 listing rows are retained. There are 31,793 unique primary
image URLs; conservative garment exclusions reduce the searchable group count.
The details dialog offers other listings with the same primary image URL and
recorded brand/store. Selecting one loads its own sizes, historical/estimated
price, ratings, description and image gallery. A shared photo does not prove a
confirmed Amazon variation family, so these are explicitly described as related
listings. Sizes from a listing's size chart do not invent separate image/price
combinations. When two sizes share their recorded primary photo, the photo stays
the same. Different images are shown only if that selected listing provides them.

The master Parquet is not reduced or rewritten: preserving underlying variants
keeps size-specific filters and original listing evidence intact. Live checks
for the user's two COOFANDY examples and blue/double-pocket query are in
`docs/verification/image-group-live-check.json`.
# Request checks and field examples

The model receives each searchable field's meaning, illustrative values and a
correct SQL example in addition to column types and categorical vocabulary.
`request_guard.py` independently enforces common explicit color, material,
size, sleeve, pattern, pocket and price requests after checking SQL safety.
It repairs omitted or incorrect predicates for those requested fields before
execution. Semicolon-separated fabrics and pipe-separated sizes/colors use
exact token matching. Other model conditions are preserved; ambiguous mixed
conditions or paired alternatives across fields are rejected rather than
rewritten with guessed meaning. This is not a parser for every possible sentence.

Sorting is chosen from the user's words by the app. A budget alone never implies
cheapest-first; recommended ranking uses office/comfort listing evidence first.
Explicit cheapest/highest-rated/most-reviewed requests retain their chosen order.
Office and comfort are soft preferences, not mandatory keyword matches.
The API includes `enforced_fields` and `preferences` for inspection. Correct SQL
does not establish that the dataset's visual labels or listing claims are correct.

