# Shirt query application

The merged 50,224-row master powers this flow:
query -> OpenAI-compatible LLM -> validated SQL -> DuckDB scans Parquet -> ranked products.
This is a working query application, not a claim of evaluated recommendation accuracy.

## Run

Double-click `Start_Query_App.bat` in this folder, then open http://127.0.0.1:8010.
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
master Parquet and locks its settings. An execution timer bounds each search.

All matches are considered before the app's top-k limit. Price provenance tiers
put exact historical prices first; within a tier a Bayesian rating average with
20 prior reviews ranks results. Rating count breaks ties. Requests for cheapest,
highest-rated or most reviewed select separate
deterministic sorts; price estimates remain labeled. This is a transparent
baseline, not the original paper's cascaded review-embedding recommender. No live
stock, personalized learning or ranking evaluation is implemented yet.

Estimated INR prices use the existing project's conversion (83 INR/USD).
Button color, live inventory, delivery and exact sizes are absent and explicitly
unsupported. Empty matches stay empty rather than silently relaxing user filters.
The default includes uncertain attributes; the UI lets users exclude them.

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
real Parquet ranking, the compatible API adapter and FastAPI behavior. All 16 pass.
Live provider checks are separate from fixtures: NVIDIA completed two real search
queries with returned filter constraints checked. See `docs/live-provider-check.json`.

Sources used for implementation:
- https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat
- https://duckdb.org/docs/stable/operations_manual/securing_duckdb/overview
- https://docs.api.nvidia.com/nim/reference/meta-llama-3_3-70b-instruct
- https://openrouter.ai/docs/guides/features/structured-outputs
- https://docs.nvidia.com/nim/large-language-models/2.0.10/get-started/advanced/get-started-nemotron-3.5-lightning.html
- https://docs.api.nvidia.com/nim/docs/run-anywhere

Saved image dimensions reflect the images processed at max-side 1,500, not a
guarantee that every remote URL will remain reachable or have an original larger image.
