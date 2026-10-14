# Catalog merge and natural-language SQL application

Authorized scope: replace the master color tags with Kaggle v3.2 predictions, populate
pattern, and implement query -> LLM-generated SQL -> validation -> DuckDB/Parquet ->
ranked recommendations. The user accepts uncertainty flags and chose an
OpenAI-compatible API. Implement inline; no extra pilot or manual labeling gate.

1. Test merge alignment, duplicate/missing IDs and image URL consistency. Preserve
   the exact pre-merge master and unrelated fields. Include needs_review predictions;
   failed masks have unknown attributes. Export Parquet, CSV and provenance.
2. Test a restricted SQL interface: one SELECT parent_asin FROM shirts with a WHERE
   expression. Reject other tables, subqueries, mutations and unapproved functions.
   DuckDB only gets access to the configured Parquet. Bound execution with a timeout.
3. Implement an OpenAI-compatible JSON completion adapter, real catalog schema and
   vocabularies in its prompt, exact multi-color token matching and no invented fields.
   Unknown button color or live inventory requests are explicitly unsupported.
4. Add local FastAPI UI, product images, provenance badges, generated SQL, count,
   relevance filters and deterministic quality ranking. Do not silently relax filters.
   Prices retain exact/estimated provenance and 2023 dataset context.
5. Verify against real merged data, API adapter fixtures, and a live LLM request if
   credentials and endpoint access permit. Record limitations. No accuracy claim
   without labeled references. Saved processing sizes answer image-resolution question.

New files stay in DuckDB idea. Test scratch goes in JUNK created by qoder.
No unrelated legacy application changes. Existing master is backed up before mutation.
