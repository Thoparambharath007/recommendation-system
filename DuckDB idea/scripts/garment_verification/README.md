# Kaggle garment verification — Dima806 only

1. Import `Kaggle_Start.ipynb` as a Kaggle notebook.
2. Upload `shirt_garment_verification_v1.zip` as its input dataset. The ZIP
   includes the current master, code, 300-image sample and tests. Do not attach
   other versions of this package; there is no need to attach another master.
3. Select **GPU T4 x2** and enable **Internet**.
4. Run setup/hardware/sample cells. `LIMIT=0` processes the entire sample.
5. Inspect sample results or fill a small reference set if measuring accuracy.
6. Set `RUN_FULL_DATASET=True` and run the full-catalog cell. It reuses sample
   predictions, processes each unique URL once, and maps results to every ID.
7. Run the last archive cell and download `garment_results_backup.zip` before
   ending the session. Save a Kaggle version with outputs as an additional copy.

## What it does

This uses an already fine-tuned 15-class ViT classifier, not a new training job.
Shirt, T-shirt, Polo, Blazer, Jacket and other categories remain distinct.
The target for the shirt-only catalog is **Shirt**. Strong T-shirt or Polo
predictions enter the non-shirt candidate partition under that specific policy;
they are not deleted, and their original garment labels are retained.

The initial score 0.85 and runner-up margin 0.20 are heuristic thresholds, not
accuracy guarantees. Low-confidence predictions, unavailable images and pending
images remain in the uncertain output. Reference labels must come from actual
verification, not title-based guesses. Sample groups are existing metadata hints.
Whole-image classification can confuse outfits, folded items and layered clothes.

## Output files

- `products_annotated.parquet`: every input row and original column, plus
  `garment_image_key`, type, score, margin, status, reason and pinned model ID.
- `shirt_candidates.parquet`: strong Shirt predictions only.
- `non_shirt_candidates.parquet`: strong predictions of the other categories.
- `uncertain_products.parquet`: low scores/margins, failures and pending work.
- `product_verdicts.csv`: lightweight per-product decisions.
- `image_predictions.parquet` / `.csv`: one raw prediction per processed URL,
  all class scores and downloaded content hashes. Duplicate URLs/content hashes
  do not automatically remove legitimate product IDs or variants.
- `checkpoint.sqlite`: resumable raw predictions.
- `review.html` + `previews/`: view up to 400 sample cards; keep these together.
- `reference_labels.csv`: blank template for optional measured evaluation.
- `summary.json`, `run_config.json`: processing counts and provenance.

## Resume and performance

Same-session reruns skip existing image keys. For a new Kaggle session, attach
the output ZIP and use the notebook's restore cell before running the full cell.
Restore the same output folder name. A changed model/code identity requires a new
output folder; this prevents combining incompatible predictions.
Threshold changes re-export saved scores without requiring re-inference.
Use `--retry-failed` to deliberately retry failed downloads.

Each T4 gets its own model copy and image batches. Downloads use 12 bounded
threads; model input is 224x224. No segmentation or slow generative model is run.
Estimate duration from actual sample throughput; no GPU runtime is promised.
The notebook installs pinned model libraries into an isolated worker overlay;
it does not downgrade Kaggle's global Hugging Face Hub or replace CUDA PyTorch.

Local commands from the project root:

```powershell
python -m unittest discover -s scripts/garment_verification/tests -v
python scripts/garment_verification/prepare_package.py
```

The live app and `data/shirts_master.parquet` are not modified by this setup.
Do not replace the live master with candidates until the results are checked.

Model: https://huggingface.co/dima806/clothes_image_detection
Pinned revision: `649dbb348805d6b711a6da28dc58531259280b77`.
