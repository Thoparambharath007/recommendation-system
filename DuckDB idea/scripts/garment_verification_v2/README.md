# Garment verification v2

Import Kaggle_Start.ipynb and attach shirt_garment_verification_v2.zip.
Enable Internet and GPU T4 x2. Run setup, GPU check, full catalog, and backup cells.
Download garment_results_v2_backup.zip when complete.

This uses pinned Marqo FashionSigLIP with three descriptions per category,
including Bodysuit, Romper, Vest and Jumpsuit. It performs image inference, not
training. Only Shirt is a shirt candidate; polo and T-shirt remain separate.
Scores are relative to these descriptions, not calibrated correctness probabilities.
The 0.85 score and 0.20 margin gates are conservative heuristics, not measured accuracy.
Ambiguous, failed and pending images remain separate. Predictions do not replace
the live master dataset automatically.

There is one inference per unique URL, using both GPUs in batches. All original
product IDs and size variants are preserved in products_annotated.parquet.
Use summary.json's measured speed for runtime estimates. No 31k-image runtime
is promised from CPU tests.

Do not restore v1/Dima checkpoints into v2. After a session interruption, upload
the v2 backup, set RESTORE_ZIP in the setup cell, and rerun the full-catalog cell.
Keep the same model, code and output directory. The runner skips checkpointed images.

Source: https://huggingface.co/Marqo/marqo-fashionSigLIP
Pinned revision: c56244cc94f92419e8369fa71efdaf403b124ce8
The local OpenCLIP loader uses pinned safetensors and configuration; no remote
Python model code is executed.
