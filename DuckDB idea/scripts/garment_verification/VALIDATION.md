# Validation

- Input master: 50,224 rows; 31,793 unique valid image URLs.
- Prepared sample: 300 unique URLs, 109 full sleeves, 86 half sleeves and 105
  unknown sleeves; all 12 existing pattern categories represented. These are
  existing metadata labels, not newly verified reference labels.
- Tests cover URL deduplication without ID loss, invalid IDs, candidate gates,
  SQLite resume/config identity, shared-image exports, failed downloads,
  independent worker dispatch, resume without downloading/loading models again,
  and evaluation precision/retention/false acceptance.
- A real local CPU smoke run loaded the pinned Safetensors weights, downloaded
  four catalog images, and completed inference/export. Three predicted Shirt
  and one predicted Polo; all were below the initial confidence gates and stayed
  uncertain. This confirms execution, not model accuracy on the catalog.
- Root model weights match the author's checkpoint-4230 artifact hash.
- CUDA inference, two physical T4s, Kaggle installation and GPU throughput have
  not been exercised locally. The script has a mocked two-device dispatch test.
- The source master checksum is recorded in package_manifest.json and checked
  by the notebook. Packaging does not modify it. No credentials are bundled.

Initial thresholds are deliberately labeled heuristic. Do not silently lower
them to manufacture a clean dataset. Compare sample predictions with reference
labels before treating candidates as confirmed shirts.
