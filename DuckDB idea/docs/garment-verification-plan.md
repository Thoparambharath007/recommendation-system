# Dima806 Kaggle garment verification

Approved scope: use Dima806 alone on a varied fixed sample, then unique catalog
images. No training, no app changes, no deletion or replacement of the master.

## Implementation

- [x] Test URL deduplication, ID mapping, candidate decisions, and resume checks.
- [x] Implement dependency-light input, checkpoint, and export helpers.
- [x] Implement pinned-model batched inference on two independent T4 devices,
  bounded parallel downloads, failure recording, and checkpoint commits.
- [x] Prepare 300 unique sample images covering sleeve lengths, patterns,
  colors and suspicious garment types. Labels remain blank until verified.
- [x] Build a Kaggle notebook and self-contained input ZIP. Include the exact
  current catalog in the ZIP without moving or editing it.
- [x] Run local unit/integration checks and a real CPU model smoke test if the
  local environment supports it. Do not claim a Kaggle GPU test was performed.

## Output rules

Class Shirt is the shirt-only target. T-shirt and Polo retain separate labels.
Strong predictions become shirt/non-shirt candidates; low scores/margins and
download failures remain uncertain. Initial thresholds are heuristic, not
validated accuracy. Shared images are classified once and mapped to all IDs;
shared URLs/content hashes never automatically delete products.

SQLite stores raw per-image predictions. Same model/code identity can resume
from sample to full catalog; output thresholds can be changed without inference.
Archives preserve checkpoints and review images. Current app data stays intact.
