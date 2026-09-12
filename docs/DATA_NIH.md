# DATA_NIH — NIH ChestX-ray8 / NIHCC dataset

## Source

The Phase 1 dataset is **NIH ChestX-ray8** (a.k.a. NIHCC), a public chest
X-ray collection released by the NIH Clinical Center. It contains **112,120
frontal-view X-ray images from 30,805 unique patients**, with **14 disease
labels** mined from associated radiology reports via NLP.

- Original release: Wang et al., *ChestX-ray8: Hospital-Scale Chest X-ray
  Database and Benchmarks on Weakly-Supervised Classification and
  Localization of Common Thorax Diseases*, CVPR 2017.
- NIHCC update (2017) refreshed the labels and added `Data_Entry_2017.csv`.
- Hosting: NIH distributes the dataset via Box public links. The
  `scripts/download_nih_subset.py` script can either consume a pre-downloaded
  copy (`--source local`) or download a custom list of URLs
  (`--source url-list`).

## License

The dataset is released for **research and educational use**; there is no
formal license file. By using it we agree to:

- Not attempt to re-identify patients (the dataset is de-identified).
- Not redistribute the raw images outside this project.
- Acknowledge the NIH Clinical Center in any publication or presentation.

## Schema

`Data_Entry_2017.csv` columns:

| Column | Description |
|---|---|
| `Image Index` | Filename (e.g. `00000001_000.png`). Patient + follow-up index. |
| `Finding Labels` | Pipe-delimited list of one or more of the 14 disease labels, or `No Finding`. |
| `Follow-up #` | Which follow-up visit this image belongs to. |
| `Patient ID` | Integer patient identifier. |
| `Patient Age` | Patient age in years. |
| `Patient Gender` | M / F. |
| `View Position` | PA or AP. |
| `OriginalImage[Width/Height]` | Native pixel dimensions. |
| `OriginalImagePixelSpacing_x/y` | Physical spacing in mm. |

### The 14 NIH disease labels

`Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Effusion`,
`Emphysema`, `Fibrosis`, `Hernia`, `Infiltration`, `Mass`, `Nodule`,
`Pleural_Thickening`, `Pneumonia`, `Pneumothorax`. Plus the meta label
`No Finding`.

## Mapping NIH labels -> our six findings

NIH has more labels than we need and uses different vocabulary. The mapping
is the single source of truth in
`backend/src/cxt/infrastructure/data/nih.py` and is used by the manifest
builder.

| Our finding (`FindingCategory`) | NIH label(s) |
|---|---|
| `PNEUMOTHORAX` | `Pneumothorax` |
| `PLEURAL_EFFUSION` | `Effusion` |
| `CARDIOMEGALY` | `Cardiomegaly` |
| `CONSOLIDATION` | `Consolidation`, `Infiltration`, `Pneumonia` |
| `NODULE_OR_MASS` | `Nodule`, `Mass` |
| `FRACTURE` | `Fracture` |
| `NO_FINDING` | `No Finding` (or a case with no positive label for any of the above) |

A case with a positive label for a target finding is a **positive case**; a
case with no positive label for any target finding is a **negative case**.

## Label noise

The NIH labels are **NLP-mined from radiology reports** and are known to be
noisy. Reported multi-label accuracy of the NIH-mined labels against
radiologist adjudication is in the 70-90 % range depending on the label.
The `label_source` column in `manifest.csv` records whether the label is
NLP-mined, manually corrected, or stub. The eval report (`docs/EVALUATION`)
breaks down the model's accuracy per finding to avoid hiding noise inside
a single flat accuracy.

## Subset selection

The Phase 1 evaluation uses a **balanced 30-60 case subset**:

- Up to 5 positive cases per finding (configurable via
  `scripts/build_manifest.py --n-per-finding`).
- Up to 5 negative cases (`--n-negatives`).
- Cases are deterministically assigned to `train`/`val`/`test` (70/15/15)
  by hashing the `case_id`, so re-runs are reproducible.

## Reproducing the subset

```bash
cd backend

# Option A — synthetic stubs for development (no network)
python -m scripts.download_nih_subset --source stub --n-per-finding 5
python -m scripts.build_manifest --n-per-finding 5 --n-negatives 5

# Option B — local NIH copy (real radiographs)
# 1. Visit https://nihcc.app.box.com/v/ChestXray-NIHCC and download
#    images_001.zip (~3.6 GB, ~9000 images) and Data_Entry_2017.csv.
# 2. Unzip images_001.zip into a directory (e.g. C:\nih\images_001\).
python -m scripts.download_nih_subset \
    --source local \
    --local-dir C:\nih\images_001 \
    --metadata-csv C:\nih\Data_Entry_2017.csv
python -m scripts.build_manifest --n-per-finding 5 --n-negatives 5
```

Or via the cross-platform dev runner (equivalent):

```powershell
# stub (default; no network)
python scripts\dev.py data

# real NIH
python scripts\dev.py data `
    --source local `
    --local-dir C:\nih\images_001 `
    --metadata-csv C:\nih\Data_Entry_2017.csv `
    --n-per-finding 5
```

Outputs:

- `data/raw/PROVENANCE.json` — checksums + source URL/path per file.
- `data/raw/images/*.png` — the raw images.
- `data/processed/manifest.csv` — the curated case table the API serves.
- `data/processed/images/*.png` — the curated images (subset of raw).
- `data/processed/case_index.json` — counts by finding and split.

> **Tip:** only one of the 12 Box zips is needed for 30-60 cases
> (~9000 images, 3.6 GB). Increase `--n-per-finding` if you want more
> diversity per category. NIH labels are NLP-mined and noisy
> (see [Label noise](#label-noise) above); treat per-finding accuracy
> in the eval report as a directional signal, not a clinical ground truth.

## Privacy and data flow

Images are de-identified and public. The dev server serves them directly
via FastAPI's `StaticFiles` mount on `/images`. In a future Phase 2 build
behind a real auth boundary, replace this with signed URLs (see
`docs/AUTH.md` when it lands). The data flow is described in plain
language in `docs/KAGGLE_RUNTIME.md` (to be written in Step 4); the
educational-use banner in the UI makes the data flow transparent to the
user.

## Phase 1.5 — MIMIC-CXR swap

The `ICaseRepository` port means swapping to MIMIC-CXR is a new
`MimicLoader` + new repository, with no domain change. We start the
PhysioNet credentialing application in parallel (free, ~1-2 weeks). See
`docs/DATA_MIMIC.md` (Phase 1.5) for the schema differences.
