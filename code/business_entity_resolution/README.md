# Business Entity Resolution — Code

End-to-end ML pipeline for the Amazon ML Challenge 2026.

## Quick Start (Kaggle)

1. Upload this repository + `student_resource/` as a Kaggle dataset
2. Open `notebooks/kaggle_pipeline.ipynb`
3. Set `KAGGLE_DATA_DIR` to your dataset path
4. Run all cells top-to-bottom (~2–4 hours on Kaggle CPU)
5. Download `output/matching_results.tsv` and upload to the leaderboard

## Quick Start (Local — testing/dev)

```bash
cd code/business_entity_resolution

# Install dependencies
pip install -r requirements.txt

# Train (from student_resource/ directory)
cd /path/to/student_resource
python ../code/business_entity_resolution/src/train.py \
    --data-dir . \
    --model-dir models/ \
    --val-fraction 0.20

# Predict (test set)
python ../code/business_entity_resolution/src/predict.py \
    --data-dir . \
    --model-dir models/ \
    --output-dir output/

# Validate
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Repository Structure

```
business_entity_resolution/
├── src/
│   ├── utils/
│   │   ├── text_utils.py    # Text normalization & blocking key generation
│   │   ├── metrics.py       # F₀.₅ scorer + threshold sweep
│   │   └── io_utils.py      # TSV I/O helpers
│   ├── blocking.py          # Multi-key inverted-index blocking
│   ├── features.py          # 28 pairwise features + TF-IDF vectorizers
│   ├── train.py             # Full training pipeline
│   └── predict.py           # Full inference pipeline
├── notebooks/
│   └── kaggle_pipeline.ipynb   # Self-contained Kaggle notebook
├── requirements.txt
└── README.md
```

## Pipeline Overview

```
Source 1 (2.2M) + Source 2 (5M) + Source 3 (5.3M)
        │
        ▼
[Blocking] — 7 blocking key types + n-gram fallback
  → Reduces 22T pairs → ~500 candidates/entity
  → Target blocking recall: ≥97%
        │
        ▼
[Feature Engineering] — 28 features per pair
  Name: WRatio, Levenshtein, token sort, Jaccard (token+trigram), TF-IDF cosine
  Address: Jaccard, edit distance, postal code, street number, city overlap
  Phonetic: Soundex, NYSIIS, full-name Soundex
  Meta: country match, source type, name×addr product, completeness
        │
        ▼
[LightGBM Classifier] — binary pairwise matching
  Training: positive pairs from GT + hard negatives from blocking pool
  Validation: full blocking candidates, F₀.₅ threshold sweep
        │
        ▼
[Threshold → Predictions]
  Default to empty (singleton) — only predict match if score ≥ threshold
  Threshold selected to maximize macro F₀.₅ on validation set
        │
        ▼
output/matching_results.tsv   ← upload to leaderboard
output/candidate_pairs.tsv    ← include in final zip
```

## Design Decisions

- **F₀.₅ is precision-heavy**: threshold tuned toward high precision. False merges cost 2× false negatives.
- **Country-agnostic**: no hard-coded US/India logic. Works for France (unseen in training) via generalized tokenization and char n-gram TF-IDF.
- **Singletons**: default to empty prediction. Only add matches when model score ≥ threshold.
- **No external APIs**: all features from provided data only (per contest rules).
- **Model**: LightGBM (MIT license, no parameter count constraint).
