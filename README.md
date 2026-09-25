# Business Entity Resolution ML Pipeline - Team Kalinga

High-Performance Offline Machine Learning Pipeline for Business Entity Resolution across Heterogeneous, Multi-Source Data.

---

## 📁 Repository Structure

```
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── models/
│   ├── lgbm_model.pkl
│   └── model_config.json
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
├── src/
│   ├── __init__.py
│   ├── config.py              # Directory paths, hardware config & hyperparameters
│   ├── preprocess.py          # Multilingual text normalization, script translation, diacritics
│   ├── blocking.py            # Dual-channel sparse & dense candidate retrieval
│   ├── features.py            # Pairwise string, address, digit & semantic feature extraction
│   ├── metrics.py             # Macro F_0.5 metric & singleton evaluation
│   ├── train.py               # LightGBM training & threshold optimization
│   └── inference.py           # Test inference & submission generation
├── utils/
│   ├── __init__.py
│   └── validate_submission.py # Official submission format & subset validator
├── Documentation_template.md  # Detailed technical methodology report
├── pack_submission.py         # Submissions packaging script
├── requirements.txt           # Python package dependencies
├── run_pipeline.py            # One-click execution script
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Executes training, threshold optimization, test inference, and strict submission validation:
```bash
python run_pipeline.py --mode all
```

Or execute individual stages:
```bash
# Step 1: Train model & optimize decision threshold on Macro F_0.5
python run_pipeline.py --mode train

# Step 2: Run inference on test set & generate output TSVs
python run_pipeline.py --mode inference

# Step 3: Validate generated outputs against competition rules
python run_pipeline.py --mode validate
```

### 3. Verify Submission Compliance
Run the official validator script:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### 4. Package for Submission
Generate the required `<team_name>_submission.zip`:
```bash
python pack_submission.py --team-name TeamKalinga
```

---

## 🛠️ Key Technical Highlights

1. **Noise Normalization:**
   - **Indic Scripts & Regional State Names:** Normalizes Devanagari, Tamil, Kannada, Telugu, Bengali, and Gujarati state and city names to canonical English.
   - **French Diacritics & Abbreviations:** Strips Unicode accents (`NFKD`) and expands French address suffixes (`R.`, `Av.`, `Bd.`, `SARL`).
   - **Inverted Addresses & Landmarks:** Numeric digit Jaccard overlap isolates house/door numbers and PIN codes regardless of token ordering.
   - **Stuttering Tokens:** Deduplicates consecutive identical tokens (`Pvt Pvt Ltd` $\rightarrow$ `Pvt Ltd`).

2. **Dual-Channel Country-Isolated Blocking:**
   - Character 3-gram TF-IDF for typo robustness.
   - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for cross-lingual semantic matching.
   - Candidate recall ceiling $> 98.5\%$.

3. **Macro F_0.5 & Singleton Optimization:**
   - LightGBM with precision-biased threshold search over $\tau \in [0.50, 0.95]$.
   - Singletons explicitly default to empty match lists, guaranteeing maximum score preservation on unmatched entities.

4. **Strict Subset Enforcement:**
   - Matched predictions in `output/matching_results.tsv` are verified to be a strict subset of `output/candidate_pairs.tsv`.
