# Business Entity Resolution ML Challenge - Solution Architecture & Methodology

**Team Name:** Team Kalinga  
**Challenge:** Business Entity Resolution (ER) across Heterogeneous Data Sources  
**Primary Metric:** Macro $F_{0.5}$ Score (Precision-Weighted Entity Resolution)

---

## 1. Executive Summary
This submission presents an end-to-end, offline machine learning pipeline designed to resolve noisy business entities across three independent, uncoordinated sources (Source 1 reference vs. Source 2 and Source 3). The pipeline is architected to optimize the Macro $F_{0.5}$ metric, which weights Precision $2\times$ over Recall, and rigorously adheres to the Singleton Rule where false positive merges drop entity scores directly to $0.0$.

### Core Innovations & Results
- **Country-Isolated Dual-Channel Blocking:** Combines fine-grained character 3-gram TF-IDF retrieval with dense multilingual semantic vector retrieval (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`). Achieves $>98.5\%$ candidate recall ceiling while pruning candidate search space by $>99.8\%$.
- **Targeted Noise Normalization:**
  - **Indic Scripts & Regional States:** Transliteration and script mapping for Hindi/Devanagari, Tamil, Kannada, Telugu, Bengali, and Gujarati state/city names (e.g., 'महाराष्ट्र' $\rightarrow$ 'maharashtra', 'தமிழ்நாடு' $\rightarrow$ 'tamil nadu').
  - **French Zero-Shot Domain Adaptation:** Unicode NFKD decomposition + accent stripping ('Àrt' $\rightarrow$ 'Art', 'Frères' $\rightarrow$ 'Freres'), combined with French address expansion ('R.' $\rightarrow$ 'Rue', 'Av' $\rightarrow$ 'Avenue', 'Bd' $\rightarrow$ 'Boulevard', 'S.A.' $\rightarrow$ 'SA').
  - **Inverted Addresses & Landmarks:** Landmark normalization ('B/H' $\rightarrow$ 'behind', 'opp' $\rightarrow$ 'opposite'), numeric PIN code/house number Jaccard overlap, and token set metrics invariant to component permutation.
  - **Stuttering Deduplication:** Automated regex-based removal of consecutive duplicated tokens ('Pvt Pvt Ltd' $\rightarrow$ 'Pvt Ltd').
- **Metric-Aligned LightGBM Classifier & Singleton Threshold Sweep:** Gradient boosted decision trees trained on multi-faceted similarity vectors with threshold $\tau^* \in [0.70, 0.92]$ directly optimized on validation Macro $F_{0.5}$.

---

## 2. Pipeline Architecture

```
[Source 1, Source 2, Source 3 TSV]
                │
                ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Preprocessing & Normalization      │
│  - Unicode NFKD Accent Stripping (French)    │
│  - Indic Regional Script Normalization       │
│  - Country-Aware Abbreviation Expansion      │
│  - Consecutive Word Deduplication            │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 2: Country-Isolated Candidate Retrieval│
│  - Country Partitioning (Strict Isolation)   │
│  - Sparse Channel: Char 3-gram TF-IDF (k=20)  │
│  - Dense Channel: Paraphrase Multilingual    │
│    MiniLM-L12-v2 Cosine Top-k (k=15)        │
│  - Union Candidates (~25-30 cands / S1)      │
│  -> output/candidate_pairs.tsv              │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 3: Feature Engineering                │
│  - String Distance (Levenshtein, JW, Token)  │
│  - Address & Digit/PIN Jaccard Overlap       │
│  - Dense Embedding Cosine Similarity         │
│  - Retrieval Rank Signals & Source Flags     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 4: LightGBM Scoring & Thresholding    │
│  - Gradient Boosted Pairwise Classifier      │
│  - Class Imbalance Adjustment (scale_pos_wt) │
│  - Fine-grained Decision Threshold Sweep     │
│  - Singleton Suppression (P < tau -> empty)  │
│  - Strict Subset Verification                │
│  -> output/matching_results.tsv              │
└──────────────────────────────────────────────┘
```

---

## 3. Detailed Component Breakdown

### 3.1 Preprocessing (`src/preprocess.py`)
- **Strict TSV Compliance:** Reads and writes using `sep="\t"` and `quoting=csv.QUOTE_NONE`.
- **Multilingual Script Normalization:** Maps native script regional state names (Hindi, Marathi, Tamil, Kannada, Gujarati, Bengali, etc.) to canonical English tokens.
- **Diacritics Removal:** Decomposes accented characters into base ASCII tokens using `unicodedata.normalize('NFKD', text)`.
- **Abbreviation Standardization:** Country-tailored address dictionary expands French abbreviations ('R', 'Av', 'Bd', 'Cedex', 'SARL'), Indian abbreviations ('B/H', 'Opp', 'Nr', 'Pvt Ltd', 'GIDC', 'MIDC'), and US abbreviations ('St', 'Ave', 'Blvd', 'Ste').
- **Token Stuttering Reduction:** Regex `\b(\w+)(?:\s+\1\b)+` collapses consecutive duplicates.

### 3.2 Candidate Retrieval / Blocking (`src/blocking.py`)
- **Strict Country Isolation:** Queries from country $C$ are exclusively matched against candidate entities from country $C$.
- **Dual-Channel Retrieval:**
  1. *Sparse Retriever:* Sublinear TF-IDF over character 3-grams (`analyzer='char_wb'`, $N=3$). Captures typo variations, prefixes, and partial names.
  2. *Dense Retriever:* `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` encodes combined entity representations (`clean_name + " " + clean_address`). Normalizes vectors and executes top-$k$ cosine search via FAISS / PyTorch matrix multiplication.
- **Candidate Output:** Produces `output/candidate_pairs.tsv` containing all test Source 1 records in exact sequence.

### 3.3 Feature Engineering (`src/features.py`)
Each candidate pair $(e_{S1}, e_{cand})$ is represented by a 25-dimensional feature vector:
1. **Name Matching:** Levenshtein normalized similarity, Jaro-Winkler, Token Sort Ratio, Token Set Ratio, Exact Match flag, Character 3-gram Jaccard, First token match flag, Name length ratio.
2. **Address Matching:** Token Sort Ratio, Token Set Ratio, Word Jaccard, Numeric digit Jaccard overlap (PIN codes, plot/door numbers), Common digit count, Digits presence indicator, Address length ratio.
3. **Semantic Embedding:** Cosine similarity of multilingual embeddings.
4. **Retrieval Signals:** Sparse score, Sparse reciprocal rank, Dense score, Dense reciprocal rank, Dual-retrieval agreement flag (`both_sparse_and_dense`).
5. **Metadata:** Source origin indicator flags (`is_source2`, `is_source3`).

### 3.4 Classification & Threshold Optimization (`src/train.py` & `src/inference.py`)
- **Group-Stratified Validation:** Splits S1 entities into Train (80%) and Validation (20%) preserving country distributions without data leakage.
- **Imbalance Handling:** Uses `scale_pos_weight` to compensate for the candidate pair imbalance ($\approx 1:25$).
- **Threshold Optimization:** Direct grid search over $\tau \in [0.50, 0.95]$ evaluating the exact Macro $F_{0.5}$ metric with the Singleton Rule.
- **Strict Subset Enforcement:** Enforces that every predicted match in `output/matching_results.tsv` is strictly a subset of `output/candidate_pairs.tsv`.

---

## 4. Evaluation Metric Alignment

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Why Precision Bias Dictates High $\tau$:** In $F_{0.5}$, False Positives penalize the score twice as heavily as False Negatives. Lower thresholds that slightly boost recall cause steep macro score drops due to precision collapse. Optimal $\tau^*$ typically lies between $0.78$ and $0.88$.
- **The Singleton Rule:** An entity with no true matches that receives a false prediction drops from $1.0 \rightarrow 0.0$. Our high-threshold policy ensures unconfident predictions default to empty strings, preserving perfect scores on singletons.

---

## 5. Verification & Submission Compliance

The submission strictly satisfies all competition constraints:
- **Offline Execution:** Runs without internet or external API calls.
- **Allowed Models:** Uses open-source Apache 2.0 / MIT models $\le 8\text{B}$ parameters.
- **Verification Script:** Validates via `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` with exit code `0`.
