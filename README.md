# 🌍 Automated SDG Mapping for MAHE Research Articles

> Automatically mapping MAHE-affiliated research articles to the most relevant **UN Sustainable Development Goals (SDGs)** using machine learning, word embeddings, and semantic similarity scoring.

**M.Sc. Data Science — DLTM Project**  
Manipal Academy of Higher Education (MAHE), 2025–27  
**Author:** Pranam Acharya | **Guide:** Kavitha Karimbi Mahesh, Associate Professor

---

## 📌 Project Overview

The United Nations' 17 Sustainable Development Goals (SDGs) provide a globally accepted framework for addressing social, economic, and environmental challenges. Manually tagging thousands of research articles with relevant SDGs is time-consuming and subjective.

This project builds an **automated, machine learning-based pipeline** that:
- Scrapes MAHE research articles (title + abstract) from the institutional portal
- Trains classifiers on the **OSDG Community Dataset** (34,005 labeled texts)
- Maps each MAHE article to its **top-5 most relevant SDGs** with confidence scores
- Implements two complementary tracks: supervised classification and cosine similarity scoring

**No keyword rules. No manual labeling. Fully learned from data.**

---

## 🏆 Key Results

| Method | Representation | Macro F1 | Category |
|--------|---------------|----------|----------|
| **SVM** | **TF-IDF** | **0.8491** | **Best Overall** |
| SVM | BERT | 0.8040 | BERT Embeddings |
| XGBoost | BoW | 0.7981 | Classical ML |
| TextCNN | CNN | 0.7930 | Deep Learning |
| SVM | GloVe | 0.7797 | Word Embeddings |
| SVM | Word2Vec | 0.7605 | Word Embeddings |

**Finding:** TF-IDF with SVM achieves the best Macro F1 of **0.8491** — outperforming BERT and CNN. This is consistent with published findings on short domain-specific text classification, where SDG-specific keywords carry more discriminative power than contextual patterns.

---

## 🗂️ Repository Structure

```
sdg-mapping-mahe/
│
├── 📄 README.md
├── 📄 .gitignore
├── 🐍 scraper_chrome.py               ← MAHE portal scraper (undetected-chromedriver)
│
├── 📓 step1_data_collection.ipynb     ← Download OSDG + load MAHE corpus
├── 📓 step2_eda.ipynb                 ← EDA, label distribution, 80/20 split
├── 📓 step3_preprocessing.ipynb       ← TF-IDF, embedding, BERT text variants
├── 📓 step4_baseline.ipynb            ← TF-IDF + classifiers (SVM, LR, XGB, RF)
├── 📓 step5_embeddings.ipynb          ← BoW, Word2Vec, GloVe comparison
├── 📓 step6_similarity.ipynb          ← Cosine similarity + combined ranking
├── 📓 step7_bert.ipynb                ← BERT embeddings + zero-shot similarity
├── 📓 step7b_cnn.ipynb                ← TextCNN (Kim 2014)
├── 📓 step8_clustering_evaluation.ipynb ← K-Means, DBSCAN, final evaluation
│
├── 📁 data/
│   ├── clean/                         ← Preprocessed datasets (train/test splits)
│   ├── results/                       ← Top-5 SDG predictions for all MAHE articles
│   └── plots/                         ← All 17 visualisation plots
│
└── 📄 SDG_Project_Report.docx         ← Full project report
```

---

## 🔄 Methodology

The pipeline is divided into two parallel tracks:

### Track A — Supervised Classification
```
OSDG labeled corpus (34,005 texts)
        ↓
  Text preprocessing
        ↓
Feature extraction: TF-IDF / BoW / Word2Vec / GloVe / BERT / CNN
        ↓
Classifiers: SVM · Logistic Regression · XGBoost · Random Forest
        ↓
Predict top-5 SDGs for 854 MAHE articles
```

### Track B — Similarity Scoring
```
MAHE article text
        ↓
GloVe / BERT embeddings (300d / 384d)
        ↓
Cosine similarity vs 17 SDG reference profiles (official UN text)
        ↓
Softmax normalisation → top-5 SDG ranking
```

### Combined Output
Both tracks are combined via **50/50 weighted average** into a final robust top-5 ranking. Inter-method agreement (30.2%) confirms the two tracks capture complementary aspects of SDG relevance.

---

## 📊 Dataset

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| OSDG Community Dataset v2024.04 | [Zenodo](https://zenodo.org/records/11441197) | 34,005 labeled texts | Training & evaluation |
| MAHE Research Articles | researcher.manipal.edu (scraped) | 854 articles | Application corpus |

**Train/Test split:** 80/20, stratified by SDG label  
**Class imbalance:** 4.7x ratio — handled with `class_weight='balanced'`

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost gensim sentence-transformers
pip install torch torchtext nltk matplotlib seaborn wordcloud tqdm
pip install undetected-chromedriver selenium beautifulsoup4
python -m spacy download en_core_web_sm
```

### Step-by-step execution

**Step 0 — Scrape MAHE data** (run from Anaconda Prompt, not Jupyter)
```bash
python scraper_chrome.py
# Produces: mahe_sdg_publications.csv
```

**Steps 1–8 — Run notebooks in order**

| Notebook | What it does | Key output |
|----------|-------------|------------|
| `step1_data_collection.ipynb` | Download OSDG + load MAHE corpus | `data/clean/osdg_clean.csv` |
| `step2_eda.ipynb` | EDA + train/test split | `data/clean/X_train.csv` etc. |
| `step3_preprocessing.ipynb` | Build 3 text variants | `*_tfidf.csv`, `*_emb.csv`, `*_bert.csv` |
| `step4_baseline.ipynb` | TF-IDF + 4 classifiers | `mahe_top5_tfidf.csv` |
| `step5_embeddings.ipynb` | Word2Vec + GloVe | `mahe_top5_glove.csv` |
| `step6_similarity.ipynb` | Cosine similarity ranking | `mahe_top5_combined.csv` |
| `step7_bert.ipynb` | BERT embeddings | `mahe_top5_bert.csv` |
| `step7b_cnn.ipynb` | TextCNN | `mahe_top5_cnn.csv` |
| `step8_clustering_evaluation.ipynb` | Clustering + final output | `mahe_final_predictions.csv` |

### Notes
- GloVe download (~380MB) happens automatically in Step 5. If blocked by your network, download `glove.6B.zip` from [Stanford NLP](https://nlp.stanford.edu/data/glove.6B.zip) and place `glove.6B.300d.txt` at `data/glove/`
- BERT model (~90MB) downloads automatically in Step 7 via HuggingFace
- CNN training takes ~25 minutes on CPU

---

## 📈 Output

The final output `data/results/mahe_final_predictions.csv` contains:

| Column | Description |
|--------|-------------|
| `Title` | Article title |
| `Final_Top5_SDGs` | Top-5 SDGs (e.g. "SDG 3, SDG 7, SDG 13, SDG 5, SDG 11") |
| `Final_Rank1_SDG` | Most relevant SDG |
| `Final_Rank1_Name` | SDG name |
| `Final_Rank1_Confidence` | Probability score (0–1) |
| `TF-IDF_SVM_Top5` | Track A predictions |
| `BERT_SVM_Top5` | BERT predictions |
| `CNN_Top5` | CNN predictions |
| `Combined_Top5` | Track A + B combined |

---

## 🔬 Clustering Summary

Unsupervised clustering on BERT embeddings of 854 MAHE articles:

| Method | k | Notes |
|--------|---|-------|
| K-Means | 16 | Matches SDGs in OSDG training data |
| K-Means | 17 | Matches all UN SDGs |
| Agglomerative | 16 | Ward linkage |
| DBSCAN | Variable | Cosine metric, noise-robust |

Low silhouette scores confirm that SDG topics naturally overlap — empirically validating the top-5 ranking approach over single-label assignment.

---

## 🔭 Future Work

- Fine-tune SciBERT directly on OSDG data for improved contextual classification
- Initialise CNN embedding layer with GloVe weights instead of random initialisation
- Add SDG 17 training examples (missing from current OSDG version)
- Extend to true multi-label classification with sigmoid output
- Integrate SHAP/LIME for prediction explainability
- Deploy as REST API for real-time SDG tagging on the MAHE portal

---

## 📚 References

- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *EMNLP 2014*
- Reimers & Gurevych (2019). Sentence-BERT. *EMNLP 2019*
- Hajikhani & Suominen (2022). Mapping the SDGs in science. *Scientometrics, 127(11)*
- OSDG Community Dataset v2024.04. [Zenodo 11441197](https://zenodo.org/records/11441197)
- United Nations (2015). [The 2030 Agenda for Sustainable Development](https://sdgs.un.org/goals)

---

## 📝 License

This project is submitted as part of the M.Sc. Data Science curriculum at MAHE, Manipal.  
© 2025 Pranam Acharya. All rights reserved.
