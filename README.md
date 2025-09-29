# CB_Sentiment_Forecasting
This project studies whether the language in Federal Reserve and European Central Bank speeches carries predictive signals for interest-rate changes beyond traditional macro indicators. We fine-tune RoBERTa classifiers and run aspect-based sentiment analysis (ABSA), then inject the best NLP signal into a SARIMAX model alongside macro variables.

**Project Title:** Enhancing Monetary Policy Forecasting Through Sentiment Analysis of Central Bank Communications

## Project Overview

This project studies whether the language in **Federal Reserve** and **European Central Bank** speeches carries predictive signals for **interest-rate changes** beyond traditional macro indicators.  

We fine-tune **RoBERTa** classifiers and run **aspect-based sentiment analysis (ABSA)**, then inject the best NLP signal into a **SARIMAX** model alongside macro variables.

> **Headline result:** Adding RoBERTa-based sentiment to SARIMAX reduced **ECB RMSE by 32.8% (8.53 → 5.64)**, while gains for the Fed were modest.

## Repository structure
.
├── README.md
├── requirements.txt
├── .gitignore
├── Notebooks/
│   ├── webscraper.ipynb
│   ├── data_merge.ipynb
│   ├── 3_EDA.ipynb
│   ├── TEMPORAL_TRIALS.ipynb
│   ├── BERT.ipynb
│   ├── CLASSIC_ML.ipynb
│   ├── ABSA.ipynb
│   ├── 8_ABSA_2.ipynb
│   ├── new_fed_sarimax.ipynb
│   ├── new_ecb_sarimax.ipynb
│   ├── Link to models.txt
│   └── data/
│       ├── all_ECB_speeches.csv
│       ├── cleaned_sentiment_data.csv
│       ├── ECB_data.csv
│       ├── federal_reserve_speeches.csv
│       ├── FEDFUNDS.csv
│       ├── temporal_aware_classifica…a.csv
│       ├── final_v2.csv
│       ├── 5_way_class.csv
│       ├── consumer_confidence_cl…d.csv
│       ├── fed_ecb_weekly_similarity.csv
│       ├── fed_ecb_similarity.csv
│       ├── unemployment_clean.csv
│       ├── gdp_clean.csv
│       ├── infl_exp_clean.csv
│       ├── inflation_cleaned.csv
│       ├── oil_clean.csv
│       ├── temporal_aware_classific…K.csv
│       ├── yield_clean.csv
│       ├── train_df1.csv
│       ├── enriched_speech_data.csv
│       ├── enhanced_sentiment_scores.csv
│       └── Loughran-McDonald_dictionary.csv
└── Report/
    └── project_report_10.pdf

## How to reproduce

### A) Environment
create env
`python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt`

### B) Typical workflow

1. **Scrape / ingest**  
   Run: `Notebooks/webscraper.ipynb`  
   (or skip if CSVs are already present)

2. **Merge & clean**  
   Run: `Notebooks/data_merge.ipynb`  
   → builds unified speech & rate datasets

3. **EDA**  
   Run: `Notebooks/3_EDA.ipynb`  
   (explore speech volume, timing vs decisions, etc.)

4. **Text classification**
   - **Baselines**: `CLASSIC_ML.ipynb` (TF-IDF + Logistic Regression)  
   - **Transformers**: `BERT.ipynb` (RoBERTa fine-tuning; binary and 5-way tasks)  
   - **Temporal experiments**: `TEMPORAL_TRIALS.ipynb` (random vs time-aware splits)

5. **ABSA pipeline**  
   Run: `ABSA.ipynb` and `8_ABSA_2.ipynb`  
   (LDA topics + FinBERT sentiment)

6. **Time-series integration**
   - **Fed**: `new_fed_sarimax.ipynb`  
   - **ECB**: `new_ecb_sarimax.ipynb`  
   Add the best classifier’s output as an exogenous feature in SARIMAX.

### C) Expected results (sanity checks)

- RoBERTa beats TF-IDF on random splits;  
  **both drop** on time-aware splits due to concept drift.

- **ECB SARIMAX + sentiment** improves RMSE by ~**32.8%**;  
  Fed improvements are smaller.

## Data sources 

Speech corpora (ECB & Fed) 1999–2025; interest-rate series and macro variables (inflation, unemployment, GDP, yield curve, consumer sentiment, 5-yr inflation expectations, oil). See the Report for exact links and series IDs.

## Method details

- Classifier configs: RoBERTa fine-tuned (max length 128, lr 1e-5 with decay/halve on plateau, dropout 0.2, weight decay 0.05, early stopping). Baseline uses character n-grams TF-IDF with feature selection. 
- ABSA: LDA topics (Inflation; Financial Stability; Micro/Business; Post-crisis reforms), sentiment via FinBERT; best F1 ≈ 0.554 using Random Forest. 
- SARIMAX: stationarity checks/differencing, exogenous macro features, then add speech-sentiment feature from the best classifier. 

## Limitations & future work

- Temporal stability: time-aware generalization is hard; periodic re-training or continual learning recommended. 
- Communication style differences limit transfer across banks; consider broader, cross-bank datasets and dynamic topic models (e.g., BERTopic)

## Acknowledgements

We acknowledge the ECB, Federal Reserve, and FRED/Eurostat data providers; and open-source NLP libraries used in this work.

## Authors
- Gaia Iori
- Michael Ladaa
- Luca Milani
- Matteo Roda
- Sofia Villa

## Contacts
For any inquiries, please contact:
- luca.milani2@studbocconi.it


