# ⌚ Grip Lab AI — Wearable Recovery Algorithm Enhancement

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

> **XGBoost algorithm correcting Oura/Whoop recovery scores using Grip Strength as a CNS fatigue proxy — detecting hidden fatigue that HRV-only systems miss in 86% of weightlifting sessions.**

---

## The Problem With Current Wearables

Oura and Whoop measure recovery using HRV and Resting Heart Rate — purely physiological signals. But athletes know: you can have a green HRV score and a completely fried Central Nervous System (CNS).

This creates the **Hidden Fatigue Problem**: athletes train heavy on what looks like a good day, and get injured or overtrain.

This project validates **Isometric Grip Strength** as a scalable CNS readiness proxy and builds an algorithm to *correct* wearable scores before athletes act on them.

---

## The Algorithm — Hybrid Scoring Model

### Data Inputs

| Source | Data | Frequency |
|--------|------|-----------|
| Oura Ring API | Sleep score, HRV, RHR | Nightly |
| Squegg Dynamometer | Grip force (left + right), duration | Morning |
| Derived | Neural Asymmetry = abs(Left - Right) / Max | Per session |

### Scoring Logic

```python
# Base score from Oura
base_score = oura_readiness_score

# Grip deficit vs 30-day rolling average
grip_deficit = (rolling_30d_avg - today_grip) / rolling_30d_avg

# Neural asymmetry: left vs right hand divergence
asymmetry = abs(left_grip - right_grip) / max(left_grip, right_grip)

# XGBoost adjustment layer
adjustment = xgb_model.predict([grip_deficit, asymmetry, base_score, ...])

# Final corrected score + recommendation
final_score = base_score + adjustment
load_rec = "High Intensity" if final_score > 75 else "Active Recovery"
```

### The Neural Asymmetry Insight
A left/right grip divergence **> 10%** consistently precedes physical fatigue symptoms by **12-24 hours** — an early warning signal HRV completely misses.

---

## Key Results

| Metric | Value |
|--------|-------|
| Hidden fatigue caught | Algorithm correctly downgraded scores in **86%** of weightlifting sessions |
| Best predictor | Grip Force Decay > Sleep Duration for daily readiness |
| Asymmetry early warning | > 10% divergence flags CNS fatigue 12-24h before symptoms |
| Model alignment | XGBoost adjustment aligns with athlete RPE in **89%** of sessions |

---

## Why This Stands Out for Analytics Roles

| Skill Demonstrated | How |
|--------------------|-----|
| Time-series feature engineering | Rolling averages, decay rates, asymmetry deltas |
| API data ingestion | Oura Ring API + Bluetooth dynamometer data |
| Interactive dashboard | Plotly — athletes see score history and adjustment breakdown |
| Explainable ML | Not just a score — shows *why* it was adjusted |
| BI-first thinking | Turns raw biometric data into actionable training decisions |

---

## Tech Stack

`Python` `XGBoost` `Pandas` `NumPy` `Scikit-Learn` `Plotly` `Matplotlib` `Oura Ring API`

---

## Project Structure

```
Grip-Lab-Wearable-AI/
├── data/
│   └── sample_sessions.csv
├── notebooks/
│   ├── 01_EDA_Grip_HRV_Correlation.ipynb
│   ├── 02_XGBoost_Adjustment_Model.ipynb
│   └── 03_Interactive_Dashboard.ipynb
├── outputs/
│   ├── readiness_dashboard.html
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

---

## Run It

```bash
git clone https://github.com/omkarpallerla/Grip-Lab-Wearable-AI.git
cd Grip-Lab-Wearable-AI
pip install -r requirements.txt
jupyter notebook notebooks/01_EDA_Grip_HRV_Correlation.ipynb
```

---

<div align="center"><sub>Omkar Pallerla · MS Business Analytics ASU · BI Engineer · Azure | GCP | Databricks Certified</sub></div>
