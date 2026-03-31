# ⌚ Grip Lab AI — Wearable Recovery Algorithm (XGBoost + Oura/Whoop)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

> **Proprietary XGBoost algorithm correcting Oura/Whoop recovery scores by integrating Grip Strength as a CNS fatigue proxy — detecting hidden fatigue that HRV-only systems miss in 86% of weightlifting sessions.**

---

## The Problem With Current Wearables

Oura and Whoop measure recovery using HRV and Resting Heart Rate — purely physiological signals. But elite athletes know: you can have a green HRV score and a completely fried Central Nervous System (CNS).

This creates the **Hidden Fatigue Problem**: athletes train heavy on what looks like a good day, and get injured. This project validates Isometric Grip Strength as a scalable CNS readiness proxy and builds an algorithm to *correct* standard wearable scores before athletes see them.

---

## Data Inputs

| Source | Data | Frequency |
|--------|------|----------|
| Oura Ring API | Sleep score, HRV, Resting HR | Nightly |
| Squegg Dynamometer | Grip force (left + right), duration | Morning |
| Derived | Neural Asymmetry = abs(Left-Right) / Max | Per session |

---

## The Algorithm

```python
# Base score from Oura
base_score = oura_readiness_score

# Grip strength deficit vs 30-day rolling average
grip_deficit = (rolling_30d_avg - today_grip) / rolling_30d_avg

# Left vs right asymmetry (CNS fatigue signal)
asymmetry = abs(left_grip - right_grip) / max(left_grip, right_grip)

# XGBoost adjustment layer
adjustment = xgb_model.predict([grip_deficit, asymmetry, base_score, sleep_score, hrv])

# Final corrected score -> load recommendation
final_score = base_score + adjustment
load_recommendation = "High Intensity" if final_score > 75 else "Active Recovery"
```

**Key Insight — Neural Asymmetry:** A left/right grip divergence >10% consistently precedes physical fatigue symptoms by 12-24 hours — an early warning signal HRV completely misses.

---

## Key Results

| Finding | Detail |
|---------|--------|
| Hidden fatigue detection | Algorithm correctly downgraded "High Readiness" in **86% of weightlifting sessions** |
| Best predictor | Grip Force Decay > Sleep Duration for daily readiness |
| Asymmetry signal | >10% divergence = CNS flag, **12-24 hours early** vs. physical symptoms |
| Model accuracy | XGBoost aligns with athlete RPE (Rate of Perceived Exertion) in **89% of sessions** |

---

## Why This Is Relevant for Analytics/BI Roles

| Skill Demonstrated | How |
|-------------------|-----|
| Time-series feature engineering | Rolling averages, decay rates, asymmetry deltas |
| API data ingestion | Oura Ring API, Bluetooth dynamometer |
| Wearable IoT pipeline | Sensor data -> model -> recommendation output |
| Explainable output | Not just a score — shows *why* it was adjusted |
| Interactive dashboard | Plotly time-series — athletes see score history + breakdown |

---

## Tech Stack

`Python` `XGBoost` `Pandas` `NumPy` `Scikit-Learn` `Plotly` `Matplotlib` `Oura Ring API` `Squegg Bluetooth Dynamometer`

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
