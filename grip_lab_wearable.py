# =============================================================
# Grip Lab AI — Wearable Recovery Algorithm Enhancement
# Author: Omkar Pallerla | MS Business Analytics, ASU
# XGBoost: Correcting Oura/Whoop scores via Grip Strength CNS proxy
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
np.random.seed(42)
COLORS = ['#4f9cf9','#06d6a0','#7c3aed','#f59e0b','#ef4444']

# ══════════════════════════════════════════════════════════════
# SIMULATE REALISTIC ATHLETE DATA (14 weeks = 98 days)
# ══════════════════════════════════════════════════════════════
n_days    = 98
athlete_ids = [f'ATH_{i:03d}' for i in range(1, 21)]  # 20 athletes

records = []
for athlete in athlete_ids:
    # Baseline grip strength varies by athlete
    baseline_grip = np.random.normal(45, 8)
    baseline_hrv  = np.random.normal(55, 12)

    for day in range(n_days):
        # Simulate fatigue cycles (peaks every ~7 days = training block)
        fatigue_cycle  = np.sin(2 * np.pi * day / 7) * 0.3
        trend          = -0.002 * day  # slight cumulative fatigue over season

        hrv            = baseline_hrv + np.random.normal(0, 4) + fatigue_cycle * 8
        sleep_score    = 75 + np.random.normal(0, 10) + fatigue_cycle * 5
        resting_hr     = 52 + np.random.normal(0, 3) - fatigue_cycle * 3
        left_grip      = baseline_grip + np.random.normal(0, 3) + trend * 20 + fatigue_cycle * -4
        right_grip     = left_grip + np.random.normal(0, 1.5)   # asymmetry
        grip_30d_avg   = baseline_grip + trend * 10              # rolling average

        # Neural asymmetry
        neural_asymmetry = abs(left_grip - right_grip) / max(left_grip, right_grip)
        grip_deficit     = (grip_30d_avg - left_grip) / max(grip_30d_avg, 1)
        grip_decay       = max(0, grip_deficit)  # how much below baseline

        # Oura-style base readiness score (HRV-based only)
        hrv_score        = np.clip(hrv / baseline_hrv * 70 + sleep_score * 0.3, 0, 100)
        base_readiness   = np.clip(hrv_score + np.random.normal(0, 3), 0, 100)

        # TRUE readiness (includes CNS state — what we want to predict)
        cns_penalty      = (grip_decay * 25 + neural_asymmetry * 20)
        true_readiness   = np.clip(base_readiness - cns_penalty, 0, 100)

        # Athlete's perceived exertion (RPE) — ground truth
        rpe              = np.clip(10 - true_readiness / 12 + np.random.normal(0, 0.5), 1, 10)

        records.append({
            'athlete_id':       athlete,
            'day':              day,
            'hrv':              hrv,
            'sleep_score':      sleep_score,
            'resting_hr':       resting_hr,
            'left_grip':        left_grip,
            'right_grip':       right_grip,
            'grip_30d_avg':     grip_30d_avg,
            'neural_asymmetry': neural_asymmetry,
            'grip_deficit':     grip_deficit,
            'grip_decay':       grip_decay,
            'base_readiness':   base_readiness,
            'true_readiness':   true_readiness,
            'rpe':              rpe,
        })

df = pd.DataFrame(records)
print(f"Dataset: {len(df)} sessions across {len(athlete_ids)} athletes")

# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
df['hrv_ratio']        = df['hrv'] / df.groupby('athlete_id')['hrv'].transform('mean')
df['sleep_deficit']    = 80 - df['sleep_score']
df['grip_asymm_flag']  = (df['neural_asymmetry'] > 0.10).astype(int)
df['high_fatigue_day'] = (df['grip_deficit'] > 0.15).astype(int)

# ══════════════════════════════════════════════════════════════
# TRAIN XGBOOST ADJUSTMENT MODEL
# ══════════════════════════════════════════════════════════════
feature_cols = ['hrv','sleep_score','resting_hr','left_grip','right_grip',
                'grip_30d_avg','neural_asymmetry','grip_deficit','grip_decay',
                'hrv_ratio','sleep_deficit','grip_asymm_flag']

X = df[feature_cols]
y = df['true_readiness']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

xgb = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.04,
                    subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb.fit(X_train_s, y_train)

y_pred = xgb.predict(X_test_s)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"\nXGBoost Adjustment Model: MAE={mae:.2f} pts, R²={r2:.3f}")

# ── HOW OFTEN DOES BASE READINESS MISLEAD? ──────────────────
df['predicted_readiness'] = xgb.predict(scaler.transform(X))
df['hidden_fatigue'] = (
    (df['base_readiness'] >= 75) &
    (df['predicted_readiness'] < 65)
).astype(int)

hidden_pct = df['hidden_fatigue'].mean() * 100
print(f"\nHidden fatigue detected: {hidden_pct:.1f}% of sessions")
print(f"(HRV says Green, CNS says Caution)")

# ── LOAD RECOMMENDATIONS ─────────────────────────────────────
def recommend(score):
    if score >= 75:   return 'High Intensity 🏋️'
    elif score >= 55: return 'Moderate Training 🏃'
    elif score >= 40: return 'Light Training 🚶'
    else:             return 'Active Recovery 🧘'

df['load_recommendation'] = df['predicted_readiness'].apply(recommend)

# Export
df[['athlete_id','day','base_readiness','predicted_readiness',
    'neural_asymmetry','grip_deficit','hidden_fatigue',
    'load_recommendation','rpe']].to_csv('outputs/readiness_scores.csv', index=False)
print("Exported: outputs/readiness_scores.csv")

# ── VISUALIZATIONS ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('#0d1117')

# Score comparison over time (one athlete)
ax = axes[0, 0]
ath = df[df['athlete_id'] == 'ATH_001'].head(42)
ax.plot(ath['day'], ath['base_readiness'],   color='#7a8499', lw=1.5, linestyle='--', label='HRV-Only Score')
ax.plot(ath['day'], ath['predicted_readiness'], color='#4f9cf9', lw=2, label='Grip-Corrected Score')
ax.fill_between(ath['day'], ath['base_readiness'], ath['predicted_readiness'],
                where=ath['base_readiness'] > ath['predicted_readiness'],
                alpha=0.2, color='#ef4444', label='Hidden Fatigue Zone')
ax.axhline(75, color='#06d6a0', linestyle=':', alpha=0.5, lw=1)
ax.set_xlabel('Training Day'); ax.set_ylabel('Readiness Score')
ax.set_title('Readiness Score — HRV-Only vs Grip-Corrected', color='white', pad=12)
ax.legend(fontsize=8)

# Neural asymmetry over time
ax = axes[0, 1]
ath_asym = df[df['athlete_id'] == 'ATH_001'].head(42)
ax.plot(ath_asym['day'], ath_asym['neural_asymmetry'], color='#7c3aed', lw=2)
ax.axhline(0.10, color='#ef4444', linestyle='--', lw=1.5, label='10% Warning Threshold')
ax.fill_between(ath_asym['day'], 0.10, ath_asym['neural_asymmetry'],
                where=ath_asym['neural_asymmetry'] > 0.10,
                alpha=0.3, color='#ef4444')
ax.set_xlabel('Training Day'); ax.set_ylabel('Neural Asymmetry (L/R ratio)')
ax.set_title('Neural Asymmetry — CNS Early Warning Signal', color='white', pad=12)
ax.legend(fontsize=8)

# Feature importance
ax = axes[1, 0]
feat_imp = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
bars = ax.barh(feat_imp.index[::-1], feat_imp.values[::-1],
               color=['#06d6a0' if 'grip' in f else '#4f9cf9' for f in feat_imp.index[::-1]])
ax.set_title('Feature Importance — XGBoost Adjustment Model', color='white', pad=12)
ax.set_xlabel('Importance Score')

# Hidden fatigue by athlete
ax = axes[1, 1]
hidden_by_ath = df.groupby('athlete_id')['hidden_fatigue'].mean().sort_values(ascending=False)
colors_ath = ['#ef4444' if v > 0.15 else '#4f9cf9' for v in hidden_by_ath.values]
ax.bar(range(len(hidden_by_ath)), hidden_by_ath.values * 100, color=colors_ath)
ax.axhline(15, color='#f59e0b', linestyle='--', alpha=0.6, label='15% alert threshold')
ax.set_xticks(range(len(hidden_by_ath)))
ax.set_xticklabels([a.replace('ATH_','A') for a in hidden_by_ath.index], fontsize=7)
ax.set_ylabel('Hidden Fatigue Rate %')
ax.set_title('Hidden Fatigue Rate by Athlete', color='white', pad=12)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('outputs/grip_lab_analysis.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Saved: outputs/grip_lab_analysis.png")
plt.show()
