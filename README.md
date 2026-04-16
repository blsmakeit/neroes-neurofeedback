# Neroes Adaptive Neurofeedback - Technical Challenge

> *"Given the current physiological state and task context, what should the system do next to improve the target outcome in the next step or time window?"*

---

## Overview

This project builds an adaptive neurofeedback pipeline that processes EEG signals and game-state data to recommend protocol adjustments that maximise `ProtocolValue` - a brain-derived scalar computed from raw EEG via calibration coefficients.

The work spans two users and two phases:

| Phase | Scope | Notebooks | Samples | Paper |
|---|---|---|---|---|
| Phase 1 | User 1 - single session prototype | 01–06 | 3,549 | `paper/short_paper.md` |
| Phase 2 | User 2 - multi-session + cross-user | 07–11 | 50,194 (+3,549) | `paper/short_paper_v2.md` |

Three core questions are answered in Phase 2:

- **RQ-A:** Does more training data fix offline RL collapse? → **No** (structural action imbalance, 94:6 Hold/Raise)
- **RQ-B:** Do prediction models generalise across users? → **Yes** (76.1% combined, 71–73% zero-shot)
- **RQ-C:** Is RL failure fixable offline with sufficient data? → **No** (identification problem; FQI worsened 19× with 16.5× more data)

---

## Repository Structure

```
neroes-neurofeedback/
├── data/
│   ├── raw/                           # Original session data (gitignored)
│   └── processed/                     # Feature matrices + metadata
├── notebooks/
│   │── Phase 1 - User 1 (single session)
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis (User 1)
│   ├── 02_feature_engineering.ipynb   # 246 features, 1 session
│   ├── 03_supervised_prediction.ipynb # LightGBM walk-forward CV
│   ├── 04_nonrl_recommendation.ipynb  # Greedy supervised policy
│   ├── 05_rl_agent.ipynb              # LinUCB + FQI (1 session)
│   ├── 06_evaluation.ipynb            # Phase 1 evaluation
│   │── Phase 2 - User 2 (multi-session)
│   ├── 07_eda_multisession.ipynb      # EDA - 13 sessions, signal QC
│   ├── 08_feature_engineering_multisession.ipynb  # 59 features, 11 sessions
│   ├── 09_prediction_multisession.ipynb           # Walk-forward CV, 10 folds
│   ├── 10_rl_multisession.ipynb                   # RL with 37,972 training rows
│   └── 11_cross_user_analysis.ipynb               # Transfer + combined model
├── src/
│   ├── data_loader.py                 # Raw session reader → validated DataFrame
│   ├── features.py                    # Shared EEG/game column definitions
│   ├── baselines.py                   # Naive reference models
│   └── evaluation.py                  # Metrics + plots
├── outputs/
│   ├── figures/                       # All plots (u2_* prefix = User 2)
│   ├── lgbm_predictor.pkl
│   ├── linucb_agent.pkl
│   └── fqi_agent.pkl
├── paper/
│   ├── short_paper.md                 # Phase 1 paper (User 1)
│   └── short_paper_v2.md              # Phase 2 paper (multi-session + cross-user)
├── README.md
├── AI_TOOLS_NOTE.md
└── requirements.txt
```

---

## Data

### User 1 (Phase 1)

- **Device:** Unicorn headset - 4 active electrodes (F3, F4, C3, C4), 5 EEG bands = 20 signals
- **Sessions:** 1 session, 10 subsessions (0–9), 3,549 rows
- **Subsession 0:** Baseline calibration (resting state, ~7 min)
- **EEG band power:** Informative - used as predictive features
- **Action space:** Hold / Raise / Lower threshold (3 actions)

### User 2 (Phase 2)

- **Device:** Unicorn headset - 8 active electrodes (+Fp1, Fp2, Oz, Pz), 5 bands = 40 signals
- **Sessions:** 13 recorded → 11 after QC (session\_2 and session\_11 excluded, both below 80% GoodSignalQuality)
- **Total rows:** 50,194 across 11 sessions (6–10 subsessions each)
- **EEG band power:** Non-informative - all 100 spectral columns dropped after zero-variance check
- **Calibration regimes:** 6 unique (TC, TrC) pairs across sessions - one-hot encoded
- **Action space:** Hold / Raise only (2 actions), 94:6 imbalance

### Target

`ProtocolValue = TangentCoefficient × (raw_signal − Baseline) + TranslationCoefficient`

A near-random-walk signal at 1-second resolution (mean ≈ 0, std ≈ 0.38). User 1 shows a monotonic learning trajectory (+0.247 across 9 subsessions); User 2 shows stable oscillation (R² = 0.107, p = 0.276 - maintaining rather than improving).

---

## Results Summary

### Phase 1 - User 1

| Method | MAE | DirAcc | Notes |
|---|---|---|---|
| LastValue baseline | 0.479 | 0% | - |
| SessionMean baseline | 0.350 | - | - |
| **LightGBM (walk-forward CV)** | **0.351** | **67.6%** | 246 features, 1 session |
| FQI Q-learning (IPS) | - | - | reward +0.002 (near-random) |

### Phase 2 - User 2 (multi-session)

| Method | MAE | DirAcc | Notes |
|---|---|---|---|
| Persistence baseline | 0.496 | 0% | - |
| Mean baseline | 0.409 | 71.3% | - |
| **LightGBM (walk-forward CV)** | **0.366** | **72.5%** | 59 features, 10 folds |
| LinUCB (IPS) | - | IPS −0.003 | 99.8% Raise collapse |
| FQI (IPS) | - | IPS −0.038 | Worse than random |

### Phase 2 - Cross-user generalisation

| Experiment | Features | MAE | DirAcc |
|---|---|---|---|
| U1 → U2 transfer | 12 shared | 0.379 | 71.4% |
| U2 → U1 transfer | 12 shared | 0.346 | 73.0% |
| **Combined → U1** | **13 (shared + user\_id)** | **0.329** | **76.1%** |

The combined 13-feature population model achieves the best result in the entire pipeline - 8.5 pp above the User 1 personalised model with 246 features.

---

## Key Findings

1. **Session volume beats feature dimensionality.** 72.5% DirAcc with 59 features across 13 sessions outperforms 67.6% with 246 features in 1 session.

2. **AR structure is universal.** PV lag features (lag-1/2/5) and rolling statistics dominate feature importance in both users, regardless of EEG band availability. Any adaptive neurofeedback predictor should include these as a non-negotiable baseline.

3. **Offline RL under 94:6 action imbalance is structurally broken - and more data makes it worse.** FQI IPS degraded from −0.002 (n=3,549) to −0.038 (n=37,972). Q(s, Raise) is not identifiable when only 6% of logged transitions involve Raise. The fix requires online exploration (ε-greedy or Thompson Sampling LinUCB during live sessions).

4. **Population model beats personalised model.** A combined model trained on both users with 13 universal features achieves 76.1% DirAcc - outperforming user-specific models with up to 19× more features.

---

## Reinforcement Learning Framing

| Element | Definition |
|---|---|
| **State s** | 59-dim vector (User 2): z-normalised PV lags/rolling + game state + session context + calibration group |
| **Action a** | 2 discrete: Hold (0), Raise threshold (1) |
| **Reward r** | Δ`ProtocolValue`(t→t+1), clipped to [−0.5, +0.5] |
| **Policy π** | LinUCB (contextual bandit) or FQI (offline Q-learning) |
| **Horizon** | Per-step (1s), actionable at subsession transitions (~3 min) |

**Why offline RL fails here:** with p(Raise) = 0.06, the Bellman bootstrap target for Q(s, Raise) is dominated by Q(s', Hold) in 94% of next-state transitions. Each FQI iteration amplifies this bias into the Raise branch. This is a formal identification problem, not a sample size problem.

---

## Deployment Roadmap

| Phase | Trigger | Action | Expected DirAcc |
|---|---|---|---|
| 1. Immediate | Day 1 | Deploy population AR predictor (13 features, zero-shot) | ~71% |
| 2. Early | Session 3+ | Begin online LinUCB with ε-greedy exploration | 72–74% |
| 3. Growth | 5+ sessions logged | Add user embedding, fine-tune combined model | 74–76% |
| 4. Mature | 10+ sessions, balanced actions | Full offline RL viable | 76%+ |

---

## Setup

```bash
git clone https://github.com/<your-username>/neroes-neurofeedback.git
cd neroes-neurofeedback

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Place session data:
cp -r /path/to/NeroesSession_Data data/raw/

# Run Phase 1 (User 1): notebooks 01 → 06
# Run Phase 2 (User 2): notebooks 07 → 11
jupyter notebook
```

---

## Limitations

**Data:**
- Two users is not enough to validate population model generalisation
- User 2 EEG band power is non-informative - the universal feature set is entirely behavioural/temporal
- Session-level action imbalance (94:6) is a property of the logging policy and cannot be corrected post-hoc

**Prediction:**
- R² ≈ 0 for all methods - `ProtocolValue` is a near-random walk at 1-second resolution
- Directional accuracy (67–76%) is the practically meaningful metric
- Session-9 is a persistent outlier (mean PV = −0.314 vs population −0.014); its inclusion degrades the walk-forward CV mean

**RL:**
- Offline RL is definitively not viable under the current logging policy - both users confirm this
- LinUCB collapse to 99.8% Raise is reproducible and structural, not a hyperparameter issue
- IPS estimates are noisy - treat as relative rankings, not absolute reward values

---

## Compute

Training and experiments run locally (Python 3.11, WSL2). The architecture is designed to scale to **Deucalion** (INCD National HPC, University of Minho) for multi-session training, DQN/PPO experiments, and hyperparameter search.

---

*Neroes Technical Challenge Submission - Bruno Sousa, April 2026*
