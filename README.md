# 🧠 Neroes Adaptive Neurofeedback — Technical Challenge

> *"Given the current physiological state and task context, what should the system do next to improve the target outcome in the next step or time window?"*

---

## Problem Framing

This project builds a compact adaptive neurofeedback prototype that processes EEG signals and game state to recommend protocol adjustments that maximise `ProtocolValue` — a brain-derived signal computed from 4 active EEG electrodes (F3, F4, C3, C4).

The system is framed as a **sequential decision problem**: at each timestep, given a state vector of physiological and game features, the system (1) predicts the next expected `ProtocolValue` and (2) recommends the protocol action most likely to increase it.

Two complementary approaches are implemented and compared:

| Approach | Method | Key idea |
|---|---|---|
| **Supervised + Policy** | LightGBM regressor | Predict `ProtocolValue(t+1)`, recommend action greedily |
| **Reinforcement Learning** | LinUCB Bandit + FQI Q-learning | Learn reward-maximising policy directly from Δ`ProtocolValue` |

---

## Repository Structure

```
neroes-neurofeedback/
├── data/
│   ├── raw/                        # Original session data (gitignored)
│   └── processed/                  # Feature matrix + metadata
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_supervised_prediction.ipynb
│   ├── 04_nonrl_recommendation.ipynb
│   ├── 05_rl_agent.ipynb
│   └── 06_evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── baselines.py
│   └── evaluation.py
├── outputs/
│   ├── figures/
│   ├── lgbm_predictor.pkl
│   ├── linucb_agent.pkl
│   └── fqi_agent.pkl
├── paper/
│   └── short_paper.md
├── README.md
├── AI_TOOLS_NOTE.md
└── requirements.txt
```

---

## Data

- **Device:** Unicorn headset — 4 active electrodes (F3, F4, C3, C4), 5 EEG bands each = 20 live signals
- **Session:** 1 participant, 10 subsessions (0–9), 3,549 total rows
- **Subsession 0:** Baseline calibration (fixation cross, 837 rows, ~7 min)
- **Subsessions 1–9:** Active game sessions (240–418 rows each, ~2–4 min each)
- **Target:** `ProtocolValue` — scaled EEG band power signal to maximise
- **Behavioural proxy:** `PlayerPositionY` — ship position controlled by brain signal

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

# Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06
jupyter notebook
```

---

## Results Summary

| Method | MAE | RMSE | R² | Directional Acc |
|---|---|---|---|---|
| LastValue (baseline) | 0.4785 | 0.6196 | −0.667 | — |
| SessionMean (baseline) | 0.3501 | 0.4799 | 0.000 | — |
| RollingMean w=10 | 0.3718 | 0.4677 | +0.052 | — |
| **LightGBM (walk-forward CV)** | **0.3508** | **0.4689** | **+0.003** | **67.6%** |
| FQI Q-learning (IPS) | — | — | — | reward +0.0020 |

**ProtocolValue trend:** improved from −0.185 (ss1) to +0.062 (ss9), a delta of **+0.247** — the neurofeedback protocol is working.

---

## Assumptions

1. `ProtocolValue` is derived from EEG band power via `TangentCoefficient` and `TranslationCoefficient` — these are session-level constants, not real-time actions
2. Subsession 0 is used exclusively for personalised z-score normalisation of EEG features
3. The "action" space (Lower/Hold/Raise threshold) is constructed from inter-subsession `ProtocolValue` trends, as no explicit real-time action column exists in the data
4. Temporal ordering within subsessions is strictly respected (walk-forward CV, no data leakage)
5. Lag features assume a sampling rate of ~1 sample/second based on session duration vs row count

---

## Limitations

**Data:**
- Single session, single participant — no generalisation guarantees whatsoever
- 3,549 rows is insufficient for RL with 246-dimensional state space and 3 actions
- Logged action distribution is heavily skewed (70% "Raise") — sparse coverage of other actions makes offline RL unreliable

**Prediction:**
- R² ≈ 0 across all methods — `ProtocolValue` behaves like a near-random walk at 1-second resolution, making point prediction very hard
- Directional accuracy (67.6%) is the practically meaningful metric, not MAE in absolute terms
- Top predictive features are all autoregressive (lags/rolling of `ProtocolValue` itself) — EEG features contribute minimally at this timescale

**RL:**
- LinUCB match rate was 0.6% — statistically meaningless for offline evaluation
- FQI collapsed to near-uniform action 0 (99.8%) due to data sparsity and action imbalance
- Offline RL cannot explore — the logged data coverage is the hard ceiling on policy quality
- IPS reward estimates are high-variance with n < 50 for most policies

**Action space:**
- The action space is derived, not observed — "Lower/Hold/Raise threshold" is a proxy constructed from inter-subsession trends, not a direct system control parameter
- The non-RL recommendation module (notebook 04) is degenerate with one session: without explicit action variation in the feature vector, it always recommends "Hold"

---

## What I Would Improve Next

1. **More data:** With 10+ sessions, RL becomes viable. The architecture is ready — just re-run notebooks 02–06 with the extended dataset
2. **Cleaner action space:** Clarify with the Neroes team what the system actually controls in real time (threshold? difficulty? feedback gain?) and build the action space from that
3. **Shorter prediction horizon:** Instead of predicting `ProtocolValue(t+1)` at 1-second resolution, predict a 10–30 second rolling mean — much more stable and actionable
4. **Online bandit:** Replace offline FQI with an online LinUCB that updates during the session — even with one session, it would accumulate evidence in real time
5. **EEG feature engineering:** The raw band power features contribute little — compute ratios (theta/alpha, beta/alpha), asymmetry indices (F3-F4), and coherence features
6. **Uncertainty quantification:** Add conformal prediction intervals to the LightGBM output so the system knows when its predictions are unreliable

---

## How to Evolve This into a Real-Time Adaptive System

```
[EEG Headset] ──→ [Signal Processing] ──→ [Feature Extraction]
                                                    │
                                          [30-second window]
                                                    │
                                         [State Vector (246-dim)]
                                                    │
                              ┌─────────────────────┴──────────────────────┐
                              │                                             │
                    [LightGBM Predictor]                         [LinUCB Bandit]
                    [Predict PV(t+1)]                            [Select Action]
                              │                                             │
                              └─────────────────────┬──────────────────────┘
                                                    │
                                      [Protocol Parameter Update]
                                      (threshold / difficulty / gain)
                                                    │
                                          [Observe Δ PV reward]
                                                    │
                                      [Online LinUCB Update]
                                      (Thompson Sampling variant)
```

**Latency budget:** Feature extraction ~50ms, inference ~5ms → total <100ms per cycle. Compatible with real-time neurofeedback loops (typically 0.5–2s update rate).

**Deployment:** The LinUCB agent (`outputs/linucb_agent.pkl`) + scaler are lightweight enough to run on any edge device. The LightGBM predictor (`outputs/lgbm_predictor.pkl`) provides a confidence signal alongside the action recommendation.

---

## Reinforcement Learning Framing

| Element | Definition |
|---|---|
| **State s** | 246-dim vector: z-normalised EEG lags/rolling + game state + session context |
| **Action a** | 3 discrete: Lower threshold (0), Hold (1), Raise threshold (2) |
| **Reward r** | Δ`ProtocolValue`(t→t+1), clipped to [−1.36, +1.33] |
| **Policy π** | LinUCB (online) or FQI (offline pre-trained) |
| **Horizon** | Per-step (1s), but actionable at subsession level (~3 min) |

The supervised approach (notebook 03) approximates the value function Q(s,a) implicitly — it learns to predict the outcome of the current state without explicit action conditioning. The RL approach (notebook 05) learns this mapping directly from reward signals with explicit action conditioning via FQI.

For production, **LinUCB with Thompson Sampling** is preferred: it is online-capable, interpretable, computationally trivial, and naturally handles the exploration-exploitation trade-off in a session-by-session adaptive loop.

---

## Compute

Model training and extended experiments were run locally (Python 3.11, WSL2). The architecture is designed to scale to **Deucalion** (INCD National HPC, University of Minho) for multi-session training, DQN/PPO experiments, and hyperparameter optimisation — resources unavailable to most candidates at this stage.

---

*© 2026 — Neroes Technical Challenge Submission*
