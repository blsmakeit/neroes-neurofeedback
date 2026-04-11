# 🧠 Neroes Adaptive Neurofeedback — Technical Challenge

> *"Given the current physiological state and task context, what should the system do next to improve the target outcome in the next step or time window?"*

---

## Problem Framing

This project builds a compact adaptive neurofeedback prototype that learns to recommend system actions based on a user's real-time physiological state, with the goal of maximising `ProtocolValue` — a brain-derived signal — across neurofeedback game sessions.

The system is framed as a **sequential decision problem**: at each timestep, given a state vector of physiological features, the system predicts the next expected `ProtocolValue` and recommends the action most likely to increase it.

Two complementary approaches are implemented and compared:

| Approach | Description |
|---|---|
| **Supervised + Policy** | LightGBM regressor predicts `ProtocolValue(t+1)`; action chosen by maximising predicted value over candidate actions |
| **Reinforcement Learning** | Contextual Bandit / Q-learning agent trained offline on session data; reward = Δ`ProtocolValue` |

---

## Repository Structure

```
neroes-neurofeedback/
│
├── data/
│   ├── raw/                        # Original NeroesSession_Data (gitignored)
│   └── processed/                  # Cleaned & feature-engineered parquet files
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_features.ipynb           # Feature engineering
│   ├── 03_prediction.ipynb         # Supervised prediction module
│   ├── 04_recommendation.ipynb     # Non-RL recommendation module
│   ├── 05_rl_agent.ipynb           # RL agent (Contextual Bandit / DQN)
│   └── 06_evaluation.ipynb         # Comparison vs. baselines
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Load + validate session data
│   ├── features.py                 # Feature extraction pipeline
│   ├── predictor.py                # Prediction module (LightGBM)
│   ├── recommender.py              # Non-RL recommendation logic
│   ├── rl_agent.py                 # RL agent implementation
│   ├── baselines.py                # Naive baselines
│   └── evaluation.py              # Metrics and evaluation utilities
│
├── outputs/
│   └── figures/                    # All saved plots
│
├── paper/
│   └── short_paper.md              # Short scientific paper
│
├── scripts/
│   └── run_pipeline.py             # End-to-end pipeline script
│
├── README.md                       # This file
├── AI_TOOLS_NOTE.md                # AI usage transparency note
└── requirements.txt
```

---

## Data

- **Session structure:** 1 session with 10 subsessions (0–9)
- **Subsession 0:** Baseline calibration (user watching a fixation cross)
- **Subsessions 1–9:** Active game sessions (user playing to increase `PlayerPositionY`)
- **Target variable:** `ProtocolValue` — brain-derived signal to maximise
- **Proxy behaviour:** `PlayerPositionY` — game metric reflecting neurofeedback performance

---

## Setup

```bash
git clone https://github.com/<your-username>/neroes-neurofeedback.git
cd neroes-neurofeedback

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Place the session data:
# cp -r /path/to/NeroesSession_Data data/raw/

jupyter notebook
```

---

## Assumptions

1. `ProtocolValue` is a continuous real-valued signal derived from EEG band power
2. Subsession 0 (baseline) provides a personalised reference; features are z-scored relative to it
3. The "action" space is defined by controllable system parameters identified during EDA
4. Samples within a subsession are temporally ordered and correlated — lag features are valid
5. Cross-subsession ordering matters: subsession index is used as a contextual feature

---

## Limitations

- Single session / single participant — no generalisation guarantees
- Offline RL training on historical data cannot account for non-stationarity in live settings
- Action space may be implicit (if no explicit control column exists, it is derived from binned states)
- No ground truth for "optimal action" — reward is entirely self-supervised from signal delta

---

## What I Would Improve Next

- Multi-session, multi-participant training for personalisation vs. generalisation trade-off
- Online learning with real-time data stream (sliding window inference)
- Proper RL with environment simulation using the Deucalion supercomputer
- Uncertainty quantification on predictions (conformal prediction or Bayesian approach)
- Integration with real-time signal pipeline

---

## How to Evolve This into a Real-Time Adaptive System

```
[EEG/Bio Sensors] → [Signal Preprocessing] → [Feature Extraction (10–30s window)]
        ↓
[State Vector] → [Trained Predictor] → [Predicted ProtocolValue(t+1)]
        ↓
[RL Agent / Policy] → [Recommended Action] → [Game/Protocol Parameter Update]
        ↓                                              ↓
[Reward = Δ ProtocolValue] ←────────────────── [Observe outcome]
        ↓
[Online Policy Update (Contextual Bandit)]
```

---

## Reinforcement Learning Framing

Even in the non-RL approach, the architecture is designed with RL in mind:

- **State:** Feature vector at time `t` (physiological signals + lag features + subsession context)
- **Action:** Discretised system parameter (e.g., difficulty threshold, feedback gain)
- **Reward:** `ProtocolValue(t+1) - ProtocolValue(t)`
- **Policy:** Learned mapping from state to action

The supervised approach approximates the value function `Q(s, a)` without explicit exploration. The RL approach (contextual bandit or DQN) learns this mapping directly from reward signals.

For production, a **contextual bandit with Thompson Sampling** would be preferred for its balance of exploration/exploitation and computational efficiency in a real-time setting.

---

## Compute Resources

Where needed, model training and hyperparameter optimisation were performed on **Deucalion** (INCD National Supercomputer, University of Minho). This enabled experiments that would not be feasible within the 6–8h challenge window on a local machine.

---

*© 2026 — Neroes Technical Challenge Submission*
