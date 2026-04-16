# AI Tools Usage Note

As requested by the Neroes challenge briefing, here is a transparent and specific account of how AI tools were used during this project. The scope covers both Phase 1 (single-session prototype) and Phase 2 (multi-session + cross-user pipeline).

---

## Tools Used

### Claude (Anthropic) — claude-sonnet-4-6

Used as a conversational assistant for debugging specific technical errors.

**Specific contributions:**

- **Debugging `read_csv` with `skiprows=8`** — helped identify that EEG export files had an 8-line metadata header causing silent parse errors
- **Debugging float32 overflow** — traced overflow in the RL feature matrix to values exceeding the float32 range during reward accumulation

**Not used for:**
- Data interpretation — Claude was not shown actual output values or plots to interpret
- Any modelling, feature engineering, or architectural decisions
- Scientific conclusions or paper content

---

### Claude Code (VSCode Extension) — claude-sonnet-4-6

Used as a coding assistant for mechanical and tooling tasks.

**Phase 1 specific contributions:**

- **`kernel.json` path fix** — resolved a Jupyter kernel registration issue caused by a broken path in the kernel spec
- **File path corrections** — fixed relative import and data-loading paths that broke when notebooks were moved into the `notebooks/` subdirectory
- **`baselines.py` NaN masking bug** — identified and fixed incorrect boolean mask logic that was silently dropping valid rows during baseline computation
- **Float32 overflow fix** — applied the dtype correction in `05_rl_agent.ipynb`

**Phase 2 specific contributions:**

- **Figure generation** — wrote matplotlib code to produce the `u2_*` figures and the system architecture diagram, from detailed layout and content specifications I provided (exact axes, colour codes, data labels, panel structure)
- **Paper compilation** — assembled `paper/short_paper_v2.md` from section outlines, all numerical results, tables, and figure references I provided; debugged the pandoc + XeLaTeX pipeline (font encoding, figure float placement, resource path resolution)
- **README and documentation** — drafted the updated README and guide documents from structured content I provided

---

## What Is Entirely My Own Work

All technical and scientific work on the project is mine. Specifically:

- **All analysis and notebooks (01–11)** — every cell, every design decision, every result
- **Problem framing** — deciding what to model, what the prediction target means, and how to frame it as a sequential decision problem
- **Data understanding** — interpreting the EEG export format, understanding the ProtocolValue formula, identifying the calibration coefficient structure
- **Session quality control** — the 80% GoodSignalQuality threshold, the anomalous PV standard deviation criterion, the decision on which sessions to exclude and which to retain with an anomaly flag
- **Feature engineering** — all decisions: which EEG columns to drop (zero-variance verification), the per-session z-normalisation scheme, the choice of lag orders (1, 2, 5), rolling window sizes (5, 10, 20), and calibration group encoding
- **Walk-forward cross-validation design** — the decision to use it over random splits, and the understanding of why temporal ordering must be preserved
- **Metric selection** — recognising that R² ≈ 0 does not mean the model is useless, and that directional accuracy is the appropriate metric for protocol decisions
- **RL framing** — the MDP definition (state, action, reward, horizon), the choice of LinUCB and FQI, the reward clipping decision
- **RL failure analysis** — the diagnosis that 94:6 action imbalance is a formal identification problem; the empirical confirmation that 16.5× more data made FQI 19× worse; the formal Bellman bootstrap argument
- **Cross-user experiment design** — the decision to compute the shared feature intersection, the four-experiment structure (zero-shot transfer in both directions, combined model, held-out test), and the interpretation of why the combined 13-feature model outperforms the personalised 246-feature model
- **All EDA interpretations** — every conclusion drawn from distributions, ACF/PACF plots, correlation analyses, and session comparisons
- **Architecture and deployment decisions** — the four-phase deployment roadmap, the population-first strategy, the ε-greedy exploration requirement
- **Scientific paper content** — all written independently; structure, arguments, and conclusions are entirely mine

---

## Philosophy

AI tools were used exclusively for mechanical execution: fixing tooling errors, translating detailed specifications into boilerplate code, and compiling documents from content I produced. Every output was verified against the actual data and either adopted or discarded based on correctness.

The reasoning, judgment, and scientific responsibility are entirely mine. This is especially important in a neurofeedback context, where analytical decisions feed into system behaviour that can affect real participants.
