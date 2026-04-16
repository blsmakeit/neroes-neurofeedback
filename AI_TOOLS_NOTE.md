# AI Tools Usage Note

As requested by the Neroes challenge briefing, here is a transparent and specific account of how AI tools were used during this project. The scope covers both Phase 1 (single-session prototype) and Phase 2 (multi-session + cross-user pipeline).

---

## Tools Used

### Claude (Anthropic) — claude-sonnet-4-6

Used as a conversational assistant for debugging and scaffolding.

**Specific contributions:**

- **EDA notebook scaffolding** — suggested the initial cell structure and section order for the exploratory analysis notebooks
- **Walk-forward CV approach** — proposed using walk-forward cross-validation after I described the dataset structure (the decision to adopt it, and the understanding of why random splits are invalid for temporal data, is mine)
- **Debugging `read_csv` with `skiprows=8`** — helped identify that EEG export files had an 8-line metadata header causing silent parse errors
- **Debugging float32 overflow** — traced overflow in the RL feature matrix to values exceeding float32 range during reward accumulation
- **Matplotlib boilerplate** — drafted reusable styling patterns (figure size, axis formatting, colour palettes) used as a starting point for exploratory plots
- **FQI identification argument** — helped me formalise the Bellman bootstrap argument: why Q(s, Raise) is unidentifiable when p(Raise) = 0.06, and why more data amplifies rather than resolves the bias

**Not used for:**
- Data interpretation — Claude was not shown actual output values or plots to interpret
- Feature selection decisions
- Scientific conclusions or paper content

---

### Claude Code (VSCode Extension) — claude-sonnet-4-6

Used as a coding assistant throughout both phases. Contributions were substantially more extensive in Phase 2.

**Phase 1 specific contributions:**

- **`kernel.json` path fix** — resolved a Jupyter kernel registration issue caused by a broken path in the kernel spec
- **File path corrections** — fixed relative import and data-loading paths that broke when notebooks were moved into the `notebooks/` subdirectory
- **`baselines.py` NaN masking bug** — identified and fixed incorrect boolean mask logic that was silently dropping valid rows during baseline computation
- **Float32 overflow fix** — applied the dtype correction in `05_rl_agent.ipynb`

**Phase 2 specific contributions:**

- **Multi-session pipeline structure** — generated the notebook scaffolding for notebooks 07–11, including the walk-forward CV loop, session-indexed feature engineering, and cross-user experiment structure
- **Figure generation** — wrote the full matplotlib code for all `u2_*` figures and the system architecture figure (`outputs/figures/system_architecture.png`), given precise specifications (layout, colours, panel content, data to display)
- **Paper compilation** — assembled `paper/short_paper_v2.md` from structured inputs I provided (section outlines, all numerical results, tables, and figure references); also debugged and ran the pandoc + XeLaTeX pipeline to produce the PDF
- **LaTeX / pandoc debugging** — resolved font encoding issues (Latin Modern → DejaVu Serif for Unicode support), figure float placement (`\def\fps@figure{H}`), and resource path resolution for embedded figures
- **Cross-user experiment code** — implemented the 4-experiment transfer learning structure (A: U1→U2, B: U2→U1, C: combined→each user, D: held-out test) after I specified the experimental design and what features to use

---

## What Is Entirely My Own Work

The following were not assisted by AI at any point:

- **Problem framing** — deciding which aspects of neurofeedback data are worth modelling, and what a meaningful prediction target looks like
- **Action space design** — the discretisation into Hold / Raise / Lower and the reasoning behind deriving it from inter-subsession transitions
- **Session QC criteria** — the 80% GoodSignalQuality threshold, the anomalous PV std cutoff, and the decision to retain sessions 9 and 5/ss4 as flagged rather than excluded
- **Feature selection** — the decision to drop all 100 EEG spectral columns for User 2 after zero-variance verification, and to retain only behavioural/temporal features
- **Metric selection** — recognising that R² ≈ 0 does not mean the model is useless, and that directional accuracy is the appropriate metric for this problem
- **RL failure analysis** — the diagnosis that 94:6 action imbalance is a formal identification problem, not a sample size problem; the empirical confirmation that more data (16.5×) makes FQI worse, not better
- **Cross-user generalisation insight** — the observation that 12 shared features produce a combined model that outperforms the 246-feature personalised model; this was unexpected and is the key scientific finding of Phase 2
- **All EDA interpretations** — every conclusion drawn from plots, distributions, and statistical tests
- **Architecture decisions** — the choice of LinUCB over DQN, the offline-only constraint, the deployment roadmap phasing
- **Deployment roadmap** — the four-phase plan (zero-shot population model → ε-greedy online exploration → user embedding fine-tuning → full offline RL) is based on my analysis of what the data shows

---

## Philosophy

AI tools were used to accelerate mechanical tasks — generating boilerplate, fixing tooling errors, producing figures from specifications I wrote, and assembling documents from results I produced. Every line of code and every paragraph was read, verified against the actual data, and either adopted or rejected based on whether it correctly represented the analysis.

The scientific reasoning, experimental design, and conclusions are entirely mine. This matters especially in a neurofeedback context: the analytical pipeline feeds into system behaviour that could affect real participants, so the responsibility for what the system does and why must remain with the researcher, not the tool.
