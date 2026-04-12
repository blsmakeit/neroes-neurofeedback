# AI Tools Usage Note

As requested by the Neroes challenge briefing, here is a transparent and specific account of how AI tools were used during this project.

---

## Tools Used

### Claude (Anthropic) - claude-sonnet-4-6

**Specific contributions:**

- **EDA notebook scaffolding** - suggested the initial cell structure and section order for the exploratory analysis notebook
- **Walk-forward CV approach** - proposed using walk-forward cross-validation for temporal data after I described the dataset structure (though the decision to adopt it, and why, was mine)
- **Debugging `read_csv` with `skiprows=8`** - helped identify that the EEG export files had an 8-line metadata header causing silent parse errors
- **Debugging float32 overflow** - helped trace the overflow in the RL feature matrix to values exceeding the float32 range during reward accumulation in `05_rl_agent.ipynb`
- **Matplotlib boilerplate** - drafted reusable styling patterns (figure size, axis formatting, colour maps) used as a starting point for plots

**Not used for:**
- Data interpretation - Claude was not shown actual output values or results
- Any modelling or architectural decisions
- Scientific conclusions or paper content

---

### Claude Code (VSCode Extension)

**Specific contributions:**

- **`kernel.json` path fix** - resolved a Jupyter kernel registration issue caused by a broken path in the kernel spec
- **File path corrections** - fixed all relative import and data-loading paths that broke when notebooks were moved into the `notebooks/` subdirectory
- **`baselines.py` NaN masking bug** - identified and fixed incorrect boolean mask logic that was silently dropping valid rows during baseline computation
- **Float32 overflow fix in `05_rl_agent.ipynb`** - applied the dtype correction to prevent overflow in the RL feature matrix

---

## What Is Entirely My Own Work

The following were not assisted by AI at any point:

- **Problem framing** - deciding which aspects of neurofeedback data are worth modelling, and what a meaningful prediction target looks like
- **Action space design** - the discretisation of the RL action space and the reasoning behind it
- **Choice of walk-forward CV** - the decision to use it over random splits, and the understanding of why random splits are invalid for this data
- **Metric selection** - recognising that R² ≈ 0 does not mean the model is useless, and that directional accuracy is the appropriate metric for this problem
- **RL failure analysis** - the diagnosis that poor action coverage, not model capacity, explains why the RL agent underperformed
- **All EDA interpretations** - every conclusion drawn from plots, distributions, and statistical tests
- **Architecture decisions** - the choice of LinUCB over DQN, and the reasoning grounded in data size and exploration constraints
- **Scientific paper content and conclusions** - all written independently; AI was not used to draft or edit the paper

---

## Philosophy

AI tools were used to accelerate mechanical tasks - debugging obscure errors, fixing paths, generating boilerplate. Every line they produced was read, understood, and adapted or rejected based on whether it made sense for the problem. Reasoning, judgment, and scientific responsibility remain entirely mine. This is especially important in a neurofeedback context, where analytical errors could have real consequences in clinical or performance settings.
