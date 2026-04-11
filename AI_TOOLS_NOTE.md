# AI Tools Usage Note

As requested by the Neroes challenge briefing, here is a transparent account of how AI tools were used during this project.

---

## Tools Used

### Claude (Anthropic) — claude-sonnet-4

**Where it was used:**
- Scaffolding the initial EDA notebook structure and suggesting analytical angles for neurofeedback data
- Proposing the overall project architecture and file organisation
- Suggesting relevant statistical tests for baseline vs. game session comparison (Mann-Whitney U)
- Drafting boilerplate code patterns (e.g., data loading loops, plot styling)
- Reviewing and improving docstrings and README prose

**Where it was NOT used:**
- All data interpretation is my own — Claude was not shown the actual data outputs
- All modelling decisions (choice of features, model architecture, RL formulation) are my own
- Critical reflections, problem framing, and limitations sections are written entirely by me
- No code was used verbatim from AI without understanding and adapting it

### GitHub Copilot (VSCode)
- Autocompletion of repetitive code patterns (e.g., matplotlib styling, pandas groupby chains)
- Not used for logic or algorithmic decisions

---

## Philosophy

I treat AI tools the same way I treat Stack Overflow or library documentation: they accelerate syntax and boilerplate, but the reasoning, judgment, and responsibility for correctness remain entirely mine. Every AI-suggested line of code was read, understood, and either accepted, modified, or rejected based on whether it made sense for the problem at hand.

This is especially important in a neurofeedback context where data interpretation errors could have downstream consequences in a real clinical or performance setting.
