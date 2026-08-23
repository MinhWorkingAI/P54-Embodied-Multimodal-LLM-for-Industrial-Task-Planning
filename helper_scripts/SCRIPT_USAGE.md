# Helper Scripts — Usage Guide

This file documents every script in `helper_scripts/`. Each script gets its own section below, in
the same fixed four-field format (**Description**, **Input**, **Output**), so the file stays easy
to scan as scripts are added.

**Adding a new script:** copy the `Template` section, rename its heading, fill in the four fields,
and add a line to the Index. Don't reorder or reformat existing sections — each one is self
contained, so appending a new one never requires touching the others.

## Index

- [`generate_Report.py`](#generate_reportpy)

---

## Template

> Copy everything between the `###` heading and the next `---` when documenting a new script. Keep
> the field order and bold labels exactly as shown.

### `<script_filename>.py`

**Description:**
One short paragraph — what the script does, and why it exists (what problem/task it covers).

**Input:**
- CLI arguments / environment variables / files the script reads, with their defaults.
- Anything it imports or references from the main codebase (`llm_backend/`, `simulation_backend/`,
  `task_planner/`, etc.) — module imports, config files, or generated data it depends on. State
  "none" if the script is fully standalone.

**Output:**
- What the script produces and in what format (file, printed report, exit code, etc.).
- Where that output is written — the exact path (relative to the repo root) — or "printed to
  terminal only" if it writes no file.

---

## `generate_Report.py`

**Description:**
Compiles the P54 evaluation results into a formatted academic PDF report: cover page, executive
summary, key takeaways, use cases, architectural novelty, system architecture overview, evaluation
results tables, methodology, and references. The results section (6. Evaluation & Benchmarking
Experimental Data) is built directly from the exported evaluation CSV — none of the numbers in that
section are hardcoded in the script.

**Input:**
- CLI arguments:
  - `--csv` — path to the evaluation metrics CSV. Default: `llm_backend/LLM_eval/evaluation_metrics.csv`.
  - `--json` — path to the raw per-case evaluation results JSON. Default:
    `llm_backend/LLM_eval/evaluation_results.json`. Accepted but not currently consumed by any
    section — reserved for a future per-case drill-down.
  - `--output` — output PDF filename. Default: `P54_Evaluation_Report.pdf`.
- Reads `llm_backend/LLM_eval/evaluation_metrics.csv` (or whatever `--csv` points to), which is
  produced by `llm_backend/LLM_eval/eval_report.py --export`. That file must exist first — if it's
  missing or empty, Section 6 renders a "no evaluation data found" notice instead of a results
  table, rather than falling back to fabricated numbers.
- Does not import any code from the main pipeline (`llm_backend/`, `simulation_backend/`,
  `task_planner/`) — it only reads the evaluation pipeline's exported *output* file, not its code.
- Third-party dependency: `reportlab` (see `requirements.txt`).

**Output:**
- A single PDF report.
- Written next to the script itself — `helper_scripts/<output filename>` (default
  `helper_scripts/P54_Evaluation_Report.pdf`) — regardless of the working directory the script is
  run from. Passing an absolute path via `--output` overrides this and writes there instead.
  Relative to that resolution rule, the effective default path is:
  `helper_scripts/P54_Evaluation_Report.pdf`.
- Progress messages (`Generating academic report → ...`, `Academic Report Generation Complete`) are
  printed to the terminal; no other output is produced.

---
