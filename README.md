# Isolating a counterfactual arithmetic edit

**Paper:** [PDF](paper_draft/main.pdf) · [LaTeX source](paper_draft/main.tex)

Can a pretrained language model answer `2+2=5` without changing other behavior? This project distinguishes an exact-string exception from an edit intended to transfer across paraphrases and dependent calculations.

## What was executed

- Qwen2.5-1.5B-Instruct, locally in BF16 on one NVIDIA RTX A6000.
- Three requests: `2+2 → 5`, `2+2 → 6`, `3+4 → 8`; three seeds each.
- **81 final-layer runs:** rank-8 MLP updates with exact-prompt training, paraphrase augmentation, or explicit preservation of alternate wordings; several KL locality weights.
- **18 exploratory middle-layer runs:** matched exact/paraphrase objectives at block 14, testing whether edit location changes propagation.
- 1,049 evaluation rows and 353 locality-training prompts. The low-digit arithmetic locality test holds out wording, not underlying sums. CounterFact preservation uses disjoint records.
- Two fixed training checkpoints; full-vocabulary next-token drift. Every final adapter also receives a **143-prompt complete-generation audit** (14,157 responses total).
- An additional **1,782 matched-control responses** test different inner sums with the same ordinary value. An executed literal wrapper provides the exact-isolation reference.

## Findings

All percentages below are means across three edit requests and three seeds. Complete answers are evaluated, not just their first digit.

| Setting | Exact success | Paraphrase adoption | Dependent adoption, base-correct only | Unrelated response change |
|---|---:|---:|---:|---:|
| Final layer, exact + KL=1 | 100% | 77.3% | 0.0% | 2.6% |
| Final layer, paraphrases + KL=1 | 100% | 91.2% | 0.0% | 2.9% |
| Final layer, boundary + KL=100 | 100% | 17.6% | 0.0% | 3.7% |
| Middle layer, exact + KL=1 | 100% | 67.6% | 37.0% | 13.8% |
| Middle layer, paraphrases + KL=1 | 100% | 73.6% | 32.7% | 11.2% |

Unregularized final-layer edits scored 100% on the desired first token but **0% on complete exact answers**, usually repeating the digit until the generation limit. The literal wrapper succeeded on all exact prompts and changed none of 426 off-key responses.

The results show a separation between paraphrase transfer and computational propagation, and a dependence on edit location. They do **not** prove that perfect parametric isolation is impossible, that only string exceptions can be isolated, or that an edit changes a model's beliefs. The base model solves only 9, 9, and 6 of the 18 dependent probes for the three requests; conditioned propagation has small denominators. Only one model, two distinct sums, and two parameter locations were studied. See the paper for seed variation and matched-control results.

## Reproduce from scratch

Use Python 3.12, a CUDA-capable GPU, and a TeX installation with `pdflatex` and `bibtex`. Packages are installed only in the local virtual environment. Model and dataset downloads are public; no LLM API calls are needed.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python src/run_all.py --workdir rerun
```

This writes new results and `rerun/paper_draft/main.pdf`, preserving the published results. It reuses existing immutable model/dataset downloads when available, or downloads the pinned revisions. The script logs each stage under `rerun/results/`. Existing complete run files are treated as checkpoints; choose a new work directory for a fully fresh repetition.

The initial final-layer sweep took about 83 seconds, and the middle-layer sweep about 170 seconds after process initialization on this GPU, excluding downloads/setup. These are measured timings for this environment, not general performance guarantees. Final-layer speed relies on exact caching of the frozen prefix computation. Full generations always run the complete model.

## Rebuild tables and paper from saved results

This requires no GPU or model download:

```bash
.venv/bin/python src/analyze.py
.venv/bin/python src/audit_analysis.py
.venv/bin/python src/paper_assets.py
cd paper_draft
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`src/validate.py` checks counts, split separation, labels, metric ranges, wrapper behavior, and saved cache-verification results. It operates on the saved results without GPU access.

The full workflow was independently rerun in a fresh directory. All 99 training-result files, all 99 generation-result files, matched controls, and the primary summary tables reproduced byte-for-byte. The command resumed successfully after an external termination signal; see `results/reproduction_check.json`.

## Workspace map

- `src/probes.py`: deterministic prompts, labels, and splits.
- `src/experiment.py`: final-layer training and distribution evaluation.
- `src/middle_layer.py`: full-forward/backward architectural replication.
- `src/generate_audit.py`: full-model greedy continuations for all final checkpoints.
- `src/matched_controls.py`: same-value, different-inner-sum specificity audit.
- `src/verify_cache.py`: same-batch equality and cross-batch numerical checks.
- `src/diagnostics.py`: descriptive activation-similarity and confidence diagnostics.
- `results/runs/`: per-step losses and per-prompt next-token measurements.
- `results/generation/`: complete generated texts and token IDs.
- `results/probes.json`, `results/design.md`: labels, splits, original design, and explicitly exploratory follow-ups.
- `results/summary.csv`, `results/generation_summary.csv`, `results/joint_summary.csv`: analysis-ready metrics.
- `results/provenance.json`, `results/metadata.json`: model/data revisions and settings.
- `paper_draft/`: eight-page conference-style paper, bibliography, generated tables and figures.
- `models/`, `data/`, `.cache/`, `.venv/`: ignored downloads, adapters, caches, and isolated environment.

## Numerical and evaluation notes

The primary generator explicitly sets `do_sample=False` and `repetition_penalty=1.0`. An initial audit inherited Qwen's default penalty of 1.1; those actual outputs are retained under `results/generation_repetition_penalty1p1/` and excluded from primary tables. BF16 batching also changes a small number of near-tied predictions. Same-batch edited cache/full-model logits match exactly in the validation; cross-batch next-token agreement for the original 81 runs is 99.36%. Complete-answer results use independent full-model generation.

Number-word normalization accepts only a whole answer (e.g. `Four.`); it does not extract numbers from explanations. Multi-digit sums are not scored as fully correct from first-token matches. Factual probes measure preservation of base behavior, not factual correctness. No results are simulated, and no failed or exploratory runs are presented as preregistered experiments.
