# AI vs. Human Text — An Honestly-Evaluated Detector

Code behind the figures in the AI-imperfection post
([app.chinmayarora.com/blog](https://app.chinmayarora.com/blog/)).

The premise: human writing carries "imperfection" signals — burstiness, edit
entropy, perplexity variance — that machine text tends to smooth out. This
artifact trains a detector on those features and, more importantly, evaluates
it honestly: not just an AUC, but a calibration curve and an explicit look at
where it fails.

## Data caveat (read this first)

**The data here is synthetic** — a labeled feature model with realistic overlap
between the human and AI classes. The evaluation *pipeline* (ROC, calibration,
failure analysis) is the real artifact. Captions state this.

To promote this to a real finding, swap the synthetic features for a real
paired corpus — [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) is the
standard human-vs-ChatGPT benchmark — and extract the same three features.

## Figures produced

`generate_figures.py` writes to `./figures/`:

| File | What it shows |
|---|---|
| `detector_evaluation.png` | Three panels: feature-distribution overlap, ROC curve (AUC ≈ 0.93), and a calibration curve against the diagonal. |

## Why the overlap panel matters

A high AUC is easy to over-trust. The feature-overlap panel is deliberately
foregrounded because that overlap *is* the error surface: heavily-edited AI text
and terse human text land in the confusion zone, and no threshold separates
them cleanly. An honest detector artifact shows where it breaks, not just where
it works.

## Reproducing

```bash
pip install matplotlib numpy scikit-learn
python generate_figures.py
```
