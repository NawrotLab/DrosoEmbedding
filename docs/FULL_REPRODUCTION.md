# Full reproduction from raw data

> **Status: in preparation.** The raw-data repository is not yet public. This page will be completed when it is; the outline below describes what it will cover. To reproduce the paper's figures today, follow the [Quick start](../README.md#quick-start-reproduce-all-figures) — it needs neither raw data nor a GPU.

The Quick start reproduces every figure from the published preprocessed frames, trained checkpoints, and cached evaluation results. This page covers the two levels beyond that:

| Level | Starts from | You re-run | Needs |
|---|---|---|---|
| 1 — Figures ([Quick start](../README.md#quick-start-reproduce-all-figures)) | Data repository | figure scripts | CPU (GPU optional) |
| 2 — Retrain and re-evaluate | Data repository (preprocessed frames + splits) | training sweep, evaluation, figures | GPU |
| 3 — Everything from raw recordings | Raw-data repository | preprocessing, neuropil knockouts, splits, then level 2 | GPU, [N — TBD] TB storage |

## Level 2 — Retrain and re-evaluate

Uses the same setup as the Quick start (steps 1–4), then:

```bash
# regenerate the split file for the other two tasks as well
TASK=MetabolicState_2 python -m scripts.analysis.regenerate_split_pickle
TASK=State_Modality_6 python -m scripts.analysis.regenerate_split_pickle

python -m scripts.training          # one run; see README "Training" for sweeps
python -m scripts.run_evaluation    # see README "Evaluation"
```

[TBD: sweep size used in the paper, seeds, expected GPU hours per task, how to point the figure scripts at your own results instead of the published ones.]

## Level 3 — From raw recordings

[TBD once the raw-data repository is public:]

- Raw-data repository link, size, and download instructions
- `.env` additions: `DROSO_RAW_RECORDINGS`, `DROSO_LOCAL_SCRATCH`, neuropil-mask location
- Preprocessing: `python -m src.preprocessing.clean_dataset` (intact frames), `python -m src.preprocessing.clean_dataset_staticfill` (neuropil knockouts), `python -m src.preprocessing.compute_peak_times`
- Creating train/validation/test splits
- Hardware and runtime expectations

SLURM launchers for every stage are in [`scripts/cluster/`](../scripts/cluster/README.md); each has a plain `python -m ...` equivalent.
