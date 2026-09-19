# Full reproduction from raw data

> **Status: in preparation.** The raw-data repository is not yet public. This page will be completed when it is; the outline below describes what it will cover. To reproduce the paper's figures today, follow the [Quick start](../README.md#quick-start-reproduce-all-figures) — it needs neither raw data nor a GPU.

The Quick start reproduces every figure from the published preprocessed frames, trained checkpoints, and cached evaluation results. This page covers the two levels beyond that:

| Level | Starts from | You re-run | Needs |
|---|---|---|---|
| 1 — Figures ([Quick start](../README.md#quick-start-reproduce-all-figures)) | Data repository | figure scripts | CPU (GPU optional) |
| 2 — Retrain and re-evaluate | Data repository (preprocessed frames + splits) | training sweep, evaluation, figures | GPU |
| 3 — Everything from raw recordings | Raw-data repository | preprocessing, neuropil knockouts, splits, then level 2 | GPU, [N — TBD] TB storage |

## Level 2 — Retrain and re-evaluate

Uses the same setup as the Quick start (steps 1–4). The split file from step 4 covers the 16-class task only; regenerate it for the other two tasks as well:

```bash
TASK=MetabolicState_2 python -m scripts.analysis.regenerate_split_pickle
TASK=State_Modality_6 python -m scripts.analysis.regenerate_split_pickle
```

The Quick start's `.env` points `DROSO_PUBLISH_ROOT` at your Data repository clone, so new checkpoints and evaluation results would be written in among the published ones in its `models/` and `evaluation/` folders. To keep your own results separate, set `DROSO_PUBLISH_ROOT` in `.env` to an empty directory of your choice first.

### Training

```bash
# single run
python -m scripts.training

# many runs (a hyperparameter/seed sweep)
bash scripts/cluster/training/run_sweep.sh
```

Set `RUN_ID`, `TASK`, `CNN_DIM`, `TRF_DIM`, etc. as environment variables to override `config.yaml` without editing it. Each run's best checkpoint is saved to `models/<task>/`.

### Evaluation

```bash
python -m scripts.run_evaluation
```

Produces accuracy, confusion matrices, latent-space embeddings (t-SNE), and class-mean GradCAMs for one model, saved to `evaluation/<task>/`. To evaluate an entire sweep at once, see `scripts/cluster/evaluation/` (`build_manifest.sh` + `run_array.sh` for a parallel SLURM array job, `summarize_sweep.sh` to check progress).

### Figures from your own results

`run_figure_latent.py` and `run_figure_accuracy_error.py` read from an aggregated cache of the full evaluation sweep (`evaluation/aggregated_results.pkl`, ~8MB) rather than opening all ~750 per-model result files on every run. The Data repository ships this cache for the published results; after your own evaluation runs, rebuild it:

```bash
EVAL_CACHE_RECOMPUTE=true python -m scripts.analysis.build_eval_cache
```

There is no staleness check against the underlying files, so rebuild whenever any evaluation result changes (e.g. a resubmitted sweep straggler). Then run the figure scripts as in the Quick start.

[TBD: sweep size used in the paper, seeds, expected GPU hours per task.]

## Level 3 — From raw recordings

[TBD once the raw-data repository is public:]

- Raw-data repository link, size, and download instructions
- `.env` additions: `DROSO_RAW_RECORDINGS`, `DROSO_LOCAL_SCRATCH`, neuropil-mask location
- Preprocessing: `python -m src.preprocessing.clean_dataset` (intact frames), `python -m src.preprocessing.clean_dataset_staticfill` (neuropil knockouts), `python -m src.preprocessing.compute_peak_times`
- Creating train/validation/test splits
- Hardware and runtime expectations

SLURM launchers for every stage are in [`scripts/cluster/`](../scripts/cluster/README.md); each has a plain `python -m ...` equivalent.
