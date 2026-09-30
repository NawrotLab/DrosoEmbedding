# Full reproduction from raw data

> **Status: in preparation.** The raw-data repository is not yet public. This page will be completed when it is; the outline below describes what it will cover. To reproduce the paper's figures today, follow the [Quick start](../README.md#quick-start-reproduce-all-figures) — it needs neither raw data nor a GPU, and takes minutes, not hours.

The Quick start reproduces every figure from the Data repository's `results/` folder (~16MB total). This page covers the two levels beyond that, which the Data repository's `inference/` folder is built for:

| Level | Starts from | You re-run | Needs |
|---|---|---|---|
| 1 — Figures ([Quick start](../README.md#quick-start-reproduce-all-figures)) | Data repository's `results/` (~16MB) | figure scripts | CPU (GPU optional; only used by --recompute) |
| 2 — Retrain and re-evaluate | Data repository's `inference/` (preprocessed frames + checkpoints + splits) | training sweep, evaluation, figures | GPU |
| 3 — Everything from raw recordings | Raw-data repository | preprocessing, neuropil knockouts, splits, then level 2 | GPU, [N — TBD] TB storage |

## Level 2 — Retrain and re-evaluate

Needs git-annex (the Quick start doesn't): the preprocessed-frame archives are large (~27GB intact + ~28GB neuropil-knockout, only the latter needed for Fig. S4) and stay annexed rather than plain-git-tracked.

```bash
# 1. Code + dependencies -- same as the Quick start
git clone https://github.com/NawrotLab/DrosoEmbedding.git
cd DrosoEmbedding
poetry env use python3.12        # or python3.10 / python3.11
poetry install --no-root
source "$(poetry env info --path)/bin/activate"

# 2. Data repository, this time with git-annex content (full clone, not
#    shallow -- git-annex needs full history for its branch-based tracking)
git clone git@gin.g-node.org:/nawrotlab/DrosoEmbedding_WBCI.git ../DrosoEmbedding_WBCI
cd ../DrosoEmbedding_WBCI && git annex init && git annex get --jobs=16 inference/preprocessed_frames/intact.tars
cd -   # back to the code repo

# 3. Extract the frame archives and write .env (one-time, safe to re-run)
bash scripts/setup_data.sh ../DrosoEmbedding_WBCI
```

Short on disk or time? The knockout archives (`ko_static_*.tars`, ~28GB, Fig. S4 only) can be skipped: fetch only `inference/preprocessed_frames/intact.tars` as above, and `setup_data.sh` will skip whatever wasn't fetched.

Measured on a real cluster connection: fetching all of `inference/` (~48GB total) took about 35 minutes. On a very large `get`, verify nothing silently failed with `git annex find --not --in=here` before moving on (empty output = everything is actually present) — `git annex get` is safe to just rerun if it isn't.

The split file (`meanZ_logTs_..._State_Modality_Valence_16.pickle` etc.) ships with the Data repository already, one per task — no separate regeneration step needed unless you want to rebuild it from scratch (`python -m scripts.analysis.regenerate_split_pickle`, `TASK=...` to pick the task).

Your own training/evaluation runs would otherwise write into the same `inference/models/`/`inference/sweep_cache/` folders as the published ones inside the Data repository clone. To keep them separate, set `DROSO_PUBLISH_ROOT` in `.env` to an empty directory of your choice instead of `${DROSO_DATA_REPO}` before continuing.

### Training

```bash
# single run
python -m scripts.training

# many runs (a hyperparameter/seed sweep)
bash scripts/cluster/training/run_sweep.sh
```

Set `RUN_ID`, `TASK`, `CNN_DIM`, `TRF_DIM`, etc. as environment variables to override `config.yaml` without editing it. Each run's best checkpoint is saved to `inference/models/<task>/`.

### Evaluation

```bash
python -m scripts.run_evaluation
```

Produces accuracy, confusion matrices, latent-space embeddings (t-SNE), and class-mean GradCAMs for one model, saved to `inference/sweep_cache/<task>/`. To evaluate an entire sweep at once, see `scripts/cluster/evaluation/` (`build_manifest.sh` + `run_array.sh` for a parallel SLURM array job, `summarize_sweep.sh` to check progress).

### Figures from your own results

`run_fig2_latent.py`, `run_figS2_latent_interactions.py`, `run_fig3_accuracy_error.py`, and `run_figS3_training_curves.py` read from an aggregate of the full evaluation sweep (`results/aggregated_results.pkl`, ~12MB) rather than opening all ~750 per-model result files (`inference/sweep_cache/<task>/`) on every run. It's built automatically the first time any of them needs it; to force a rebuild after your own evaluation runs (there's no staleness check against the underlying per-model files):

```bash
RESULTS_RECOMPUTE=true python -m scripts.analysis.build_results
```

`run_fig4_gradcam_neuropils.py` and `run_figS4_ko_neuropils.py` similarly cache their (GPU-computed) results in `results/`; pass `--recompute` to force a fresh run instead of loading what's there. Measured on an H200:

| Step | Wall time | Peak memory |
|---|---|---|
| `run_fig4_gradcam_neuropils.py --recompute` (GradCAM, one model over the test set) | ~2 min | ~13GB |
| `run_figS4_ko_neuropils.py --recompute` (50 checkpoints × 12 neuropils) | **~17.7 hours** | ~1.7GB |

Fig. S4's recompute is by far the most expensive step here: each of the 50 checkpoints runs a baseline pass plus 12 knockout passes over the full test set (~20 min per checkpoint). Give the job at least a 24-hour limit — a 12-hour SLURM limit gets killed at roughly 35/50 checkpoints, and there is currently no resume capability, so a killed job means starting over. `scripts/cluster/figures/run_level2_recompute.sh` runs both steps with a suitable time limit.

[TBD: sweep size used in the paper, seeds, expected GPU hours for a full training sweep.]

## Level 3 — From raw recordings

[TBD once the raw-data repository is public:]

- Raw-data repository link, size, and download instructions
- `.env` addition: `DROSO_RAW_RECORDINGS` (already in `.env.example`, derived from `DROSO_DATA_REPO`), plus the neuropil-mask location
- Preprocessing: `python -m src.preprocessing.clean_dataset` (intact frames), `python -m src.preprocessing.clean_dataset_staticfill` (neuropil knockouts), `python -m src.preprocessing.compute_peak_times`
- Creating train/validation/test splits
- Hardware and runtime expectations

SLURM launchers for every stage are in [`scripts/cluster/`](../scripts/cluster/README.md); each has a plain `python -m ...` equivalent.
