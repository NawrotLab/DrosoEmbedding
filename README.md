# DrosoEmbedding

Code accompanying the paper:

> **"Deep Representation Learning Reveals Factorized Neural Codes in Whole-Brain Population Dynamics"** — [Authors — TBD] — *[Journal/venue, year — TBD]*

A CNN-Transformer model that classifies *Drosophila* whole-brain calcium imaging sequences by metabolic state, stimulus modality, and valence. The model learns a latent embedding of brain-wide activity, and this repository contains everything needed to train it, evaluate it, and reproduce every figure in the paper.

This is the **code** repository. It has two companions:

- **Data** — [gin.g-node.org/nawrotlab/DrosoEmbedding_WBCI](https://gin.g-node.org/nawrotlab/DrosoEmbedding_WBCI), preprocessed imaging data, trained model checkpoints, and cached evaluation results
- **Raw data** — [link — TBD], the original whole-brain calcium imaging recordings and neuropil masks

You'll need the Data repository alongside this one to actually run anything below — this repository holds no imaging data itself.

---

## Quick start: reproduce all figures

No raw data or model training needed — the Data repository ships the trained checkpoints and cached evaluation results the figures are built from. A GPU is optional (Fig. 4 and Fig. S4 run model inference on the test set and are slow on CPU). You need about [N — TBD] GB of free disk space.

Prerequisites:

- **Python 3.10, 3.11 or 3.12** — not 3.13+, for which the pinned PyTorch version has no builds
- **[Poetry](https://python-poetry.org/docs/#installation) ≥ 2.0** — `curl -sSL https://install.python-poetry.org | python3 -`
- **[git-annex](https://git-annex.branchable.com/install/)** — `brew install git-annex` (macOS) or `sudo apt install git-annex` (Debian/Ubuntu)

```bash
# 1. Code + dependencies
git clone https://github.com/aminaabdelbaki/DrosoEmbedding.git
cd DrosoEmbedding
poetry env use python3.12        # or python3.10 / python3.11 -- whichever you have
poetry install --no-root
source "$(poetry env info --path)/bin/activate"

# 2. Data (clone next to the code, then download the file contents)
git clone https://gin.g-node.org/nawrotlab/DrosoEmbedding_WBCI.git ../DrosoEmbedding_WBCI
git -C ../DrosoEmbedding_WBCI annex get .

# 3. Extract the frame archives and write .env (one-time, safe to re-run)
bash scripts/setup_data.sh ../DrosoEmbedding_WBCI

# 4. Rebuild the train/val/test split file from the published assignments (one-time)
TASK=State_Modality_Valence_16 python -m scripts.analysis.regenerate_split_pickle

# 5. Generate every figure -> results/CombiPlots/
for f in scripts/figures/run_*.py; do python -m scripts.figures.$(basename "$f" .py); done
```

Short on disk or time? The 12 `data/ko_static_*.tars` archives (the bulk of the download) are only used by Fig. S4. Replace the `annex get .` in step 2 with `annex get data/intact.tars data/splits models evaluation` to skip them; every other figure still works, and `setup_data.sh` skips whatever wasn't fetched.

To run a single figure, see the table under [Figures](#figures). To go further than the figures — retrain the models, or rebuild everything from the raw recordings (GPU required) — see [docs/FULL_REPRODUCTION.md](docs/FULL_REPRODUCTION.md).

---

## Repository structure

```
scripts/
  setup_data.sh             # one-time data setup: extract frame archives, write .env
  training.py               # training entry point
  run_evaluation.py         # evaluation: accuracy, latent space, GradCAMs
  figures/run_figure_*.py   # reproduce paper figures (one script per figure)
  analysis/                 # diagnostic, verification, and packaging scripts
  cluster/                  # optional SLURM launchers -- see cluster/README.md
src/
  data/dataset.py           # dataset and data loading
  models/cnn_transformer.py # model architecture
  preprocessing/            # raw recordings -> preprocessed frames, train/val/test splitting
  training/training_routine.py
  analysis/ko_permutation.py   # neuropil-knockout importance analysis
  utils/
    config.yaml              # all hyperparameters and path placeholders
    config_loader.py         # loads config + applies .env overrides
    helpers.py                # sweep-result aggregation, plotting utilities
  visualization/             # figure and plotting modules
```

---

## Configuration

All machine-specific paths live in `.env` (git-ignored), which `scripts/setup_data.sh` writes for you from `.env.example`. It needs only two entries — `DROSO_ROOT` (this repository) and `DROSO_DATA_REPO` (your Data repository clone); every other path is derived from those. Edit `.env` if you move either clone. Hyperparameters and all other settings are in `src/utils/config.yaml`.

---

## Training

```bash
# single run
python -m scripts.training

# many runs (a hyperparameter/seed sweep)
bash scripts/cluster/training/run_sweep.sh
```

Set `RUN_ID`, `TASK`, `CNN_DIM`, `TRF_DIM`, etc. as environment variables to override `config.yaml` without editing it. Each run's best checkpoint is saved to `models/<task>/`.

## Evaluation

```bash
python -m scripts.run_evaluation
```

Produces accuracy, confusion matrices, latent-space embeddings (t-SNE), and class-mean GradCAMs for one model, saved to `evaluation/<task>/`. To evaluate an entire sweep at once, see `scripts/cluster/evaluation/` (`build_manifest.sh` + `run_array.sh` for a parallel SLURM array job, `summarize_sweep.sh` to check progress).

## Figures

`run_figure_latent.py` and `run_figure_accuracy_error.py` read from an aggregated cache of the full evaluation sweep rather than opening all ~750 per-model result files on every run. Build (or rebuild) it once with:

```bash
python -m scripts.analysis.build_eval_cache
```

This writes `evaluation/aggregated_results.pkl` (~8MB). It's built automatically the first time either figure script needs it, but running it standalone avoids the confusing side effect of "generating a figure" just to warm the cache. Pass `EVAL_CACHE_RECOMPUTE=true` to force a fresh rebuild (needed if any evaluation result changed, e.g. a resubmitted sweep straggler — there's no staleness check against the underlying files).

Each script in `scripts/figures/` reproduces one figure or supplementary figure, reading from published/evaluated results:

| Figure | Script | Reads from the Data repository |
|---|---|---|
| Fig. 1 — Study overview | `run_figure_overview.py` | example raw recordings |
| Fig. 2 — Latent-space geometry | `run_figure_latent.py` | `evaluation/` |
| Fig. 3 — Classification performance and error structure | `run_figure_accuracy_error.py` | `evaluation/` |
| Fig. 4 — GradCAM-based neuropil importance | `run_figure_gradcam_neuropils.py` | `models/`, intact frames, split file |
| Fig. S1 — Experimental design | `run_sfigure_exp_design.py` | intact frames |
| Fig. S2 — Latent-space interaction effects | `run_sfigure_latent_interactions.py` | `evaluation/` |
| Fig. S3 — Training/validation curves across the sweep | `run_sfigure_training_curves.py` | `evaluation/` |
| Fig. S4 — Neuropil-knockout importance | `run_sfigure_ko_neuropils.py` | `models/`, intact + KO frames, split file |

Run any one with `python -m scripts.figures.<script name without .py>`; output goes to `results/CombiPlots/`. Figs. 2, 3, S2 and S3 need only the small `evaluation/` folder, so they run without downloading or extracting any frames.

The figures can equally be built from your own training and evaluation runs above instead of the published results.

---

## Classification tasks

| Config key | Classes | Description |
|---|---|---|
| `MetabolicState_2` | 2 | Starved vs. Fed |
| `State_Modality_6` | 6 | Metabolic state × stimulus modality |
| `State_Modality_Valence_16` | 16 | State × modality × valence |

---

## Citation

If you use this code, please cite:

> [Authors — TBD]. "Deep Representation Learning Reveals Factorized Neural Codes in Whole-Brain Population Dynamics." [Journal/venue, year — TBD]

## License

MIT — see [LICENSE](LICENSE).
