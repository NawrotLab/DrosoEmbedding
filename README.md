# DrosoEmbedding

Code accompanying the paper:

> **"Deep Representation Learning Reveals Factorized Neural Codes in Whole-Brain Population Dynamics"** — Abdelbaki et al. — *[Journal/venue, year — TBD]*

A wiring-agnostic deep-learning framework — a convolutional encoder followed by a temporal transformer — that learns compact representations directly from whole-brain calcium imaging of *Drosophila melanogaster*, without neuronal identification or anatomical annotation. Trained only to classify 16 factorially combined sensory and internal-state conditions from flat class labels, the model organizes brain-wide activity along three near-orthogonal latent axes: metabolic state, sensory modality, and stimulus valence. GradCAM attribution and neuropil-knockout analyses then link these representations to brain regions.

This is the **code** repository: everything needed to train and evaluate the model and to reproduce every figure in the paper. It has two companions:

- **Data** — [gin.g-node.org/nawrotlab/DrosoEmbedding_WBCI](https://gin.g-node.org/nawrotlab/DrosoEmbedding_WBCI), preprocessed imaging data, trained model checkpoints, and evaluation results
- **Raw data** — [link — TBD], the original whole-brain calcium imaging recordings and neuropil masks

You'll need the Data repository alongside this one to actually run anything below — this repository holds no imaging data itself.

---

## Quick start: reproduce all figures

No raw data, no GPU, and no git-annex needed — the Data repository's `results/` folder (~16MB, plain `git`-tracked) is everything the figures are built from. Measured on a from-scratch clone: cloning takes seconds, and every figure loads and renders in under a second, under 700MB peak memory.

Prerequisites:

- **Python 3.10, 3.11 or 3.12** — not 3.13+, for which the pinned PyTorch version has no builds
- **[Poetry](https://python-poetry.org/docs/#installation) ≥ 2.0** — `curl -sSL https://install.python-poetry.org | python3 -`

```bash
# 1. Code + dependencies
git clone https://github.com/NawrotLab/DrosoEmbedding.git
cd DrosoEmbedding
poetry env use python3.12        # or python3.10 / python3.11 -- whichever you have
poetry install --no-root
source "$(poetry env info --path)/bin/activate"

# 2. Data -- a shallow, plain clone is enough, no git-annex required
#    (--depth 1: skips full history, which still carries some now-annexed
#    content from past commits; NOTE: no ".git" at the end of the URL --
#    GIN only serves file content directly on the plain URL)
git clone --depth 1 https://gin.g-node.org/nawrotlab/DrosoEmbedding_WBCI ../DrosoEmbedding_WBCI

# 3. Point this repo at it
cp .env.example .env
# edit .env: set DROSO_ROOT to this checkout, DROSO_DATA_REPO to ../DrosoEmbedding_WBCI

# 4. Generate every figure -> results/CombiPlots/
for f in scripts/figures/run_*.py; do python -m "scripts.figures.$(basename "$f" .py)"; done
```

To run a single figure, see the table under [Figures](#figures). To go further than the figures — retrain the models, or rebuild everything from the raw recordings (GPU required) — see [docs/FULL_REPRODUCTION.md](docs/FULL_REPRODUCTION.md).

---

## Repository structure

```
scripts/
  setup_data.sh              # Level-2+ only: extract frame archives, write .env
  training.py                # training entry point
  run_evaluation.py          # evaluation: accuracy, latent space, GradCAMs
  figures/run_fig*.py        # reproduce paper figures (one script per figure)
  analysis/                  # diagnostic, verification, and packaging scripts
  cluster/                   # optional SLURM launchers -- see cluster/README.md
src/
  data/dataset.py            # dataset and data loading
  models/cnn_transformer.py  # model architecture
  preprocessing/             # raw recordings -> preprocessed frames, train/val/test splitting
  training/training_routine.py
  analysis/                  # neuropil-knockout, GradCAM, and example-frame results (save/load)
  utils/
    config.yaml               # all hyperparameters and path placeholders
    config_loader.py          # loads config + applies .env overrides
    helpers.py                 # sweep-result aggregation, plotting utilities
    resource_log.py            # wall-time/peak-memory reporting for load/compute paths
  visualization/              # figure and plotting modules
```

---

## Configuration

All machine-specific paths live in `.env` (git-ignored, see `.env.example`). Level 1 needs only two entries — `DROSO_ROOT` (this repository) and `DROSO_DATA_REPO` (your Data repository clone); every other path derives from those. Levels 2 and 3 need a few more (preprocessed frames, raw recordings, scratch storage) — `.env.example` documents each, grouped by the level that first needs it. Hyperparameters and all other settings are in `src/utils/config.yaml`.

---

## Retraining and full reproduction

Reproducing the figures needs none of this. To retrain and re-evaluate the models yourself, or to rebuild everything from the raw recordings (GPU required), see **[docs/FULL_REPRODUCTION.md](docs/FULL_REPRODUCTION.md)**.

## Figures

Each script in `scripts/figures/` reproduces one figure or supplementary figure, in paper order:

| Figure | Script |
|---|---|
| Fig. 1 — Study overview | `run_fig1_overview.py` |
| Fig. S1 — Experimental design | `run_figS1_exp_design.py` |
| Fig. S2 — Latent-space interaction effects | `run_figS2_latent_interactions.py` |
| Fig. 2 — Latent-space geometry | `run_fig2_latent.py` |
| Fig. S3 — Training/validation curves across the sweep | `run_figS3_training_curves.py` |
| Fig. 3 — Classification performance and error structure | `run_fig3_accuracy_error.py` |
| Fig. 4 — GradCAM-based neuropil importance | `run_fig4_gradcam_neuropils.py` |
| Fig. S4 — Neuropil-knockout importance | `run_figS4_ko_neuropils.py` |

All 8 read only from `results/` — the Data repository's Level-1 folder. Run any one with `python -m scripts.figures.<script name without .py>`; output goes to `results/CombiPlots/`. `run_fig2_latent.py`, `run_figS2_latent_interactions.py`, `run_fig3_accuracy_error.py`, and `run_figS3_training_curves.py` read `results/aggregated_results.pkl`, an aggregate of the full evaluation sweep (~750 per-model results, which themselves live in `inference/sweep_cache/` — see below) built once so figures don't reopen every pkl on every run — it ships pre-built in the Data repository; see [docs/FULL_REPRODUCTION.md](docs/FULL_REPRODUCTION.md) if you need to rebuild it from your own evaluation runs. `run_fig4_gradcam_neuropils.py` and `run_figS4_ko_neuropils.py` similarly cache their (GPU-computed) results in `results/`; pass `--recompute` to force a fresh run instead of loading what's published.

The figures can equally be built from your own training and evaluation runs instead of the published results — see [docs/FULL_REPRODUCTION.md](docs/FULL_REPRODUCTION.md).

---

## Classification tasks

The models are trained on three classification tasks of increasing granularity. Their config keys are the values the `TASK` environment variable accepts, and the names of the per-task folders in the Data repository (`inference/models/<task>/`, `inference/sweep_cache/<task>/`):

| Config key | Classes | Description |
|---|---|---|
| `MetabolicState_2` | 2 | Starved vs. Fed |
| `State_Modality_6` | 6 | Metabolic state × stimulus modality |
| `State_Modality_Valence_16` | 16 | State × modality × valence |

---

## Citation

If you use this code, please cite:

> Abdelbaki et al. "Deep Representation Learning Reveals Factorized Neural Codes in Whole-Brain Population Dynamics." [Journal/venue, year — TBD]

## License

MIT — see [LICENSE](LICENSE).
