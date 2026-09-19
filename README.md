# DrosoEmbedding

Code accompanying the paper:

> **"Deep Representation Learning Reveals Factorized Neural Codes in Whole-Brain Population Dynamics"** — Abdelbaki et al. — *[Journal/venue, year — TBD]*

A wiring-agnostic deep-learning framework — a convolutional encoder followed by a temporal transformer — that learns compact representations directly from whole-brain calcium imaging of *Drosophila melanogaster*, without neuronal identification or anatomical annotation. Trained only to classify 16 factorially combined sensory and internal-state conditions from flat class labels, the model organizes brain-wide activity along three near-orthogonal latent axes: metabolic state, sensory modality, and stimulus valence. GradCAM attribution and neuropil-knockout analyses then link these representations to brain regions.

This is the **code** repository: everything needed to train and evaluate the model and to reproduce every figure in the paper. It has two companions:

- **Data** — [gin.g-node.org/nawrotlab/DrosoEmbedding_WBCI](https://gin.g-node.org/nawrotlab/DrosoEmbedding_WBCI), preprocessed imaging data, trained model checkpoints, and cached evaluation results
- **Raw data** — [link — TBD], the original whole-brain calcium imaging recordings and neuropil masks

You'll need the Data repository alongside this one to actually run anything below — this repository holds no imaging data itself.

---

## Quick start: reproduce all figures

No raw data or model training needed — the Data repository ships the trained checkpoints and cached evaluation results the figures are built from. A GPU is optional (Fig. 4 and Fig. S4 run model inference on the test set and are slow on CPU). You need about 100 GB of free disk space (a 47 GB download, plus the same again once the frame archives are extracted) — or about 40 GB if you skip the knockout data, see below.

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

# 2. Data (clone next to the code, then download the file contents; no GIN account needed)
#    NOTE: no ".git" at the end of the URL -- GIN only serves file contents on the plain URL
git clone https://gin.g-node.org/nawrotlab/DrosoEmbedding_WBCI ../DrosoEmbedding_WBCI
git -C ../DrosoEmbedding_WBCI annex get .

# 3. Extract the frame archives and write .env (one-time, safe to re-run)
bash scripts/setup_data.sh ../DrosoEmbedding_WBCI

# 4. Rebuild the train/val/test split file from the published assignments (one-time)
TASK=State_Modality_Valence_16 python -m scripts.analysis.regenerate_split_pickle

# 5. Generate every figure -> results/CombiPlots/
for f in scripts/figures/run_*.py; do python -m scripts.figures.$(basename "$f" .py); done
```

Short on disk or time? The 12 `ko_static_*.tars` knockout archives (29 GB of the 47 GB download) are only used by Fig. S4. Replace the `annex get .` in step 2 with `annex get --exclude='*/ko_static_*' .` to skip them; every other figure still works, and `setup_data.sh` skips whatever wasn't fetched.

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

## Retraining and full reproduction

Reproducing the figures needs none of this. To retrain and re-evaluate the models yourself, or to rebuild everything from the raw recordings (GPU required), see **[docs/FULL_REPRODUCTION.md](docs/FULL_REPRODUCTION.md)**.

## Figures

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

The figures can equally be built from your own training and evaluation runs instead of the published results — see [docs/FULL_REPRODUCTION.md](docs/FULL_REPRODUCTION.md).

---

## Classification tasks

The models are trained on three classification tasks of increasing granularity. Their config keys are the values the `TASK` environment variable accepts, and the names of the per-task folders in the Data repository (`models/<task>/`, `evaluation/<task>/`):

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
