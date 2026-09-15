# DrosoEmbedding

Code accompanying the paper:

> **"Deep Representation Learning Reveals Factorized Neural Codes in Whole-Brain Population Dynamics"** — [Authors — TBD] — *[Journal/venue, year — TBD]*

A CNN-Transformer model that classifies *Drosophila* whole-brain calcium imaging sequences by metabolic state, stimulus modality, and valence. The model learns a latent embedding of brain-wide activity, and this repository contains everything needed to train it, evaluate it, and reproduce every figure in the paper.

This is the **code** repository. It has two companions:

- **Data** — [G-Node link — TBD], preprocessed imaging data, trained model checkpoints, and cached evaluation results
- **Raw data** — [link — TBD], the original whole-brain calcium imaging recordings and neuropil masks

You'll need the Data repository alongside this one to actually run anything below — this repository holds no imaging data itself.

---

## Repository structure

```
scripts/
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

## Environment setup

Requires Python ≥ 3.10. Dependencies are managed with [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/aminaabdelbaki/DrosoEmbedding.git
cd DrosoEmbedding
poetry install
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Data setup

Clone the [Data repository — TBD] and fetch its content, then point this repository at it:

```bash
cp .env.example .env
```

Edit `.env` to set `DROSO_DATA_ROOT`/`DROSO_ALLT_BASE`/etc. to your Data repo clone — see `.env.example` for the full list of path variables and what each one is for.

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

Each script in `scripts/figures/` reproduces one figure or supplementary figure, reading from published/evaluated results:

| Script | Figure |
|---|---|
| `run_figure_overview.py` | Study overview |
| `run_figure_latent.py` | Latent-space geometry |
| `run_figure_accuracy_error.py` | Classification performance and error structure |
| `run_figure_gradcam_neuropils.py` | GradCAM-based neuropil importance |
| `run_sfigure_exp_design.py` | Experimental design |
| `run_sfigure_training_curves.py` | Training/validation curves across the sweep |
| `run_sfigure_latent_interactions.py` | Latent-space interaction effects |
| `run_sfigure_ko_neuropils.py` | Neuropil-knockout importance |

Run them after the evaluation results they depend on exist (either from the Data repository directly, or from your own evaluation runs above).

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

[License — TBD]
