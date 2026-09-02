# DrosoEmbedding

Code accompanying the paper:

> **[Paper title]** — [Authors] — *[Journal/Conference, Year]* — [DOI]

A CNN-Transformer model that classifies *Drosophila* whole-brain calcium imaging sequences by metabolic state, stimulus modality, and valence. The model learns a latent embedding of brain-wide activity and is evaluated via classification accuracy, latent-space geometry, and GradCAM-based interpretability.

---

## Repository structure

```
scripts/
  training.py              # training entry point
  run_evaluation.py        # evaluation and latent-space analysis
  figures/run_figure_*.py  # reproduce paper figures
  analysis/                # diagnostic and analysis scripts
  cluster/                 # optional SLURM launchers (preprocessing/, training/,
                            # evaluation/, figures/) -- see cluster/README.md
src/
  data/dataset.py         # dataset and data loading
  models/cnn_transformer.py
  preprocessing/          # within-animal train/val/test splitting
  training/training_routine.py
  utils/
    config.yaml           # all hyperparameters and path placeholders
    config_loader.py      # loads config + applies .env overrides
  visualization/          # figure and plotting modules
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

---

## Path configuration

All machine-specific paths are set via a `.env` file that is never committed.

```bash
cp .env.example .env
```

Edit `.env` and fill in your paths:

```dotenv
PYTHONPATH=/path/to/DrosoEmbedding

DROSO_ROOT=/path/to/DrosoEmbedding        # repo root
DROSO_DATA_ROOT=/path/to/data_root        # recordings spreadsheet and imgs4DL/
DROSO_ALLT_BASE=/path/to/allTs_base       # pre-extracted image tensors
DROSO_PEAK_IDS=/path/to/IDs_logTs.pickle  # peak-ID lookup table
```

These override the placeholder values in `src/utils/config.yaml`. All other hyperparameters (task, architecture, training schedule) are configured directly in that file.

---

## Training

```bash
# single run
python -m scripts.training --run_name my_run

# SLURM array job
sbatch scripts/shTrain.sh
```

Set `RUN_ID`, `TASK`, `EPOCHS`, etc. in `.env` or as environment variables to override config without editing the YAML. Results are written to `{DROSO_ROOT}/results/{task}_{run_id}/`.

## Evaluation

```bash
python -m scripts.run_evaluation
```

Produces accuracy metrics, confusion matrices, latent-space embeddings (t-SNE/UMAP), and GradCAM saliency maps under the run's `evaluation/` and `visualizations/` directories.

## Figures

Each `scripts/run_figure_*.py` script reproduces a specific paper figure. Run them after evaluation outputs exist for the relevant runs.

---

## Classification tasks

| Config key | Classes | Description |
|---|---|---|
| `MetabolicState_2` | 2 | Starved vs. Fed |
| `State_Modality_6` | 6 | Metabolic state × stimulus modality |
| `State_Modality_Valence_16` | 16 | State × modality × valence |
