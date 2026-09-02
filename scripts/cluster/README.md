# SLURM cluster scripts

These are convenience launchers for the lab's SLURM cluster — **not** the
reproduction path itself. Every step here has a plain, SLURM-free
equivalent documented in the top-level README (`python -m scripts.training`,
`python -m scripts.run_evaluation`, `python -m scripts.figures.run_figure_*`,
`python -m src.preprocessing.clean_dataset*`). If you don't have access to
a SLURM cluster, use those directly.

## Layout

```
preprocessing/
  run_main.sh        # main dataset prep + isolate-neuropil ablation runs
  run_static_ko.sh   # neuropil knockout via static-fill (used by Fig S4)
training/
  run_single.sh       # train one run
  run_sweep.sh         # train N runs from a config list (e.g. 50 replicates)
evaluation/
  run_single.sh       # evaluate one run
  run_sweep.sh         # evaluate N runs from a config list
figures/
  run_all.sh          # generate every paper figure, in paper order
  run_interactive.sh   # quick ad hoc figure generation on the interactive partition
verify_module_refs.py # dry-run checker: confirms every `python -m ...`
                       # reference in these scripts resolves and parses
```

Each stage follows the same two patterns:
- **`run_single.sh` / `run_main.sh` / `run_static_ko.sh` / `run_all.sh` / `run_interactive.sh`** — `#SBATCH`-decorated, submit directly with `sbatch`.
- **`run_sweep.sh`** — plain bash, run directly (`bash run_sweep.sh`, not `sbatch`); it generates and submits one job per entry in its `CONFIGURATIONS` array.

## Before you run these on your own cluster

Partition names, GPU types, and node names are specific to this lab's
SLURM setup and won't exist anywhere else — you'll need to edit the
`#SBATCH` lines (`--partition`, `--gres`, `--nodelist`) in every script to
match your own cluster's resource names.

The `run_single.sh`/`run_main.sh`/`run_static_ko.sh`/`run_all.sh`/
`run_interactive.sh` scripts also assume the repo is cloned directly under
your home directory (`BASE_DIR="$HOME/DrosoEmbedding"`) — this is a
deliberate choice, not an oversight: SLURM's `slurmd` on this cluster runs
the job with its working directory at its own spool dir rather than the
submission directory, so neither `$(dirname "$0")`-style path detection nor
`$SLURM_SUBMIT_DIR` reliably points back at the repo. Edit `BASE_DIR` if
your clone lives elsewhere.

Run `python scripts/cluster/verify_module_refs.py` after editing any of
these scripts (needs no GPU/data/config — just checks that every
`python -m ...` reference still resolves to a real, syntactically valid
module).
