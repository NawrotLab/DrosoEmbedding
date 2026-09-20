"""
Read-only diagnostic: compare the per-run evaluation results that fed the
submitted figures (the old, hand-built _chkpt_finals/<task>/{runs,dim_runs}/
folders) against the current published evaluation/<task>/ layout.

For every task, at the canonical architecture (cnn_dim=16, trf_dim=16 by
default), it matches old and new runs by run number and reports:
  - which runs exist only in the old set / only in the new set (a different
    set of runs feeds the 50-run dots and means in Fig 3 c/d)
  - runs whose test accuracy or confusion matrix changed, largest first
    (a re-evaluated or differently-loaded model)
  - the lowest-accuracy runs in the new set (outlier runs that pull the
    Fig 3 panel-d bar means up; the plot's y-limit is max(mean) * 1.45, so
    outliers above that are clipped out of view but still counted)

Usage (from repo root, on a machine that has both trees):
    python -m scripts.analysis.compare_old_vs_new_eval
    python -m scripts.analysis.compare_old_vs_new_eval --old-dir /path/to/_chkpt_finals --new-dir /path/to/evaluation
    python -m scripts.analysis.compare_old_vs_new_eval --cnn 16 --trf 16 --top 15
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np

from src.utils.helpers import parse_run_id

# Old strict filenames: C{classes}_E{cnn}_H{trf}{_|-}{run}.pkl
_OLD_RE = re.compile(
    r'^C(?P<classes>\d+)_E(?P<cnn>\d+)_H(?P<trf>\d+)[_-](?P<run>\d+)\.pkl$', re.IGNORECASE
)


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _old_runs(task_dir: Path, cnn: int, trf: int) -> dict:
    """run index -> pkl path, from the old runs/ and dim_runs/ folders."""
    found, dupes = {}, []
    for sub in ('runs', 'dim_runs'):
        d = task_dir / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob('*.pkl')):
            m = _OLD_RE.match(p.name)
            if not m or int(m['cnn']) != cnn or int(m['trf']) != trf:
                continue
            run = int(m['run'])
            if run in found:
                dupes.append((run, found[run].parent.name, p.parent.name))
            found[run] = p
    if dupes:
        print(f"  WARNING: {len(dupes)} run(s) appear in both runs/ and dim_runs/ "
              f"(old loader counted each): {[d[0] for d in dupes][:10]}")
    return found


def _new_runs(task_dir: Path, cnn: int, trf: int) -> dict:
    """run index -> pkl path, from evaluation/<task>/{run_id}_evalResults.pkl."""
    found = {}
    for p in sorted(task_dir.glob('*_evalResults.pkl')):
        parsed = parse_run_id(p.name[:-len('_evalResults.pkl')])
        if parsed is None:
            continue
        _, c, t, run, is_control = parsed
        if is_control or c != cnn or t != trf:
            continue
        found[run] = p
    return found


def _acc_cm(path):
    d = _load(path)
    cm = d.get('confusion_matrix')
    return d.get('accuracy'), (np.array(cm) if cm is not None else None)


def compare_task(task, old_task_dir, new_task_dir, cnn, trf, top):
    print(f"\n{'=' * 70}\n{task}  (E{cnn}_H{trf})\n{'=' * 70}")
    old = _old_runs(old_task_dir, cnn, trf) if old_task_dir.is_dir() else {}
    new = _new_runs(new_task_dir, cnn, trf) if new_task_dir.is_dir() else {}
    print(f"  old runs: {len(old)}   new runs: {len(new)}")
    if not old:
        print(f"  (no old runs found under {old_task_dir})")

    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    if only_old:
        print(f"  only in OLD ({len(only_old)}): {only_old}")
    if only_new:
        print(f"  only in NEW ({len(only_new)}): {only_new}")

    rows, cm_changed = [], []
    for run in sorted(set(old) & set(new)):
        a_old, cm_old = _acc_cm(old[run])
        a_new, cm_new = _acc_cm(new[run])
        if a_old is None or a_new is None:
            continue
        rows.append((run, float(a_old), float(a_new)))
        if cm_old is not None and cm_new is not None and not np.array_equal(cm_old, cm_new):
            cm_changed.append(run)

    if rows:
        arr = np.array([(r[1], r[2]) for r in rows])
        delta = arr[:, 1] - arr[:, 0]
        n_diff = int(np.sum(np.abs(delta) > 1e-9))
        print(f"  matched runs: {len(rows)}; accuracy differs in {n_diff}; "
              f"confusion matrix differs in {len(cm_changed)}")
        print(f"  mean acc old={arr[:, 0].mean():.4f} new={arr[:, 1].mean():.4f}")
        order = np.argsort(-np.abs(delta))[:top]
        if n_diff:
            print(f"  largest accuracy changes (run: old -> new):")
            for i in order:
                if abs(delta[i]) > 1e-9:
                    print(f"    run {rows[i][0]:>3}: {rows[i][1]:.4f} -> {rows[i][2]:.4f}  ({delta[i]:+.4f})")

    # Lowest-accuracy runs in the NEW set: outliers that inflate the error-bar means
    if new:
        accs = []
        for run, p in new.items():
            a, _ = _acc_cm(p)
            if a is not None:
                accs.append((run, float(a)))
        accs.sort(key=lambda x: x[1])
        vals = np.array([a for _, a in accs])
        print(f"  NEW accuracy: median={np.median(vals):.4f}  min={vals.min():.4f}  max={vals.max():.4f}")
        print(f"  lowest-accuracy NEW runs: " + ", ".join(f"{r} ({a:.4f})" for r, a in accs[:min(top, 8)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old-dir', help="old _chkpt_finals dir (default: paths['checkpoints_dir'])")
    ap.add_argument('--new-dir', help="new evaluation dir (default: paths['eval_base_dir'])")
    ap.add_argument('--cnn', type=int, default=16)
    ap.add_argument('--trf', type=int, default=16)
    ap.add_argument('--top', type=int, default=10, help='rows to list per section')
    args = ap.parse_args()

    if args.old_dir and args.new_dir:
        old_base, new_base = Path(args.old_dir), Path(args.new_dir)
    else:
        from src.utils.config_loader import load_config
        paths = load_config()['paths']
        old_base = Path(args.old_dir or paths['checkpoints_dir'])
        new_base = Path(args.new_dir or paths['eval_base_dir'])
    print(f"old: {old_base}\nnew: {new_base}")

    tasks = ['MetabolicState_2', 'State_Modality_6', 'State_Modality_Valence_16']
    for task in tasks:
        compare_task(task, old_base / task, new_base / task, args.cnn, args.trf, args.top)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
