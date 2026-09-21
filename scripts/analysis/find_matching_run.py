"""
Read-only diagnostic: compare the pkls that fed the submitted Fig 2 (the old
<old-dir>/<task>/best/ and <old-dir>/<task>/control*/ files) against the current
evaluation/<task>/{run_id}_evalResults.pkl of the same run id.

For each old pkl it reports:
  - file date (which perplexity regime, if any, it was evaluated under)
  - confusion matrix / accuracy / test-sample count old vs new
  - test-sample order (latent_labels): same order, same counts but reordered, or different
  - latent-space difference (only meaningful if the order is the same)
  - t-SNE: identical or not; and order-proof similarity measures:
      * Procrustes disparity (same order only; 0 = identical up to rotation/scale/flip)
      * kNN label purity in the 2D embedding, old vs new (same structure => similar values)
      * rank correlation of the class-centroid distance matrices, old vs new
  - optional side-by-side PNG of old vs new t-SNE per pkl (--plot-dir)

Optional:
  --scan      also search ALL current pkls of the task for an exact confusion-matrix
              match (slow: loads every pkl)
  --retsne    recompute t-SNE on the OLD latent at each --perplexities (default 30 500)
              and compare it to the t-SNE stored in the old pkl (slow: minutes per pkl);
              the perplexity that reproduces the stored layout is the one it was made with

Usage (from repo root, on a machine that has both trees):
    python -m scripts.analysis.find_matching_run --old-dir <path>/_chkpt_finals \\
        --new-dir <path>/evaluation --plot-dir <somewhere>/tsne_compare
"""

import argparse
import datetime
import os
import pickle
from pathlib import Path

import numpy as np

TASKS = ['MetabolicState_2', 'State_Modality_6', 'State_Modality_Valence_16']


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _old_pkls(task_dir: Path):
    out = []
    subs = sorted(p for p in task_dir.iterdir() if p.is_dir()) if task_dir.is_dir() else []
    for sub in subs:
        if sub.name == 'best' or sub.name.startswith('control'):
            out += [(sub.name, p) for p in sorted(sub.glob('*.pkl'))]
    return out


# ── t-SNE similarity measures ────────────────────────────────────────────────

def _knn_purity(emb, labels, k=15):
    """Mean fraction of each point's k nearest 2D neighbours sharing its label."""
    from sklearn.neighbors import NearestNeighbors
    k = min(k, len(emb) - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(emb, return_distance=False)[:, 1:]
    return float((labels[idx] == labels[:, None]).mean())


def _centroid_corr(a, la, b, lb):
    """Spearman correlation of class-centroid pairwise distances in two embeddings
    (invariant to rotation/reflection and to sample order)."""
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    classes = sorted(set(la.tolist()) & set(lb.tolist()))
    if len(classes) < 3:
        return float('nan')
    ca = np.array([a[la == c].mean(axis=0) for c in classes])
    cb = np.array([b[lb == c].mean(axis=0) for c in classes])
    return float(spearmanr(pdist(ca), pdist(cb)).correlation)


def _dist_corr(a, b, k=500):
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    idx = np.random.default_rng(0).choice(len(a), min(k, len(a)), replace=False)
    return float(spearmanr(pdist(a[idx]), pdist(b[idx])).correlation)


def _plot_pair(path, old_t, old_l, new_t, new_l, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    classes = sorted(set(old_l.tolist()) | set(new_l.tolist()))
    cmap = plt.get_cmap('tab20', max(len(classes), 2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, t, l, name in ((axes[0], old_t, old_l, 'OLD (submitted)'), (axes[1], new_t, new_l, 'NEW')):
        colours = [cmap(classes.index(c)) for c in l.tolist()]
        ax.scatter(t[:, 0], t[:, 1], c=colours, s=3, linewidths=0)
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── per-pkl comparison ───────────────────────────────────────────────────────

def compare_pair(folder, old_path, new_path, task, args):
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(old_path)).strftime('%Y-%m-%d')
    old = _load(old_path)
    print(f"\n  OLD {folder}/{old_path.name}  (file date {mtime})")
    if new_path is None:
        print(f"    no current pkl with the same run id ({old_path.stem}) -- "
              f"the submitted model is not in the current sweep under that name")
        return
    new = _load(new_path)

    ocm, ncm = old.get('confusion_matrix'), new.get('confusion_matrix')
    oacc, nacc = old.get('accuracy'), new.get('accuracy')
    if ocm is not None and ncm is not None and np.shape(ocm) == np.shape(ncm):
        d = int(np.abs(np.array(ocm) - np.array(ncm)).sum())
        cm_txt = 'IDENTICAL' if d == 0 else f'differs (|dCM|={d})'
    else:
        cm_txt = 'not comparable'
    n_old = int(np.sum(ocm)) if ocm is not None else None
    n_new = int(np.sum(ncm)) if ncm is not None else None
    print(f"    accuracy old={oacc} new={nacc}; test samples old={n_old} new={n_new}; confusion matrix {cm_txt}")

    ol, nl = old.get('latent_labels'), new.get('latent_labels')
    oz, nz = old.get('transformer_latent_space'), new.get('transformer_latent_space')
    ot, nt = old.get('tsne_2d'), new.get('tsne_2d')
    if ol is None or nl is None:
        print("    latent_labels missing in one of the pkls")
        return
    ol, nl = np.asarray(ol), np.asarray(nl)
    if ol.shape != nl.shape:
        print(f"    test-set size differs: {ol.shape} vs {nl.shape}")
        return
    if np.array_equal(ol, nl):
        order = 'same'
    elif np.array_equal(np.sort(ol), np.sort(nl)):
        order = 'REORDERED (same label counts, different sample order)'
    else:
        order = 'DIFFERENT label counts'
    print(f"    test-sample order: {order}")

    if oz is not None and nz is not None and order == 'same' and np.shape(oz) == np.shape(nz):
        diff = np.abs(np.asarray(oz) - np.asarray(nz))
        print(f"    latent space: max|diff|={diff.max():.3g}, mean|diff|={diff.mean():.3g} "
              f"(mean |latent|={np.abs(oz).mean():.3g})")
    elif oz is not None and nz is not None:
        print("    latent space: not comparable row-by-row (order/shape differs)")

    if ot is None or nt is None:
        print("    tsne_2d missing in one of the pkls")
        return
    ot, nt = np.asarray(ot), np.asarray(nt)
    identical = ot.shape == nt.shape and np.allclose(ot, nt, atol=1e-3)
    print(f"    stored t-SNE identical: {identical}")
    if order == 'same' and ot.shape == nt.shape:
        from scipy.spatial import procrustes
        _, _, disp = procrustes(ot, nt)
        print(f"    Procrustes disparity (0 = same layout up to rotation/flip/scale): {disp:.4f}; "
              f"pairwise-distance rank corr: {_dist_corr(ot, nt):.3f}")
    print(f"    kNN label purity in 2D (k=15): old={_knn_purity(ot, ol):.3f} new={_knn_purity(nt, nl):.3f}")
    print(f"    class-centroid distance rank corr old vs new: {_centroid_corr(ot, ol, nt, nl):.3f}")

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        out = Path(args.plot_dir) / f"{task}__{folder}__{old_path.stem}.png"
        _plot_pair(out, ot, ol, nt, nl, f"{task} / {folder} / {old_path.stem}")
        print(f"    saved side-by-side plot: {out}")

    if args.retsne and oz is not None:
        from sklearn.manifold import TSNE
        oz = np.asarray(oz)
        for perp in args.perplexities:
            # Same call as src.utils.helpers.compute_tsne (kept local so this script
            # doesn't import torch, which can clash with sklearn's OpenMP on some machines).
            r = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(oz)
            print(f"    re-run t-SNE on OLD latent, perplexity={perp:g}: equals stored old t-SNE: "
                  f"{np.allclose(r, ot, atol=1e-3)}; distance rank-corr with stored: {_dist_corr(r, ot):.3f}")


def scan_all(old_path, task, new_base, top):
    """Exact confusion-matrix match against every current pkl of the task."""
    ocm = _load(old_path).get('confusion_matrix')
    if ocm is None:
        return
    ocm = np.array(ocm)
    scored = []
    for p in sorted((new_base / task).glob('*_evalResults.pkl')):
        cm = _load(p).get('confusion_matrix')
        if cm is None or np.shape(cm) != ocm.shape:
            continue
        scored.append((int(np.abs(np.array(cm) - ocm).sum()), p.name[:-len('_evalResults.pkl')]))
    scored.sort()
    if scored and scored[0][0] == 0:
        print(f"    scan: EXACT confusion-matrix match in current sweep: "
              f"{', '.join(r for d, r in scored if d == 0)}")
    else:
        print("    scan: no exact match in the current sweep; closest (|dCM|, run_id): "
              + ", ".join(f"({d}, {r})" for d, r in scored[:top]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old-dir', required=True, help='old _chkpt_finals dir')
    ap.add_argument('--new-dir', required=True, help='current evaluation dir')
    ap.add_argument('--plot-dir', help='write a side-by-side old/new t-SNE PNG per pkl here')
    ap.add_argument('--scan', action='store_true', help='also search all current pkls for an exact CM match (slow)')
    ap.add_argument('--retsne', action='store_true', help='recompute t-SNE on the old latent (slow)')
    ap.add_argument('--perplexities', type=float, nargs='+', default=[30, 500])
    ap.add_argument('--top', type=int, default=3)
    args = ap.parse_args()
    old_base, new_base = Path(args.old_dir), Path(args.new_dir)

    for task in TASKS:
        print(f"\n{'=' * 70}\n{task}\n{'=' * 70}")
        olds = _old_pkls(old_base / task)
        if not olds:
            print(f"  no best/ or control*/ pkls under {old_base / task}")
            continue
        for folder, op in olds:
            newp = new_base / task / f"{op.stem}_evalResults.pkl"
            compare_pair(folder, op, newp if newp.exists() else None, task, args)
            if args.scan:
                scan_all(op, task, new_base, args.top)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
