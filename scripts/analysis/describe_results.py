"""
Print the structure of the aggregated results file (what load_or_build_all_results()
returns / build_results.py writes): which tasks and fields it holds, their shapes,
and how much of the file each part takes.

Usage (from repo root):
    python -m scripts.analysis.describe_results                 # the configured results file
    python -m scripts.analysis.describe_results path/to/aggregated_results.pkl
"""

import pickle
import sys

import numpy as np


def _size(obj):
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def _mb(n):
    return f'{n / 1e6:6.2f} MB' if n >= 1e4 else f'{n / 1e3:6.1f} KB'


def _what(v):
    if isinstance(v, np.ndarray):
        return f'array {v.shape} {v.dtype}'
    if isinstance(v, (list, tuple)):
        return f'{type(v).__name__}[{len(v)}]'
    if isinstance(v, dict):
        return f'dict[{len(v)}]'
    return type(v).__name__ if v is None else f'{type(v).__name__} = {v!r}'[:60]


def _fields(d, indent):
    for k, v in d.items():
        print(f'{indent}{k:<28} {_what(v):<34} {_mb(_size(v))}')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        from src.utils.config_loader import load_config
        path = load_config()['paths']['eval_results_path']
    with open(path, 'rb') as f:
        results = pickle.load(f)

    import os
    print(f'{path}\n  file size {os.path.getsize(path) / 1e6:.2f} MB, {len(results)} tasks\n')
    for task, e in results.items():
        print(f"== {task}   (layout version {e.get('__version__')}, {_mb(_size(e)).strip()})")
        print(f"  best run: {e.get('best_run_id')}   top 5 by validation accuracy:")
        for r in e.get('best_ranking', []):
            print(f"    {r['run_id']:<18} val_acc_best={r['val_acc_best']:.4f}  test accuracy={r['accuracy']:.4f}")
        for part in ('control', 'best'):
            d = e.get(part)
            print(f"  {part}: {'-' if d is None else _mb(_size(d)).strip()}  (one full evaluation result)")
            if d:
                _fields(d, '    ')
        print('  runs (grouped by transformer size; E16 = the same runs seen by CNN size):')
        for grp, recs in e['runs'].items():
            print(f'    {grp:<5} {len(recs):>3} runs   {_mb(_size(recs)).strip()}')
        canon = e['runs'].get('H16', [])
        if canon:
            print('  fields of one H16 run record:')
            _fields(canon[0], '    ')
            print('    (curves holds:', ', '.join(f'{k} {v.shape}' for k, v in canon[0].get('curves', {}).items()) + ')')
        others = [k for k in e if k not in ('control', 'best', 'runs', '__version__', 'best_ranking')
                  and not k.startswith('best_')]
        if others:
            print('  other fields:', ', '.join(others))
        print()


if __name__ == '__main__':
    main()
