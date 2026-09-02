"""
Dry-run checker for scripts/cluster/**/*.sh.

Scans every shell script under this directory (recursively, e.g.
preprocessing/, training/, evaluation/, figures/) for `python -m
<module.path>` invocations (active or commented-out — commented lines are
toggle-able options in these scripts, not dead code) and checks that each
referenced module actually exists as a file and parses without a
SyntaxError.

Does NOT run anything, needs no GPU/data/config — just repo-relative file
and syntax checks. Run it whenever a scripts/cluster/**/*.sh or the modules
they reference change.

Usage (from repo root):
    python scripts/cluster/verify_module_refs.py
"""

import ast
import re
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
CLUSTER_DIR = Path(__file__).resolve().parent

# Matches "python -m some.dotted.module" or "python3 -m ..." or
# "/abs/path/to/python -m ...", ignoring any trailing CLI args.
MODULE_REF_RE = re.compile(r'\bpython3?\b(?:\s+\S+)*?\s+-m\s+([A-Za-z_][A-Za-z0-9_.]*)')


def find_module_refs(sh_path: Path) -> list[str]:
    text = sh_path.read_text()
    return MODULE_REF_RE.findall(text)


def check_module(dotted: str) -> tuple[bool, str]:
    """Returns (ok, message)."""
    rel_path = Path(*dotted.split('.')).with_suffix('.py')
    abs_path = REPO_ROOT / rel_path
    if not abs_path.exists():
        return False, f'no such file: {rel_path}'
    try:
        ast.parse(abs_path.read_text())
    except SyntaxError as e:
        return False, f'SyntaxError in {rel_path}: {e}'
    return True, str(rel_path)


def main() -> int:
    sh_files = sorted(CLUSTER_DIR.rglob('*.sh'))
    if not sh_files:
        print(f'No .sh files found under {CLUSTER_DIR}')
        return 1

    # module -> set of scripts (relative path, e.g. "training/run_sweep.sh") that reference it
    refs: dict[str, set] = {}
    for sh_path in sh_files:
        rel_name = str(sh_path.relative_to(CLUSTER_DIR))
        for module in find_module_refs(sh_path):
            refs.setdefault(module, set()).add(rel_name)

    if not refs:
        print('No `python -m ...` references found.')
        return 0

    failures = []
    print(f'Checking {len(refs)} unique module reference(s) from {len(sh_files)} script(s):\n')
    for module in sorted(refs):
        ok, msg = check_module(module)
        status = 'OK  ' if ok else 'FAIL'
        callers = ', '.join(sorted(refs[module]))
        print(f'  [{status}] {module:<55s} ({callers})')
        if not ok:
            print(f'         -> {msg}')
            failures.append((module, refs[module], msg))

    print()
    if failures:
        print(f'{len(failures)} broken reference(s) found:')
        for module, callers, msg in failures:
            print(f'  - {module} (referenced in {", ".join(sorted(callers))}): {msg}')
        return 1

    print('All module references resolve and parse cleanly.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
