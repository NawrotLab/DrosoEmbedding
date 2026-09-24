"""
Lightweight, dependency-free resource reporting for figure/results scripts --
wall time and peak memory, printed/logged at the end of a block. Stdlib only
(time + resource), so it costs nothing to have on by default everywhere.

Exists so every run of a Level-1 ("load stored results") or Level-2
("compute results from raw/preprocessed data") code path leaves behind a
real, measured number -- for the reproducibility docs' time/storage/compute
table -- instead of a guess written once and never re-checked.

Usage:
    from src.utils.resource_log import report
    with report('Fig 4 (load path)', logger):
        ...
"""

import resource
import sys
import time
from contextlib import contextmanager


def peak_memory_mb() -> float:
    """Peak resident set size of this process so far, in MB. Unix only
    (the stdlib resource module isn't available on Windows). ru_maxrss is
    bytes on macOS, KB on Linux -- normalised to MB either way."""
    kb_or_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == 'darwin' else 1024
    return kb_or_bytes / divisor


@contextmanager
def report(label: str, logger=None):
    """Times the wrapped block and reports wall time + peak memory (of the
    whole process, not just this block -- there's no cheap stdlib way to
    scope memory to a block, only a high-water mark) when it exits, even on
    an exception. Logs via logger.info if given, else print()."""
    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        _log(f'[resource] {label}: wall time {elapsed:.1f}s, peak memory so far {peak_memory_mb():.0f} MB')
