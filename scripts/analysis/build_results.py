"""
Builds (or rebuilds) the aggregated evaluation results
(paths['eval_results_path']) that every figure/analysis script reads via
load_or_build_all_results() -- previously this only ever got built as a
side effect of running one of those scripts, which meant "just building
the results" required running an unrelated figure. This script does only
that, standalone.

Usage (from repo root):
    python -m scripts.analysis.build_results
    RESULTS_RECOMPUTE=true python -m scripts.analysis.build_results  # force rebuild
"""

import os

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.utils.helpers import load_or_build_all_results, get_style


def main():
    logger = setup_logger(task_name='build_results', log_dir='logs/build_results')
    config = load_config()
    paths = config['paths']

    base_results_dir = paths['eval_base_dir']
    logger.info(f"eval_base_dir: {base_results_dir}")
    logger.info(f"eval_results_path: {paths['eval_results_path']}")

    _, task_class_names, *_ = get_style(style="styles")

    recompute = os.environ.get('RESULTS_RECOMPUTE', '').lower() in ('true', '1', 't')
    if recompute:
        logger.info("RESULTS_RECOMPUTE set -- forcing a fresh rebuild.")

    results_dict = load_or_build_all_results(
        paths['eval_results_path'], base_results_dir, task_class_names,
        fixed_trf_for_E=16, fixed_cnn_for_H=16,
        only_cnn_dim=16, recompute=recompute, logger=logger,
    )

    for task_name, task_data in results_dict.items():
        logger.info(
            f"Task '{task_name}': control={task_data.get('control') is not None}, "
            f"best={task_data.get('best') is not None}, "
            f"run groups={list(task_data.get('runs', {}).keys())}"
        )

    logger.info("Done.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
