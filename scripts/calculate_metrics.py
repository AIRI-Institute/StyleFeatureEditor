import sys
import torch
import random

sys.path =  ['.'] + sys.path

from argparse import ArgumentParser
from pathlib import Path
from metrics.metrics import metrics_registry
from utils.common_utils import setup_seed


setup_seed(777)


def run(test_opts):
    metrics = []
    for metric_name in test_opts.metrics:
        metrics.append(
            metrics_registry[metric_name]()
        )

    out_path = None
    for metric in metrics:
        print("Calculating", metric.get_name())
        if test_opts.metrics_dir != "":
            out_path = Path(test_opts.metrics_dir) / metric.get_name()
            out_path = f"{out_path}.json"
        _, value, _ = metric(
            test_opts.orig_path,
            test_opts.reconstr_path,
            out_path=str(out_path) if out_path else None,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--metrics", nargs="+", help="List of calculated metrics")
    parser.add_argument(
        "--orig_path", type=str, help="Path to directory of original images to evaluate"
    )
    parser.add_argument(
        "--reconstr_path",
        type=str,
        help="Path to directory of reconstructions of images to evaluate",
    )
    parser.add_argument(
        "--batch_size", default=4, type=int, help="Batch size for testing and inference"
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        help="Number of test/inference dataloader workers",
    )
    parser.add_argument(
        "--metrics_dir",
        default="",
        type=str,
        help="Directory to save .json metrics info",
    )

    test_opts = parser.parse_args()
    run(test_opts)
