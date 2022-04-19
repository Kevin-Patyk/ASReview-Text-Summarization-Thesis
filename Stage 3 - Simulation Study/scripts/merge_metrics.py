"""Script to flatten state file.

Example
-------

> python scripts/merge_metrics.py

or

> python scripts/merge_metrics.py -s output/simulation/*/simulation_metrics_*.json

or

> python scripts/merge_metrics.py -o output/my_table.json

Authors
-------
- De Bruin, Jonathan
"""

# version 0.1.1+31.g1dde98a

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def flatten_state_metrics(file, metrics):

    return {
        "dataset_name": file,
        "model": metrics["settings"]["model"],
        "query_strategy": metrics["settings"]["query_strategy"],
        "balance_strategy": metrics["settings"]["balance_strategy"],
        "feature_extraction": metrics["settings"]["feature_extraction"],
        "n_instances": metrics["settings"]["n_instances"],
        "wss95": metrics["wss"]["95"],
        "wss100": metrics["wss"]["100"],
        "rrf5": metrics["rrf"]["5"],
        "rrf10": metrics["rrf"]["10"],
        "loss": metrics["loss"],
        # "n_queries": metrics["general"]["n_queries"],
        "n_states": metrics["general"]["n_states"],
    }


def create_table_state_matrics(states):
    """Merge state descriptives."""

    metrics = []

    for s in states:
        with open(s, "r") as f:
            metrics_s = json.load(f)
            for file, m in metrics_s.items():
                res = flatten_state_metrics(file, m)
                metrics.append(res)

    # flatten state files
    df = pd.DataFrame(metrics).set_index("dataset_name")

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Merge metrics of multiple states into single table."
    )
    parser.add_argument(
        "-s",
        type=str,
        default="output/simulation/*/simulation_metrics_*.json",
        help="states location")
    parser.add_argument(
        "-o",
        type=str,
        default="output/tables/data_metrics.csv",
        help="Output table location")
    args = parser.parse_args()

    # load states
    states = glob.glob(args.s)
    print("states found:")
    for state in states:
        print(state)

    # merge results
    result = create_table_state_matrics(states)

    # store result in output folder
    Path(args.o).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(args.o))
    result.to_excel(Path(args.o).with_suffix('.xlsx'))