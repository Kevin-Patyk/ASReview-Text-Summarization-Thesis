"""Script to flatten state file.

Example
-------

> python scripts/merge_descriptives.py

or

> python scripts/merge_descriptives.py -s output/simulation/*/descriptives/data_stats_*.json

or

> python scripts/merge_descriptives.py -o output/my_table.json

Authors
-------
- De Bruin, Jonathan
"""

# version 0.1.1+31.g1dde98a

import argparse
import glob
from pathlib import Path

import pandas as pd


def create_table_descriptives(datasets):
    """Merge dataset descriptives."""
    df = pd.concat(
        [pd.read_json(ds, orient="index") for ds in datasets],
        axis=0
    )

    df.index.name = "dataset_name"
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Merge descriptives of multiple files into single table."
    )
    parser.add_argument(
        "-s",
        type=str,
        nargs="*",
        default="output/simulation/*/descriptives/data_stats_*.json",
        help="Datasets location")
    parser.add_argument(
        "-o",
        type=str,
        default="output/tables/data_descriptives.csv",
        help="Output table location")
    args = parser.parse_args()

    # load datasets
    datasets = glob.glob(args.s)
    print("Datasets found:")
    for dataset in datasets:
        print(dataset)

    # merge results
    result = create_table_descriptives(datasets)

    # store result in output folder
    Path(args.o).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(args.o))
    result.to_excel(Path(args.o).with_suffix('.xlsx'))