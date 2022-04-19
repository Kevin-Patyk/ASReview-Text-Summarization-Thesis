"""Script to get dataset sorted on the average time to discovery.

Example
-------

> python scripts/get_atd.py

or

> python scripts/get_atd.py -s output/simulation/ptsd/state_files -d data/ptsd.csv -o output/simulation/ptsd/atd_ptsd.csv


Authors
-------
- De Bruin, Jonathan
"""

# version 0.1.1+31.g1dde98a

import argparse
from pathlib import Path

import pandas as pd
from asreview.analysis import Analysis
from asreview import ASReviewData


def get_atd_from_states(fp_states, as_data):
    """Merge states descriptives and get ATD."""
    a = Analysis.from_path(fp_states)

    atd = pd.DataFrame(
        a.avg_time_to_discovery().items(),
        columns=["idx", "atd"]).set_index("idx")

    # this part of the code is bit hacky because asreview
    # analysis with row numbers and as_data sometimes not.
    # to be fixed in the future when entire workflow works
    # with row numbers.
    result = as_data.df.iloc[atd.index].copy()
    result["atd"] = atd["atd"].values
    return result.sort_values("atd")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute ATD from table."
    )
    parser.add_argument(
        "-s",
        type=str,
        help="States location")
    parser.add_argument(
        "-d",
        type=str,
        help="Dataset location")
    parser.add_argument(
        "-o",
        type=str,
        help="Dataset location")
    args = parser.parse_args()

    # merge results
    as_data = ASReviewData.from_file(args.d)
    result = get_atd_from_states(args.s, as_data)

    # store result in output folder
    Path(args.o).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(args.o))