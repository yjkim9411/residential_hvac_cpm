"""Command-line entry point for fitting one residential HVAC change-point model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from cpm_data import build_cpm_dataset, prepare_ecobee_data, prepare_eplus_data
from cpm_model import fit_cpm, plot_cpm_fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV or parquet time-step input file")
    parser.add_argument("--source", choices=("eplus", "ecobee"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--minimum-r-squared", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "eplus":
        _, day_frames = prepare_eplus_data(args.input, days=args.days)
    else:
        _, day_frames = prepare_ecobee_data(args.input, days=args.days)

    daily = build_cpm_dataset(day_frames, days=args.days)
    plot_data, results, _ = fit_cpm(daily, minimum_r_squared=args.minimum_r_squared)
    daily.to_csv(args.output_dir / "daily_cpm_data.csv", index=False)
    results.to_csv(args.output_dir / "cpm_results.csv")
    plot_cpm_fit(plot_data)
    plt.tight_layout()
    plt.savefig(args.output_dir / "cpm_fit.png", dpi=180)
    print(results)


if __name__ == "__main__":
    main()
