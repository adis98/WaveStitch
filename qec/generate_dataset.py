"""Stable QEC entrypoint for the main application's WaveStitch submodule.

The application passes a materialized CSV/Parquet source plus a target-column
selection. All remaining columns are hierarchical/conditional. The wrapper
normalizes the source to CSV and invokes training + generation with one shared
model path and one shared set of WaveStitch hyperparameters.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REQUIRED_HIERARCHY = ("shot_id", "round_id")


def run(cmd: list[str], *, cwd: Path) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "train", "generate"), default="all")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--target-columns", nargs="+", required=True)
    parser.add_argument("--hierarchical-columns", nargs="+", required=True)

    # Kept identical across training and generation so model dimensions/config agree.
    parser.add_argument("--window-size", type=int, default=34)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def load_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported QEC source format {suffix!r}; expected CSV or Parquet")

    unnamed = [column for column in df.columns if str(column).casefold().startswith("unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def validate_partition(
    columns: list[str], target_columns: list[str], hierarchical_columns: list[str]
) -> None:
    if len(columns) != len(set(columns)):
        raise ValueError("Source dataset contains duplicate column names")
    missing_required = [column for column in REQUIRED_HIERARCHY if column not in columns]
    if missing_required:
        raise ValueError(f"QEC source is missing required columns: {missing_required}")

    if not target_columns:
        raise ValueError("At least one target column is required")
    if set(target_columns) & set(REQUIRED_HIERARCHY):
        raise ValueError("shot_id and round_id must be hierarchical, not targets")
    if set(target_columns) & set(hierarchical_columns):
        raise ValueError("Target and hierarchical columns must not overlap")

    expected = set(columns)
    received = set(target_columns) | set(hierarchical_columns)
    if expected != received:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        raise ValueError(
            f"Target/hierarchy partition must cover the source exactly; missing={missing}, extra={extra}"
        )

    # Preserve source order: the model scripts receive hierarchy in the same order
    # used by the backend and by PreprocessorOrdinal.
    expected_hierarchy = [column for column in columns if column not in set(target_columns)]
    if hierarchical_columns != expected_hierarchy:
        raise ValueError(
            "hierarchical_columns must be exactly the source columns not selected as targets"
        )


def main() -> None:
    args = get_args()
    repo = Path(__file__).resolve().parents[1]
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    model_path = Path(args.model_path).resolve()
    workdir = output.parent
    workdir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_source(source)
    columns = [str(column) for column in df.columns]
    validate_partition(columns, args.target_columns, args.hierarchical_columns)

    # Always give the research scripts a local normalized CSV. This lets the main
    # backend materialize either CSV or Parquet without making WaveStitch care about
    # object storage or catalog implementation details.
    normalized_source = workdir / "wavestitch_source.csv"
    df.to_csv(normalized_source, index=False)

    common = [
        "-dataset", "QEC",
        "-qec_file_path", str(normalized_source),
        "-hierarchical_features", *args.hierarchical_columns,
        "-window_size", str(args.window_size),
        "-stride", str(args.stride),
        "-timesteps", str(args.timesteps),
        "-batch_size", str(args.batch_size),
    ]

    if args.phase in {"all", "train"}:
        training_script = repo / "training_wavestitch_ordinal.py"
        run([
            sys.executable,
            str(training_script),
            *common,
            "-epochs", str(args.epochs),
            "-model_path", str(model_path),
        ], cwd=repo)

    if args.phase in {"all", "generate"}:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"WaveStitch model does not exist at {model_path}; run the train phase first"
            )
        generation_script = repo / "simulate_qec_wavestitch_pipeline_strided_preconditioning_ordinal.py"
        run([
            sys.executable,
            str(generation_script),
            *common,
            "-n_trials", "1",
            "-model_path", str(model_path),
            "-output_path", str(output),
        ], cwd=repo)

        if not output.is_file():
            raise RuntimeError(f"WaveStitch generation did not create {output}")


if __name__ == "__main__":
    main()
