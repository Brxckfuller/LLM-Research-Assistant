from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_FILE = PROJECT_ROOT / "logs" / "benchmark.csv"

TIME_COLUMNS = ["retrieval", "evidence", "generation", "total"]


def main():
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(f"Could not find {BENCHMARK_FILE}")

    df = pd.read_csv(BENCHMARK_FILE)

    print("\nBenchmark Summary")
    print("=" * 60)
    print(f"Total runs: {len(df)}")

    print("\nRuns by mode")
    print("-" * 60)
    print(df["mode"].value_counts().to_string())

    print("\nTiming summary seconds")
    print("-" * 60)

    summary = df[TIME_COLUMNS].agg(["mean", "median", "min", "max", "std"]).T
    summary = summary.round(2)
    print(summary.to_string())

    print("\nMarkdown table for README")
    print("-" * 60)

    print("| Stage | Mean | Median | Min | Max | Std Dev |")
    print("|---|---:|---:|---:|---:|---:|")

    labels = {
        "retrieval": "Retrieval",
        "evidence": "Evidence extraction",
        "generation": "Answer generation",
        "total": "Total response time",
    }

    for col in TIME_COLUMNS:
        row = summary.loc[col]
        print(
            f"| {labels[col]} | "
            f"{row['mean']:.2f}s | "
            f"{row['median']:.2f}s | "
            f"{row['min']:.2f}s | "
            f"{row['max']:.2f}s | "
            f"{row['std']:.2f}s |"
        )

    print("\nAverage by mode")
    print("-" * 60)

    mode_summary = df.groupby("mode")[TIME_COLUMNS].mean().round(2)
    print(mode_summary.to_string())


if __name__ == "__main__":
    main()