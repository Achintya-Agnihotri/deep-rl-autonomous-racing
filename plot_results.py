import csv
from pathlib import Path

import matplotlib.pyplot as plt


SUMMARY_FILE = Path("results/benchmark_summary.csv")
OUTPUT_DIR = Path("results/plots")


def read_results():

    if not SUMMARY_FILE.exists():
        raise SystemExit(
            "Benchmark summary not found. "
            "Run benchmark.py first."
        )

    with SUMMARY_FILE.open(
        newline="",
        encoding="utf-8",
    ) as file:

        return list(csv.DictReader(file))


def create_bar_chart(
    rows,
    track,
    field,
    ylabel,
    title,
    filename,
):

    selected = [
        row for row in rows
        if row["track"] == track
        and row[field] not in ("", None)
    ]

    if not selected:
        print(f"No data available for {title}")
        return

    labels = [
        row["model"] for row in selected
    ]

    values = [
        float(row[field]) for row in selected
    ]

    plt.figure(figsize=(9, 5))

    bars = plt.bar(labels, values)

    plt.ylabel(ylabel)
    plt.title(f"{title} - Track {track}")

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=180,
    )

    plt.close()

    print(f"Saved {filename}")


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_results()

    metrics = [
        (
            "average_lap_seconds",
            "Seconds",
            "Average Lap Time",
            "average_lap_time",
        ),
        (
            "fastest_lap_seconds",
            "Seconds",
            "Fastest Lap Time",
            "fastest_lap",
        ),
        (
            "crash_rate_percent",
            "Crash Rate (%)",
            "Crash Rate",
            "crash_rate",
        ),
        (
            "total_completed_laps",
            "Completed Laps",
            "Total Completed Laps",
            "completed_laps",
        ),
        (
            "average_episode_reward",
            "Reward",
            "Average Evaluation Reward",
            "average_reward",
        ),
    ]

    for track in ["A", "B"]:

        for (
            field,
            ylabel,
            title,
            filename,
        ) in metrics:

            create_bar_chart(
                rows=rows,
                track=track,
                field=field,
                ylabel=ylabel,
                title=title,
                filename=(
                    f"track_{track.lower()}_{filename}.png"
                ),
            )

    print("\nAll available graphs generated.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()