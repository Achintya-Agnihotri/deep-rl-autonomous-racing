import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from stable_baselines3 import PPO, SAC
from racing_env import RacingEnv


MODEL_CONFIG = [
    ("SAC v2", "SAC", "models/sac_racing_agent_v2.zip"),
    ("SAC v3", "SAC", "models/sac_racing_agent_v3.zip"),
    ("PPO v2", "PPO", "models/ppo_racing_agent_v2.zip"),
]

TRACKS = ["A", "B"]
RESULTS_DIR = Path("results")


def load_model(algorithm, path):
    model_class = SAC if algorithm.upper() == "SAC" else PPO
    return model_class.load(path)


def run_episode(
    model,
    model_name,
    algorithm,
    model_path,
    track_name,
    episode,
    seed,
    max_steps,
):
    env = RacingEnv(
        render_mode=None,
        max_steps=max_steps,
        track_name=track_name,
    )

    try:
        observation, _ = env.reset(seed=seed)

        total_reward = 0.0
        total_progress = 0.0
        steps = 0
        info = {}

        while True:
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += float(reward)
            total_progress += float(
                info.get("progress_delta", 0.0)
            )

            steps += 1

            if terminated or truncated:
                break

        lap_times = info.get("completed_lap_times", [])

        return {
            "model": model_name,
            "algorithm": algorithm,
            "model_path": model_path,
            "track": track_name,
            "episode": episode,
            "seed": seed,
            "steps": steps,
            "total_reward": total_reward,
            "total_progress": total_progress,
            "completed_laps": int(info.get("lap_count", 0)),
            "crashed": bool(info.get("crashed", False)),
            "fastest_lap_seconds": (
                min(lap_times) if lap_times else ""
            ),
            "mean_lap_seconds": (
                statistics.mean(lap_times)
                if lap_times else ""
            ),
            "lap_times_seconds": json.dumps(
                [round(float(x), 4) for x in lap_times]
            ),
        }

    finally:
        env.close()


def summarize_results(rows):
    groups = {}

    for row in rows:
        key = (
            row["model"],
            row["algorithm"],
            row["track"],
        )
        groups.setdefault(key, []).append(row)

    summaries = []

    for (model, algorithm, track), group in groups.items():

        lap_times = []

        for row in group:
            if row["lap_times_seconds"]:
                lap_times.extend(
                    json.loads(row["lap_times_seconds"])
                )

        episode_count = len(group)

        completed_laps = sum(
            int(row["completed_laps"])
            for row in group
        )

        crashed_episodes = sum(
            bool(row["crashed"])
            for row in group
        )

        lap_counts = [
            int(row["completed_laps"])
            for row in group
        ]

        summaries.append({
            "model": model,
            "algorithm": algorithm,
            "track": track,
            "episodes": episode_count,

            "crash_rate_percent": (
                100.0 * crashed_episodes / episode_count
            ),

            "episodes_without_crash": (
                episode_count - crashed_episodes
            ),

            "total_completed_laps": completed_laps,

            "mean_laps_per_episode": (
                statistics.mean(lap_counts)
            ),

            "average_episode_reward": statistics.mean(
                float(row["total_reward"])
                for row in group
            ),

            "average_progress_per_episode": statistics.mean(
                float(row["total_progress"])
                for row in group
            ),

            "mean_episode_steps": statistics.mean(
                int(row["steps"])
                for row in group
            ),

            "fastest_lap_seconds": (
                min(lap_times) if lap_times else ""
            ),

            "average_lap_seconds": (
                statistics.mean(lap_times)
                if lap_times else ""
            ),

            "lap_time_std_seconds": (
                statistics.stdev(lap_times)
                if len(lap_times) > 1 else ""
            ),
        })

    return summaries


def main():

    parser = argparse.ArgumentParser(
        description="Benchmark SAC and PPO racing agents."
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    episode_results = []
    started = time.time()

    for model_name, algorithm, model_path in MODEL_CONFIG:

        if not Path(model_path).is_file():
            print(f"Skipping missing model: {model_path}")
            continue

        print(f"\nLoading {model_name}")

        model = load_model(algorithm, model_path)

        for track_name in TRACKS:

            print(
                f"\n========== {model_name} "
                f"| TRACK {track_name} =========="
            )

            for episode in range(1, args.episodes + 1):

                row = run_episode(
                    model=model,
                    model_name=model_name,
                    algorithm=algorithm,
                    model_path=model_path,
                    track_name=track_name,
                    episode=episode,
                    seed=episode - 1,
                    max_steps=args.max_steps,
                )

                episode_results.append(row)

                fastest = row["fastest_lap_seconds"]

                fastest_text = (
                    f"{float(fastest):.2f}s"
                    if fastest != ""
                    else "N/A"
                )

                print(
                    f"Episode {episode}/{args.episodes} | "
                    f"Laps: {row['completed_laps']} | "
                    f"Crash: {row['crashed']} | "
                    f"Fastest: {fastest_text} | "
                    f"Reward: {row['total_reward']:.2f} | "
                    f"Steps: {row['steps']}"
                )

    if not episode_results:
        raise SystemExit(
            "No models were evaluated. Check model paths."
        )

    # Save individual episode results.
    episode_file = RESULTS_DIR / "episode_results.csv"

    with episode_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(episode_results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(episode_results)

    # Calculate aggregate results.
    summary_results = summarize_results(
        episode_results
    )

    summary_file = RESULTS_DIR / "benchmark_summary.csv"

    with summary_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary_results)

    # Save experiment metadata.
    metadata = {
        "evaluation_type": "deterministic",
        "episodes_per_model_track": args.episodes,
        "max_steps": args.max_steps,
        "tracks": TRACKS,
        "models_evaluated": [
            {
                "name": name,
                "algorithm": algorithm,
                "path": path,
            }
            for name, algorithm, path in MODEL_CONFIG
            if Path(path).is_file()
        ],
        "elapsed_seconds": round(
            time.time() - started, 2
        ),
        "note": (
            "Results are deterministic evaluation results, "
            "not training rewards. All models use the same "
            "episode seed schedule."
        ),
    }

    metadata_file = (
        RESULTS_DIR / "benchmark_metadata.json"
    )

    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Print final summary.
    print("\n========== FINAL BENCHMARK ==========")

    for row in summary_results:

        lap = (
            f"{row['average_lap_seconds']:.2f}s"
            if row["average_lap_seconds"] != ""
            else "N/A"
        )

        print(
            f"{row['model']} | Track {row['track']} | "
            f"Average lap: {lap} | "
            f"Crash rate: {row['crash_rate_percent']:.1f}% | "
            f"Completed laps: {row['total_completed_laps']} | "
            f"Reward: {row['average_episode_reward']:.2f}"
        )

    print("\nResults saved successfully:")
    print(episode_file)
    print(summary_file)
    print(metadata_file)


if __name__ == "__main__":
    main()