
import argparse
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO, SAC

from racing_env import RacingEnv


DEFAULT_MODEL = "sac_racing_agent_v2.zip"
MAX_STEPS = 5000


def evaluate_model(
    model_path,
    episodes=10,
    render=False,
    track_name="A",
    algorithm="SAC",
):
    algorithm = algorithm.upper()
    track_name = track_name.upper()

    if algorithm == "SAC":
        model_class = SAC
    elif algorithm == "PPO":
        model_class = PPO
    else:
        raise ValueError("algorithm must be 'SAC' or 'PPO'")

    model = model_class.load(model_path)

    env = RacingEnv(
        render_mode="human" if render else None,
        max_steps=MAX_STEPS,
        track_name=track_name,
    )

    records = []

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=episode)

            total_reward = 0.0
            total_progress = 0.0
            steps = 0
            info = {}

            while True:
                if render:
                    env.render()

                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )
                if steps >= 950 and steps <= 1550 and steps % 50 == 0:
                    print("Observation:", np.round(observation, 3))
                    print("Predicted action:", np.round(action, 3))

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)
                if steps >= 1000 and steps <= 1600 and steps % 50 == 0:
                    print(
                        f"Step={steps}, "
                        f"steering={action[0]:.3f}, "
                        f"throttle={action[1]:.3f}, "
                        f"speed={info['speed']:.2f}, "
                        f"progress_delta={info['progress_delta']:.4f}, "
                        f"progress_index={info['progress_index']}"
                    )
                total_reward += float(reward)
                total_progress += float(
                    info.get("progress_delta", 0.0)
                )

                # Print the first 30 steps, then every 500 steps.
                if steps < 30 or (steps + 1) % 500 == 0:
                    print(
                        f"Step={steps + 1}, "
                        f"action={action}, "
                        f"speed={info['speed']:.2f}, "
                        f"progress_delta="
                        f"{info['progress_delta']:.4f}, "
                        f"progress_index="
                        f"{info['progress_index']}, "
                        f"reward={reward:.4f}, "
                        f"crashed={info['crashed']}"
                    )

                steps += 1

                if terminated or truncated:
                    break

            lap_times = info.get("completed_lap_times", [])

            record = {
                "episode": episode + 1,
                "steps": steps,
                "reward": total_reward,
                "progress": total_progress,
                "crashed": bool(info.get("crashed", False)),
                "laps": int(info.get("lap_count", 0)),
                "lap_times": list(lap_times),
                "fastest_lap": min(lap_times) if lap_times else None,
            }

            records.append(record)

            fastest = (
                f"{record['fastest_lap']:.2f}s"
                if record["fastest_lap"] is not None
                else "N/A"
            )

            print(
                f"\n{Path(model_path).name} | "
                f"Episode {episode + 1}/{episodes} | "
                f"Steps: {steps} | "
                f"Laps: {record['laps']} | "
                f"Fastest lap: {fastest} | "
                f"Reward: {total_reward:.2f} | "
                f"Progress: {total_progress:.2f} | "
                f"Crash: {record['crashed']}"
            )

            if lap_times:
                print(
                    "Lap times: "
                    + ", ".join(f"{t:.2f}s" for t in lap_times)
                )

    finally:
        env.close()

    if not records:
        print("No episodes were evaluated.")
        return records

    all_laps = [
        lap_time
        for record in records
        for lap_time in record["lap_times"]
    ]

    crash_rate = sum(
        record["crashed"] for record in records
    ) / len(records)

    completed_laps = sum(
        record["laps"] for record in records
    )

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Model: {Path(model_path).name}")
    print(f"Algorithm: {algorithm}")
    print(f"Track: {track_name}")
    print(f"Episodes: {episodes}")
    print(f"Crash rate: {crash_rate:.1%}")
    print(f"Total completed laps: {completed_laps}")
    print(
        "Average episode reward: "
        f"{sum(r['reward'] for r in records) / len(records):.2f}"
    )
    print(
        "Average progress per episode: "
        f"{sum(r['progress'] for r in records) / len(records):.2f}"
    )

    if all_laps:
        print(f"Fastest lap: {min(all_laps):.2f} seconds")
        print(
            "Average lap time: "
            f"{sum(all_laps) / len(all_laps):.2f} seconds"
        )
    else:
        print("No complete laps were detected.")

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an RL racing model."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_MODEL],
        help="One or more model ZIP files",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes per model",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the Pygame simulation",
    )

    parser.add_argument(
        "--track",
        choices=["A", "B", "a", "b"],
        default="A",
        help="Track to evaluate on",
    )

    parser.add_argument(
        "--algorithm",
        choices=["SAC", "PPO", "sac", "ppo"],
        default="SAC",
        help="Algorithm used to train the model",
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    for model_path in args.models:
        if not Path(model_path).is_file():
            parser.error(f"Model file not found: {model_path}")

    for model_path in args.models:
        evaluate_model(
            model_path=model_path,
            episodes=args.episodes,
            render=args.render,
            track_name=args.track,
            algorithm=args.algorithm,
        )


if __name__ == "__main__":
    main()
