
from racing_env import RacingEnv
import numpy as np

env = RacingEnv(render_mode=None, max_steps=200, track_name="A")

try:
    obs, info = env.reset(seed=42)

    print("Initial state")
    print(f"Speed: {info['speed']:.2f}")
    print(f"Progress index: {info['progress_index']}")

    print("\nTest 1: Accelerate with straight steering")
    for step in range(60):
        action = np.array([0.0, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 10 == 0 or terminated or truncated:
            print(
                f"Step={step + 1:3d} | "
                f"Speed={info['speed']:7.2f} | "
                f"Progress={info['progress_index']:3d} | "
                f"Delta={info['progress_delta']:8.4f} | "
                f"Crash={info['crashed']}"
            )

        if terminated or truncated:
            print("Episode ended.")
            break

    print("\nTest 2: Apply neutral throttle")
    for step in range(30):
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 5 == 0:
            print(
                f"Step={step + 1:3d} | "
                f"Speed={info['speed']:7.2f} | "
                f"Throttle={info['throttle_brake']:.3f} | "
                f"Reward={reward:.4f}"
            )

        if terminated or truncated:
            break

finally:
    env.close()
