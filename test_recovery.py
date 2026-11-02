
from stable_baselines3 import SAC
from racing_env import RacingEnv

model = SAC.load("sac_racing_agent_progress.zip")
env = RacingEnv(max_steps=5000)

try:
    obs, _ = env.reset(seed=0)

    # First, reproduce the model's normal behavior.
    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if info["speed"] == 0:
            print(f"Stopped at step {step}")
            break

    print("\nTesting recovery from the stopped state:")

    # Apply controlled positive throttle while keeping steering
    # at the model's current predicted steering value.
    for test_step in range(10):
        action, _ = model.predict(obs, deterministic=True)
        action = action.copy()
        action[1] = 0.8

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"Test={test_step:2d} | "
            f"steer={action[0]:+.3f} | "
            f"throttle={action[1]:+.3f} | "
            f"speed={info['speed']:.2f} | "
            f"progress={info['progress_delta']:+.4f} | "
            f"crashed={info['crashed']}"
        )

        if terminated or truncated:
            print("Episode ended.")
            break

finally:
    env.close()
