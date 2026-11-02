from gymnasium.utils.env_checker import check_env
from racing_env import RacingEnv


def main():
    env = RacingEnv()

    try:
        print("Checking Gymnasium API...")
        check_env(env, skip_render_check=True)
        print("API check passed.")

        print("\nRunning 3 random-action test episodes...")
        for episode in range(3):
            observation, info = env.reset(seed=episode)
            total_reward = 0.0
            crashed = False

            for step in range(1000):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                crashed = info.get("crashed", False)

                if terminated or truncated:
                    break

            print(
                f"Episode {episode + 1}: "
                f"steps={step + 1}, "
                f"reward={total_reward:.2f}, "
                f"crashed={crashed}, "
                f"terminated={terminated}, "
                f"truncated={truncated}, "
                f"observation_shape={observation.shape}"
            )

            assert observation.shape == (12,), (
                f"Expected observation shape (12,), got {observation.shape}"
            )
            assert env.observation_space.contains(observation), (
                "Observation is outside the declared observation space"
            )

        print("\nTest finished. Random driving is only an environment check, not training.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
