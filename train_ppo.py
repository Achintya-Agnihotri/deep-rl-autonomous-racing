
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from racing_env import RacingEnv


TOTAL_TIMESTEPS = 500_000
MODEL_PATH = "ppo_racing_agent_v2"


def main():
    env = Monitor(
        RacingEnv(
            render_mode=None,
            max_steps=5000,
            track_name="A",
        )
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log="./tensorboard_logs/",
        verbose=1,
        device="cpu",
        seed=42,
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            progress_bar=False,
            tb_log_name="PPO_Racing",
        )

        model.save(MODEL_PATH)
        print(f"\nTraining complete. Model saved as {MODEL_PATH}.zip")

    finally:
        env.close()


if __name__ == "__main__":
    main()
