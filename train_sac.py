
import os

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from racing_env import RacingEnv

TOTAL_TIMESTEPS = 500_000

MODEL_PATH = "models/sac_racing_agent_v3"
REPLAY_BUFFER_PATH = "models/sac_racing_replay_buffer_v3.pkl"
TENSORBOARD_RUN_NAME = "SAC_Racing_V3"


def main():
    os.makedirs("models", exist_ok=True)

    env = Monitor(
        RacingEnv(
            render_mode=None,
            max_steps=5000,
            track_name="A"
        )
    )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
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
            tb_log_name=TENSORBOARD_RUN_NAME,
        )

        model.save(MODEL_PATH)
        model.save_replay_buffer(REPLAY_BUFFER_PATH)

        print(f"Model saved: {MODEL_PATH}.zip")
        print(f"Replay buffer saved: {REPLAY_BUFFER_PATH}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
