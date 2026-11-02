import argparse
import math
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import PPO, SAC

from racing_env import RacingEnv


COLORS = {
    "SAC": (255, 65, 65),
    "PPO": (60, 170, 255),
}
FINISH_LAPS = 3
MAX_STEPS = 5000
FPS = 60


def load_agent(algorithm, model_path):
    algorithm = algorithm.upper()
    model_class = {"SAC": SAC, "PPO": PPO}.get(algorithm)
    if model_class is None:
        raise ValueError("Algorithm must be SAC or PPO.")
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path.resolve()}\n"
            "Check that the model is inside your project's models folder."
        )
    return model_class.load(str(path))


def set_start_offset(env, lateral_offset):
    """Place cars side-by-side near the same start line, then refresh progress."""
    x, y, angle, _ = env.track.reset_car()
    # Normal to the track heading. Small offsets keep both cars on the road.
    nx, ny = -math.sin(angle), math.cos(angle)
    env.car_x = float(x + nx * lateral_offset)
    env.car_y = float(y + ny * lateral_offset)
    env.car_angle = float(angle)
    env.speed = 0.0
    env.crashed = False
    env.steps = 0

    env.previous_progress, env.previous_progress_distance = (
        env._project_to_path(full_search=True)
    )
    env.start_line_distance = env.previous_progress_distance
    env.relative_progress_distance = 0.0
    env.lap_count = 0
    env.lap_start_step = 0
    env.completed_lap_times = []
    return env._get_observation()


def draw_hud(screen, font, big_font, states, track_name, finish_laps, winner):
    y = 42
    for label, state in states.items():
        info = state["info"]
        color = COLORS[label]
        laps = info.get("lap_count", 0)
        speed = info.get("speed", 0.0)
        crash_text = "CRASHED" if state["crashed"] else ""
        line = (
            f"{label}: {laps}/{finish_laps} laps | "
            f"Speed {speed:5.1f} | "
            f"Last lap {state['last_lap']}"
        )
        if crash_text:
            line += f" | {crash_text}"
        screen.blit(font.render(line, True, color), (14, y))
        y += 25

    title = big_font.render(
        f"AI HEAD-TO-HEAD  |  Track {track_name}", True, (255, 255, 255)
    )
    screen.blit(title, (14, 10))

    if winner:
        msg = big_font.render(winner, True, (255, 235, 80))
        rect = msg.get_rect(center=(screen.get_width() // 2, 55))
        pygame.draw.rect(screen, (25, 25, 25), rect.inflate(24, 14))
        screen.blit(msg, rect)

    help_text = font.render(
        "SPACE: pause/resume   R: restart   ESC: quit",
        True, (255, 255, 255)
    )
    screen.blit(help_text, (14, screen.get_height() - 25))


def run_race(track_name, sac_path, ppo_path, finish_laps, max_steps, render):
    env_sac = RacingEnv(render_mode=None, max_steps=max_steps, track_name=track_name)
    env_ppo = RacingEnv(render_mode=None, max_steps=max_steps, track_name=track_name)

    sac_model = load_agent("SAC", sac_path)
    ppo_model = load_agent("PPO", ppo_path)

    pygame.init()
    track_surface = env_sac.track.track
    screen = pygame.display.set_mode((env_sac.track.W, env_sac.track.H))
    pygame.display.set_caption("SAC vs PPO - AI Racing")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 16, bold=True)
    big_font = pygame.font.SysFont("arial", 22, bold=True)

    def reset_race():
        obs_sac, _ = env_sac.reset()
        obs_ppo, _ = env_ppo.reset()
        # Start beside each other rather than overlapping. Red = SAC, blue = PPO.
        obs_sac = set_start_offset(env_sac, -9.0)
        obs_ppo = set_start_offset(env_ppo, 9.0)
        return {
            "SAC": {
                "env": env_sac, "model": sac_model, "obs": obs_sac,
                "info": {"speed": 0.0, "lap_count": 0},
                "done": False, "crashed": False, "last_lap": "--",
                "total_reward": 0.0,
            },
            "PPO": {
                "env": env_ppo, "model": ppo_model, "obs": obs_ppo,
                "info": {"speed": 0.0, "lap_count": 0},
                "done": False, "crashed": False, "last_lap": "--",
                "total_reward": 0.0,
            },
        }

    states = reset_race()
    paused = False
    running = True
    global_step = 0
    winner = None

    try:
        while running:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        states = reset_race()
                        global_step = 0
                        winner = None
                        paused = False

            if not running:
                break

            if not paused and winner is None and global_step < max_steps:
                # Both policies advance exactly once per shared simulation frame.
                for label in ("SAC", "PPO"):
                    state = states[label]
                    if state["done"]:
                        continue

                    action, _ = state["model"].predict(
                        state["obs"], deterministic=True
                    )
                    obs, reward, terminated, truncated, info = state["env"].step(action)
                    state["obs"] = obs
                    state["info"] = info
                    state["total_reward"] += float(reward)
                    state["crashed"] = bool(info.get("crashed", False))
                    laps = info.get("completed_lap_times", [])
                    if laps:
                        state["last_lap"] = f"{laps[-1]:.2f}s"
                    state["done"] = bool(terminated or truncated)

                    if info.get("lap_count", 0) >= finish_laps:
                        winner = f"{label} WINS!"
                    elif state["crashed"]:
                        other = "PPO" if label == "SAC" else "SAC"
                        if states[other]["crashed"]:
                            winner = "DRAW - BOTH CARS CRASHED"
                        elif not states[other]["done"]:
                            winner = f"{other} WINS - {label} CRASHED"

                global_step += 1

            screen.blit(track_surface, (0, 0))

            for label in ("SAC", "PPO"):
                state = states[label]
                car_env = state["env"]
                car_color = (130, 130, 130) if state["crashed"] else COLORS[label]
                car_env.track.draw_car(
                    screen, car_env.car_x, car_env.car_y,
                    car_env.car_angle, car_color
                )

            draw_hud(screen, font, big_font, states, track_name, finish_laps, winner)
            if paused:
                msg = big_font.render("PAUSED", True, (255, 255, 255))
                screen.blit(msg, msg.get_rect(center=(screen.get_width() // 2, 100)))
            pygame.display.flip()

            if all(state["done"] for state in states.values()):
                winner = winner or "RACE OVER - NO FINISHER"

        print("\n========== RACE SUMMARY ==========")
        print(f"Track: {track_name}")
        print(f"Target: {finish_laps} laps")
        print(f"Winner: {winner or 'Race stopped before finish'}")
        for label, state in states.items():
            times = state["info"].get("completed_lap_times", [])
            print(
                f"{label}: laps={state['info'].get('lap_count', 0)}, "
                f"crashed={state['crashed']}, "
                f"lap_times={[round(t, 2) for t in times]}, "
                f"reward={state['total_reward']:.2f}"
            )
    finally:
        env_sac.close()
        env_ppo.close()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Race trained SAC and PPO agents.")
    parser.add_argument("--track", choices=["A", "B"], default="A")
    parser.add_argument("--sac", default="models/sac_racing_agent_v2.zip")
    parser.add_argument("--ppo", default="models/ppo_racing_agent_v2.zip")
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    run_race(
        track_name=args.track,
        sac_path=args.sac,
        ppo_path=args.ppo,
        finish_laps=args.laps,
        max_steps=args.max_steps,
        render=True,
    )


if __name__ == "__main__":
    main()
