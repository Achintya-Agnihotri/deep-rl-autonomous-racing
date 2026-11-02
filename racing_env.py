
import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

import track
import track_b


class RacingEnv(gym.Env):
    """Gymnasium environment for the custom racing simulator."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, max_steps=5000, track_name="A"):
        super().__init__()

        track_name = str(track_name).upper()
        if track_name not in ("A", "B"):
            raise ValueError("track_name must be 'A' or 'B'")

        self.track_name = track_name
        self.track = track if track_name == "A" else track_b

        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")

        self.render_mode = render_mode
        self.max_steps = max_steps

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 11 LiDAR readings + 1 normalized speed value.
        self.observation_space = spaces.Box(
            low=np.zeros(12, dtype=np.float32),
            high=np.ones(12, dtype=np.float32),
            dtype=np.float32,
        )

        self.screen = None
        self.clock = None
        self.font = None

        if render_mode == "human":
            if not pygame.get_init():
                pygame.init()

            self.screen = pygame.display.set_mode(
                (self.track.W, self.track.H)
            )
            pygame.display.set_caption("RL Racing Agent")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 16, bold=True)

        self.steps = 0

        self.path_points = np.asarray(
            self.track.path, dtype=np.float64
        )
        self.path_segments = (
            np.roll(self.path_points, -1, axis=0) - self.path_points
        )
        self.path_segment_lengths = np.linalg.norm(
            self.path_segments, axis=1
        )
        self.path_segment_lengths_squared = (
            self.path_segment_lengths ** 2
        )

        self.path_cumulative_lengths = np.concatenate(
            (
                np.array([0.0]),
                np.cumsum(self.path_segment_lengths),
            )
        )

        self.track_length = float(self.path_cumulative_lengths[-1])
        self.progress_reward_scale = (
            0.1 / float(self.path_segment_lengths.mean())
        )

        self.previous_progress = 0
        self.previous_progress_distance = 0.0

        self._reset_state()

    def _reset_state(self):
        (
            self.car_x,
            self.car_y,
            self.car_angle,
            self.speed,
        ) = self.track.reset_car()

        self.crashed = False
        self.steps = 0

        (
            self.previous_progress,
            self.previous_progress_distance,
        ) = self._project_to_path(full_search=True)

        self.start_line_distance = self.previous_progress_distance
        self.relative_progress_distance = 0.0

        self.lap_count = 0
        self.lap_start_step = 0
        self.completed_lap_times = []

    def _project_to_path(self, full_search=False):
        n = len(self.path_points)

        if full_search:
            candidates = np.arange(n)
        else:
            window = 120
            candidates = (
                self.previous_progress
                + np.arange(-window, window + 1)
            ) % n

        starts = self.path_points[candidates]
        segments = self.path_segments[candidates]

        offsets = (
            np.array([self.car_x, self.car_y]) - starts
        )

        fractions = np.clip(
            np.einsum("ij,ij->i", offsets, segments)
            / self.path_segment_lengths_squared[candidates],
            0.0,
            1.0,
        )

        projections = starts + fractions[:, np.newaxis] * segments

        distances_squared = np.sum(
            (
                projections
                - np.array([self.car_x, self.car_y])
            ) ** 2,
            axis=1,
        )

        candidate_position = int(np.argmin(distances_squared))
        segment_index = int(candidates[candidate_position])

        progress_distance = float(
            self.path_cumulative_lengths[segment_index]
            + fractions[candidate_position]
            * self.path_segment_lengths[segment_index]
        )

        progress_index = (
            segment_index
            + int(fractions[candidate_position] >= 0.5)
        ) % n

        return progress_index, progress_distance

    def _get_observation(self):
        distances, _, _ = self.track.lidar_scan(
            self.car_x,
            self.car_y,
            self.car_angle,
        )

        lidar = (
            np.asarray(distances, dtype=np.float32)
            / float(self.track.LIDAR_MAX_RANGE)
        )

        normalized_speed = (
            self.speed / float(self.track.MAX_SPEED)
        )

        observation = np.concatenate(
            [
                np.clip(lidar, 0.0, 1.0),
                np.array(
                    [np.clip(normalized_speed, 0.0, 1.0)],
                    dtype=np.float32,
                ),
            ]
        )

        return observation.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        observation = self._get_observation()

        info = {
            "speed": self.speed,
            "track_name": self.track_name,
            "progress_index": self.previous_progress,
            "crashed": False,
            "lap_count": self.lap_count,
            "completed_lap_times": list(self.completed_lap_times),
            "elapsed_time": 0.0,
        }

        return observation, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        if action.shape != (2,):
            raise ValueError(
                f"Expected action shape (2,), received {action.shape}"
            )

        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))

        dt = 1.0 / self.metadata["render_fps"]

        old_x = self.car_x
        old_y = self.car_y
        old_angle = self.car_angle

        # Positive throttle accelerates; negative throttle brakes.
        if throttle_brake >= 0.0:
            self.speed += (
                self.track.ACCELERATION * throttle_brake * dt
            )
        else:
            self.speed = max(
                0.0,
                self.speed
                + self.track.BRAKING * throttle_brake * dt,
            )

        # Apply friction near neutral throttle.
        if abs(throttle_brake) < 0.05:
            self.speed = max(
                0.0,
                self.speed - self.track.FRICTION * dt,
            )

        self.speed = float(
            np.clip(self.speed, 0.0, self.track.MAX_SPEED)
        )

        speed_factor = min(self.speed / 65.0, 1.0)

        if self.speed > 1.0:
            self.car_angle += (
                steering
                * self.track.STEERING_RATE
                * speed_factor
                * dt
            )

        self.car_x += math.cos(self.car_angle) * self.speed * dt
        self.car_y += math.sin(self.car_angle) * self.speed * dt

        # Detect collision with the road boundary.
        if not self.track.car_is_on_road(
            self.car_x,
            self.car_y,
            self.car_angle,
        ):
            self.crashed = True
            self.speed = 0.0

            # Keep the car at its previous valid position.
            self.car_x = old_x
            self.car_y = old_y
            self.car_angle = old_angle

        old_progress_distance = self.previous_progress_distance

        new_index, new_progress_distance = self._project_to_path()

        delta = new_progress_distance - old_progress_distance

        # Correct for progress wrapping around the closed circuit.
        if delta > self.track_length / 2.0:
            delta -= self.track_length
        elif delta < -self.track_length / 2.0:
            delta += self.track_length

        self.previous_progress = new_index
        self.previous_progress_distance = new_progress_distance

        old_relative_distance = self.relative_progress_distance

        new_relative_distance = (
            new_progress_distance - self.start_line_distance
        ) % self.track_length

        crossed_start_finish = (
            delta > 0.0
            and old_relative_distance > self.track_length * 0.85
            and new_relative_distance < self.track_length * 0.15
        )

        self.relative_progress_distance = new_relative_distance

        self.steps += 1

        if crossed_start_finish:
            self.lap_count += 1

            lap_steps = self.steps - self.lap_start_step
            lap_time = (
                lap_steps / float(self.metadata["render_fps"])
            )

            self.completed_lap_times.append(lap_time)
            self.lap_start_step = self.steps

       
        # Reward genuine forward progress.
        progress_reward = delta * self.progress_reward_scale

        # Modest reward for maintaining speed.
        speed_reward = (
            self.speed / float(self.track.MAX_SPEED)
        ) * 0.05

        # Time penalty.
        time_penalty = -0.01

        # Penalize remaining stationary, but allow brief slowdowns
        # while navigating difficult corners.
        stuck_penalty = -0.05 if self.speed < 3.0 else 0.0

        reward = (
            progress_reward
            + speed_reward
            + time_penalty
            + stuck_penalty
        )


        if self.crashed:
            reward -= 100.0

        terminated = bool(self.crashed)
        truncated = bool(
            self.steps >= self.max_steps and not terminated
        )

        observation = self._get_observation()

        info = {
            "speed": self.speed,
            "track_name": self.track_name,
            "steering": steering,
            "throttle_brake": throttle_brake,
            "progress_delta": delta,
            "progress_reward": progress_reward,
            "speed_reward": speed_reward,
            "time_penalty": time_penalty,
            "crashed": self.crashed,
            "progress_index": self.previous_progress,
            "lap_count": self.lap_count,
            "completed_lap_times": list(self.completed_lap_times),
            "elapsed_time": (
                self.steps / float(self.metadata["render_fps"])
            ),
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            if not pygame.get_init():
                pygame.init()

            self.screen = pygame.display.set_mode(
                (self.track.W, self.track.H)
            )
            pygame.display.set_caption("RL Racing Agent")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 16, bold=True)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.blit(self.track.track, (0, 0))

        _, endpoints, hits = self.track.lidar_scan(
            self.car_x,
            self.car_y,
            self.car_angle,
        )

        for endpoint, hit in zip(endpoints, hits):
            color = (255, 220, 0) if hit else (0, 220, 100)

            pygame.draw.line(
                self.screen,
                self.track.CYAN,
                (round(self.car_x), round(self.car_y)),
                (round(endpoint[0]), round(endpoint[1])),
                1,
            )

            pygame.draw.circle(
                self.screen,
                color,
                (round(endpoint[0]), round(endpoint[1])),
                3,
            )

        self.track.draw_car(
            self.screen,
            self.car_x,
            self.car_y,
            self.car_angle,
            (255, 60, 60) if self.crashed else self.track.RED,
        )

        status_text = self.font.render(
            f"Speed: {self.speed:.0f} px/s  "
            f"Progress: {self.previous_progress}  "
            f"Laps: {self.lap_count}",
            True,
            self.track.WHITE,
        )

        self.screen.blit(status_text, (15, 40))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human" and pygame.display.get_init():
            pygame.display.quit()

        self.screen = None
        self.clock = None
        self.font = None
