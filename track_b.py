import math
import pygame

pygame.font.init()

# Window setup
W, H = 1000, 720

font = pygame.font.SysFont("arial", 16, bold=True)
big = pygame.font.SysFont("arial", 22, bold=True)

GRASS = (34, 110, 52)
ASPHALT = (55, 55, 60)
KERB = (235, 235, 235)
WHITE = (245, 245, 245)
RED = (225, 30, 30)
CYAN = (0, 220, 255)

# Track and car dimensions, measured in pixels
ROAD_HALF_WIDTH = 35
KERB_HALF_WIDTH = 40
CAR_LENGTH = 28
CAR_WIDTH = 14

# Simple vehicle physics
MAX_SPEED = 220.0
MAX_REVERSE_SPEED = 45.0
ACCELERATION = 145.0
BRAKING = 260.0
FRICTION = 45.0
STEERING_RATE = 2.5

# Virtual LiDAR settings
NUM_RAYS = 11
LIDAR_MAX_RANGE = 150
LIDAR_STEP = 2
LIDAR_FOV = 180

# Track control points, connected in driving order
P = [
    (110, 610), (350, 610), (610, 610), (760, 600),
    (850, 550), (875, 475), (840, 410), (760, 385),
    (660, 405), (585, 370), (600, 300), (690, 270),
    (805, 275), (875, 225), (875, 155), (820, 110),
    (700, 100), (590, 125), (520, 180), (445, 140),
    (340, 105), (220, 115), (130, 155), (85, 220),
    (95, 295), (155, 340), (240, 315), (320, 350),
    (335, 425), (270, 470), (180, 455), (105, 490),
    (85, 555), (100, 590),
]

TURNS = {
    3: 1, 4: 2, 6: 3, 8: 4, 10: 5, 11: 6, 13: 7,
    15: 8, 17: 9, 18: 10, 20: 11, 22: 12, 24: 13,
    26: 14, 28: 15, 30: 16, 32: 17,
}


# Generate a smooth closed curve through the track control points
def catmull(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t

    return tuple(
        0.5 * (
            2 * p1[i]
            + (-p0[i] + p2[i]) * t
            + (2 * p0[i] - 5 * p1[i]
               + 4 * p2[i] - p3[i]) * t2
            + (-p0[i] + 3 * p1[i]
               - 3 * p2[i] + p3[i]) * t3
        )
        for i in (0, 1)
    )


STEPS = 20
N_CONTROL = len(P)
path = []
turn_pos = {}

for i in range(N_CONTROL):
    p0 = P[i - 1]
    p1 = P[i]
    p2 = P[(i + 1) % N_CONTROL]
    p3 = P[(i + 2) % N_CONTROL]

    if i in TURNS:
        turn_pos[TURNS[i]] = p1

    for s in range(STEPS):
        path.append(catmull(p0, p1, p2, p3, s / STEPS))

N = len(path)


# Calculate the direction of travel at a point on the track
def heading(i):
    a = path[i % N]
    b = path[(i + 1) % N]
    return math.atan2(b[1] - a[1], b[0] - a[0])


# Draw the circuit and create a separate mask for collision detection
track = pygame.Surface((W, H))
track.fill(GRASS)

# White pixels represent drivable asphalt; black pixels represent everything else
road_mask = pygame.Surface((W, H))
road_mask.fill((0, 0, 0))

for x, y in path:
    pygame.draw.circle(
        track, KERB, (round(x), round(y)), KERB_HALF_WIDTH
    )

for x, y in path:
    pygame.draw.circle(
        track, ASPHALT, (round(x), round(y)), ROAD_HALF_WIDTH
    )
    pygame.draw.circle(
        road_mask, WHITE, (round(x), round(y)), ROAD_HALF_WIDTH
    )

# Dashed center markings
for i in range(0, N, 6):
    a = path[i]
    b = path[(i + 2) % N]

    pygame.draw.line(
        track, (90, 90, 95),
        (round(a[0]), round(a[1])),
        (round(b[0]), round(b[1])),
        2
    )

# Start/finish line
START_INDEX = STEPS
sf = path[START_INDEX]
ang = heading(START_INDEX) + math.pi / 2

dx = math.cos(ang) * ROAD_HALF_WIDTH
dy = math.sin(ang) * ROAD_HALF_WIDTH

for k in range(12):
    color = WHITE if k % 2 == 0 else (0, 0, 0)

    pygame.draw.line(
        track,
        color,
        (
            sf[0] - dx + dx * k / 6,
            sf[1] - dy + dy * k / 6
        ),
        (
            sf[0] - dx + dx * (k + 1) / 6,
            sf[1] - dy + dy * (k + 1) / 6
        ),
        5
    )

# Label the turns on the circuit
cx = sum(p[0] for p in P) / N_CONTROL
cy = sum(p[1] for p in P) / N_CONTROL

for num, (x, y) in turn_pos.items():
    vx, vy = x - cx, y - cy
    mag = math.hypot(vx, vy) or 1

    lx = x + vx / mag * 32
    ly = y + vy / mag * 32

    txt = font.render(f"T{num}", True, WHITE)
    track.blit(txt, txt.get_rect(center=(lx, ly)))

title = big.render(
    "Track B ",
    True, WHITE
)
track.blit(title, (15, 12))


# Calculate the four corners of the rotated rectangular car
def get_car_corners(x, y, angle):
    forward_x = math.cos(angle)
    forward_y = math.sin(angle)

    side_x = -math.sin(angle)
    side_y = math.cos(angle)

    half_length = CAR_LENGTH / 2
    half_width = CAR_WIDTH / 2

    corners = []

    for forward_sign, side_sign in [
        (1, 1), (1, -1), (-1, -1), (-1, 1)
    ]:
        px = (
            x
            + forward_sign * half_length * forward_x
            + side_sign * half_width * side_x
        )

        py = (
            y
            + forward_sign * half_length * forward_y
            + side_sign * half_width * side_y
        )

        corners.append((px, py))

    return corners


# Check the car body against the road mask to detect crashes
def car_is_on_road(x, y, angle):
    points = get_car_corners(x, y, angle)
    samples = list(points)

    # Sample the edges as well as the corners to catch boundary crossings
    for i in range(4):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % 4]

        for t in (0.25, 0.5, 0.75):
            samples.append((
                x1 + (x2 - x1) * t,
                y1 + (y2 - y1) * t
            ))

    for px, py in samples:
        ix, iy = int(px), int(py)

        if not (0 <= ix < W and 0 <= iy < H):
            return False

        if road_mask.get_at((ix, iy))[0] == 0:
            return False

    return True


# Simulate LiDAR by advancing rays until they leave the road mask
def lidar_scan(x, y, angle):
    distances = []
    endpoints = []
    hit_flags = []

    for i in range(NUM_RAYS):
        offset = (
            -LIDAR_FOV / 2
            + i * LIDAR_FOV / (NUM_RAYS - 1)
        )

        ray_angle = angle + math.radians(offset)
        distance = LIDAR_MAX_RANGE
        hit = False

        for d in range(0, LIDAR_MAX_RANGE + 1, LIDAR_STEP):
            ray_x = x + math.cos(ray_angle) * d
            ray_y = y + math.sin(ray_angle) * d

            px, py = int(ray_x), int(ray_y)

            if not (0 <= px < W and 0 <= py < H):
                distance = d
                hit = True
                break

            if road_mask.get_at((px, py))[0] == 0:
                distance = d
                hit = True
                break

        distances.append(distance)
        endpoints.append((
            x + math.cos(ray_angle) * distance,
            y + math.sin(ray_angle) * distance
        ))
        hit_flags.append(hit)

    return distances, endpoints, hit_flags


# Draw the car body and its front windshield
def draw_car(surface, x, y, angle, color):
    corners = get_car_corners(x, y, angle)

    pygame.draw.polygon(
        surface,
        color,
        [(round(px), round(py)) for px, py in corners]
    )

    fx = x + math.cos(angle) * 5
    fy = y + math.sin(angle) * 5

    pygame.draw.circle(
        surface, (100, 190, 220),
        (round(fx), round(fy)), 3
    )


# Reset the car at the starting point, facing along the track
def reset_car():
    x, y = path[START_INDEX]
    return float(x), float(y), heading(START_INDEX), 0.0



def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Track B - Technical Circuit")
    clock = pygame.time.Clock()

    car_x, car_y, car_angle, speed = reset_car()
    crashed = False
    paused = False
    running = True

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.04)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:
                    car_x, car_y, car_angle, speed = reset_car()
                    crashed = False

        keys = pygame.key.get_pressed()

        if not paused and not crashed:
            # Update speed using acceleration, braking and friction
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                speed += ACCELERATION * dt

            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                if speed > 0:
                    speed = max(0, speed - BRAKING * dt)
                else:
                    speed -= ACCELERATION * 0.5 * dt

            else:
                if speed > 0:
                    speed = max(0, speed - FRICTION * dt)
                elif speed < 0:
                    speed = min(0, speed + FRICTION * dt)

            speed = max(-MAX_REVERSE_SPEED, min(MAX_SPEED, speed))

            # Steering changes the car's heading, with less effect at low speeds
            steering = 0

            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                steering -= 1

            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                steering += 1

            speed_factor = min(abs(speed) / 65.0, 1.0)

            if abs(speed) > 1:
                direction = 1 if speed > 0 else -1

                car_angle += (
                    steering * STEERING_RATE
                    * speed_factor * direction * dt
                )

            # Move the car along its current heading
            old_x, old_y, old_angle = car_x, car_y, car_angle

            car_x += math.cos(car_angle) * speed * dt
            car_y += math.sin(car_angle) * speed * dt

            # Stop the car and restore its previous position if it leaves the road
            if not car_is_on_road(car_x, car_y, car_angle):
                crashed = True
                speed = 0.0
                car_x, car_y, car_angle = old_x, old_y, old_angle

        screen.blit(track, (0, 0))

        # Scan the road and draw the LiDAR rays
        lidar_distances, lidar_endpoints, lidar_hits = lidar_scan(
            car_x, car_y, car_angle
        )

        for endpoint, hit in zip(lidar_endpoints, lidar_hits):
            end_x, end_y = endpoint
            color = (255, 220, 0) if hit else (0, 220, 100)

            pygame.draw.line(
                screen, CYAN,
                (round(car_x), round(car_y)),
                (round(end_x), round(end_y)),
                1
            )

            pygame.draw.circle(
                screen, color,
                (round(end_x), round(end_y)),
                3
            )

        draw_car(
            screen, car_x, car_y, car_angle,
            (255, 60, 60) if crashed else RED
        )

        # Display selected LiDAR measurements and current speed
        left_distance = lidar_distances[2]
        front_distance = lidar_distances[NUM_RAYS // 2]
        right_distance = lidar_distances[-3]

        sensor_text = font.render(
            f"LiDAR L: {left_distance:.0f}px  "
            f"Front: {front_distance:.0f}px  "
            f"R: {right_distance:.0f}px",
            True, WHITE
        )
        screen.blit(sensor_text, (W - 330, 70))

        speed_text = font.render(
            f"Speed: {abs(speed):.0f} px/s",
            True, WHITE
        )
        screen.blit(speed_text, (W - 190, 45))

        if crashed:
            message = big.render("CRASHED! Press R to reset", True, WHITE)
            screen.blit(message, message.get_rect(center=(W // 2, 55)))

        elif paused:
            message = big.render("PAUSED", True, WHITE)
            screen.blit(message, message.get_rect(center=(W // 2, 55)))

        controls = font.render(
            "W/UP: accelerate  S/DOWN: brake  "
            "A/LEFT, D/RIGHT: steer  SPACE: pause  R: reset  ESC: quit",
            True, WHITE
        )
        screen.blit(controls, (15, H - 28))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
