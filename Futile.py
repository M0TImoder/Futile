import pygame
import sys
import math

pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Futile - Network Ready Camera")

clock = pygame.time.Clock()
cx, cy = WIDTH // 2, HEIGHT // 2

# Core param
SPEED = 5
MOUSE_SENSITIVITY = 0.002
FOV_DISTANCE = 400

# Cube geometry
CUBE_SIZE = 100
CUBE_VERTICES = [
    [-CUBE_SIZE, -CUBE_SIZE, -CUBE_SIZE],
    [ CUBE_SIZE, -CUBE_SIZE, -CUBE_SIZE],
    [ CUBE_SIZE,  CUBE_SIZE, -CUBE_SIZE],
    [-CUBE_SIZE,  CUBE_SIZE, -CUBE_SIZE],
    [-CUBE_SIZE, -CUBE_SIZE,  CUBE_SIZE],
    [ CUBE_SIZE, -CUBE_SIZE,  CUBE_SIZE],
    [ CUBE_SIZE,  CUBE_SIZE,  CUBE_SIZE],
    [-CUBE_SIZE,  CUBE_SIZE,  CUBE_SIZE]
]
CUBE_EDGES = [
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,6), (3,7)
]

# World setup
WORLD_CUBES = []
for z in range(0, 2000, 400):
    for x in [-400, 0, 400]:
        WORLD_CUBES.append((x, 0, z + 500))

# Camera status
camera = {
    "pos": [0.0, 0.0, 0.0],
    "yaw": 0.0,    # 左右回転
    "pitch": 0.0   # 上下回転
}

def project_point(x, y, z):
    """単純な透視投影"""
    factor = FOV_DISTANCE / (z + FOV_DISTANCE)
    x_proj = x * factor + cx
    y_proj = -y * factor + cy
    return (x_proj, y_proj)

def rotate_camera(px, py, pz, cam):
    """カメラ方向の回転のみ適用"""
    yaw, pitch = cam["yaw"], cam["pitch"]

    # yaw
    x = px * math.cos(yaw) - pz * math.sin(yaw)
    z = px * math.sin(yaw) + pz * math.cos(yaw)

    # pitch
    y = py * math.cos(pitch) - z * math.sin(pitch)
    z = py * math.sin(pitch) + z * math.cos(pitch)

    return x, y, z

def update_camera_input():
    """マウスとキー入力からカメラ更新"""
    mx, my = pygame.mouse.get_rel()
    camera["yaw"] += mx * MOUSE_SENSITIVITY
    camera["pitch"] -= my * MOUSE_SENSITIVITY
    camera["pitch"] = max(-math.pi/2, min(math.pi/2, camera["pitch"]))

    keys = pygame.key.get_pressed()
    forward_x = math.sin(camera["yaw"])
    forward_z = math.cos(camera["yaw"])
    right_x = math.cos(camera["yaw"])
    right_z = -math.sin(camera["yaw"])

    if keys[pygame.K_w]:
        camera["pos"][0] += forward_x * SPEED
        camera["pos"][2] += forward_z * SPEED
    if keys[pygame.K_s]:
        camera["pos"][0] -= forward_x * SPEED
        camera["pos"][2] -= forward_z * SPEED
    if keys[pygame.K_a]:
        camera["pos"][0] -= right_x * SPEED
        camera["pos"][2] -= right_z * SPEED
    if keys[pygame.K_d]:
        camera["pos"][0] += right_x * SPEED
        camera["pos"][2] += right_z * SPEED
    if keys[pygame.K_SPACE]:
        camera["pos"][1] += SPEED
    if keys[pygame.K_LSHIFT]:
        camera["pos"][1] -= SPEED

# Input config
pygame.event.set_grab(True)
pygame.mouse.set_visible(False)

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    update_camera_input()
    screen.fill((15, 15, 20))

    # すべての立方体を描画
    for (wx, wy, wz) in WORLD_CUBES:
        projected = []
        for vx, vy, vz in CUBE_VERTICES:
            # カメラ基準で変換
            x = vx + wx - camera["pos"][0]
            y = vy + wy - camera["pos"][1]
            z = vz + wz - camera["pos"][2]

            # カメラ方向へ回転
            x, y, z = rotate_camera(x, y, z, camera)

            if z > -50:  # 背面カット
                projected.append(project_point(x, y, z))

        if len(projected) == 8:
            for e in CUBE_EDGES:
                pygame.draw.line(screen, (255, 255, 255),
                                 projected[e[0]], projected[e[1]], 1)

    # デバッグ表示（座標・角度）
    font = pygame.font.SysFont("consolas", 18)
    pos = camera["pos"]
    yaw = math.degrees(camera["yaw"])
    pitch = math.degrees(camera["pitch"])
    info = f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | Yaw: {yaw:.1f} | Pitch: {pitch:.1f}"
    screen.blit(font.render(info, True, (200, 200, 200)), (10, 10))

    pygame.display.flip()
    clock.tick(60)
