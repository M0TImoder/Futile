import math
import sys
from pathlib import Path

import pygame

from futile import (
    CollisionWorld,
    DirectionalLight,
    LightingSetup,
    MeshManager,
    PhysicsState,
    PointLight,
    RenderContext,
    WorldObject,
    build_default_collision_world,
    draw_debug,
    draw_grid,
    ground_height,
    handle_movement,
    read_input,
    render_scene,
    upd_phys,
)
from futile.world import Material

W, H = 1280, 720
SPD = 180.0
DASH_MUL = 1.9
ACC = 15.0
FRIC = 8.0
M_SENS = 0.002
FOV_D = 400.0
GRAV = -300.0
JUMP_STEP = 10.0
GROUND_Y_BASE = -200.0
GRID_OFF = -30.0
G_SIZE = 100
G_RANGE = 900
EYE_H = 40.0
MAX_STEP = 30.0
PITCH_MIN = -math.pi / 2 + 0.01
PITCH_MAX = math.pi / 2 - 0.01
ASSET_DIR = Path(__file__).parent / "assets"


def material_from_rgb(rgb: tuple[int, int, int], shininess: float, rim: float) -> Material:
    return Material(
        diffuse_color=tuple(max(0.0, min(1.0, c / 255.0)) for c in rgb),
        specular_color=(0.9, 0.9, 0.9),
        shininess=shininess,
        ambient_factor=1.0,
        rim_strength=rim,
    )


def main() -> None:
    pygame.init()

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Futile - Commented")
    clk = pygame.time.Clock()
    lighting_setup = LightingSetup(
        ambient_color=(0.18, 0.19, 0.22),
        directional_lights=[
            DirectionalLight(
                direction=(0.3, 0.8, 0.5),
                color=(1.0, 0.98, 0.95),
                intensity=0.75,
                specular_intensity=0.2,
            ),
            DirectionalLight(
                direction=(-0.2, -0.7, -0.4),
                color=(0.4, 0.5, 0.65),
                intensity=0.25,
                specular_intensity=0.05,
            ),
        ],
        point_lights=[
            PointLight(
                position=(0.0, -80.0, 520.0),
                color=(0.9, 0.85, 0.8),
                intensity=0.45,
                radius=520.0,
                attenuation=1.0,
                specular_intensity=0.08,
            ),
        ],
        rim_intensity=0.08,
    )
    # キーライトを基準に補助光で陰影を柔らかく調整
    ctx = RenderContext(screen=screen, cx=W // 2, cy=H // 2, fov_d=FOV_D, lighting=lighting_setup)

    cam = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0}
    slopes: list[dict[str, float]] = []
    collision_world: CollisionWorld = build_default_collision_world(GROUND_Y_BASE, slopes)
    state = PhysicsState()
    jump_v = 140.0
    jump_default = jump_v
    mesh_manager = MeshManager(ASSET_DIR)
    # メッシュリソースのキャッシュを活用
    world_objects = [
        WorldObject(
            mesh=mesh_manager.load("cube.fsm"),
            position=(0.0, -120.0, 420.0),
            rotation=(0.0, 0.0, 0.0),
            scale=60.0,
            color=(190, 180, 210),
            material=material_from_rgb((190, 180, 210), shininess=28.0, rim=0.65),
        ),
        WorldObject(
            mesh=mesh_manager.load("pyramid.obj"),
            position=(160.0, -140.0, 600.0),
            rotation=(0.0, math.radians(25.0), 0.0),
            scale=80.0,
            color=(210, 170, 150),
            material=material_from_rgb((210, 170, 150), shininess=34.0, rim=0.55),
        ),
    ]

    cam["y"] = ground_height(cam["x"], cam["z"], GROUND_Y_BASE, collision_world) + EYE_H

    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_LEFTBRACKET:
                    jump_v = max(0.0, jump_v - JUMP_STEP)
                if event.key == pygame.K_RIGHTBRACKET:
                    jump_v += JUMP_STEP
                if event.key == pygame.K_0:
                    jump_v = jump_default

            if event.type == pygame.ACTIVEEVENT:
                try:
                    if event.state & 2 and not event.gain:
                        pygame.mouse.get_rel()
                except Exception:
                    pass

        ms = clk.tick(60)
        dt = ms / 1000.0
        fps = clk.get_fps()

        mvx, mvz, dash, jump = read_input(cam, M_SENS, PITCH_MIN, PITCH_MAX)
        handle_movement(
            cam,
            state,
            mvx,
            mvz,
            dash,
            jump,
            dt,
            SPD,
            DASH_MUL,
            ACC,
            FRIC,
            collision_world,
            EYE_H,
            MAX_STEP,
            GROUND_Y_BASE,
            jump_v,
        )
        upd_phys(cam, state, dt, GRAV, EYE_H, collision_world, GROUND_Y_BASE)

        screen.fill((15, 15, 20))
        draw_grid(ctx, cam, slopes, GROUND_Y_BASE, GRID_OFF, G_SIZE, G_RANGE)
        render_scene(ctx, cam, world_objects)
        draw_debug(ctx, cam, state, fps, jump_v, GRAV, slopes)
        pygame.display.flip()


if __name__ == "__main__":
    main()
