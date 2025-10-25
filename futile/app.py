from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pygame

from .input import read_input
from .physics import (
    CollisionWorld,
    PhysicsState,
    build_default_collision_world,
    ground_height,
    handle_movement,
    upd_phys,
)
from .render import (
    DirectionalLight,
    LightingSetup,
    PointLight,
    RenderContext,
    draw_debug,
    draw_grid,
    render_scene,
)
from .resources import MeshManager
from .world import Material, WorldObject


class GameApplication:
    width: int = 1280
    height: int = 720
    window_title: str = "Futile - Commented"
    target_fps: int = 60
    move_speed: float = 180.0
    dash_multiplier: float = 1.9
    acceleration: float = 15.0
    friction: float = 8.0
    mouse_sensitivity: float = 0.002
    fov_distance: float = 400.0
    gravity: float = -300.0
    jump_step: float = 10.0
    default_jump_velocity: float = 140.0
    ground_y_base: float = -200.0
    grid_offset: float = -30.0
    grid_size: int = 100
    grid_range: int = 900
    eye_height: float = 40.0
    max_step: float = 30.0
    pitch_min: float = -math.pi / 2 + 0.01
    pitch_max: float = math.pi / 2 - 0.01
    clear_color: tuple[int, int, int] = (15, 15, 20)

    def __init__(self) -> None:
        pygame.init()
        self.running = True
        self.clock = pygame.time.Clock()
        size = self.get_window_size()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(self.get_window_title())
        self.lighting = self.create_lighting_setup()
        self.ctx = RenderContext(
            screen=self.screen,
            cx=size[0] // 2,
            cy=size[1] // 2,
            fov_d=self.fov_distance,
            lighting=self.lighting,
        )
        self.cam = self.create_camera()
        self.slopes = self.create_slopes()
        self.collision_world = self.create_collision_world(self.slopes)
        self.state = self.create_physics_state()
        self.jump_velocity = self.get_default_jump_velocity()
        self.jump_default = self.jump_velocity
        self.asset_directory = self.get_asset_directory()
        self.mesh_manager = self.create_mesh_manager(self.asset_directory)
        self.world_objects = self.create_world_objects(self.mesh_manager)
        self.cam["y"] = ground_height(
            self.cam["x"],
            self.cam["z"],
            self.ground_y_base,
            self.collision_world,
        ) + self.eye_height
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        self.fps = 0.0

    def get_window_size(self) -> tuple[int, int]:
        return self.width, self.height

    def get_window_title(self) -> str:
        return self.window_title

    def get_target_fps(self) -> int:
        return self.target_fps

    def get_default_jump_velocity(self) -> float:
        return self.default_jump_velocity

    def get_asset_directory(self) -> Path:
        return Path(__file__).resolve().parent.parent / "assets"

    def create_camera(self) -> dict[str, float]:
        return {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
        }

    def create_slopes(self) -> list[dict[str, Any]]:
        return []

    def create_collision_world(self, slopes: list[dict[str, Any]]) -> CollisionWorld:
        return build_default_collision_world(self.ground_y_base, slopes)

    def create_physics_state(self) -> PhysicsState:
        return PhysicsState()

    def create_mesh_manager(self, asset_directory: Path) -> MeshManager:
        return MeshManager(asset_directory)

    def material_from_rgb(self, rgb: tuple[int, int, int], shininess: float, rim: float) -> Material:
        return Material(
            diffuse_color=tuple(max(0.0, min(1.0, c / 255.0)) for c in rgb),
            specular_color=(0.9, 0.9, 0.9),
            shininess=shininess,
            ambient_factor=1.0,
            rim_strength=rim,
        )

    def create_world_objects(self, mesh_manager: MeshManager) -> list[WorldObject]:
        return [
            WorldObject(
                mesh=mesh_manager.load("cube.fsm"),
                position=(0.0, -120.0, 420.0),
                rotation=(0.0, 0.0, 0.0),
                scale=60.0,
                color=(190, 180, 210),
                material=self.material_from_rgb((190, 180, 210), shininess=28.0, rim=0.65),
            ),
            WorldObject(
                mesh=mesh_manager.load("pyramid.obj"),
                position=(160.0, -140.0, 600.0),
                rotation=(0.0, math.radians(25.0), 0.0),
                scale=80.0,
                color=(210, 170, 150),
                material=self.material_from_rgb((210, 170, 150), shininess=34.0, rim=0.55),
            ),
        ]

    def create_lighting_setup(self) -> LightingSetup:
        return LightingSetup(
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

    def process_events(self) -> None:
        for event in pygame.event.get():
            self.handle_event(event)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.stop()
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.stop()
                return
            if event.key == pygame.K_LEFTBRACKET:
                self.jump_velocity = max(0.0, self.jump_velocity - self.jump_step)
                return
            if event.key == pygame.K_RIGHTBRACKET:
                self.jump_velocity += self.jump_step
                return
            if event.key == pygame.K_0:
                self.jump_velocity = self.jump_default
                return
        if event.type == pygame.ACTIVEEVENT:
            state = getattr(event, "state", 0)
            gain = getattr(event, "gain", 1)
            if state & 2 and not gain:
                try:
                    pygame.mouse.get_rel()
                except pygame.error:
                    pass

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        try:
            while self.running:
                self.process_events()
                if not self.running:
                    break
                ms = self.clock.tick(self.get_target_fps())
                dt = ms / 1000.0
                self.fps = self.clock.get_fps()
                self.update(dt)
                self.render()
        finally:
            self.shutdown()

    def update(self, dt: float) -> None:
        mvx, mvz, dash, jump = read_input(
            self.cam,
            self.mouse_sensitivity,
            self.pitch_min,
            self.pitch_max,
        )
        handle_movement(
            self.cam,
            self.state,
            mvx,
            mvz,
            dash,
            jump,
            dt,
            self.move_speed,
            self.dash_multiplier,
            self.acceleration,
            self.friction,
            self.collision_world,
            self.eye_height,
            self.max_step,
            self.ground_y_base,
            self.jump_velocity,
        )
        upd_phys(
            self.cam,
            self.state,
            dt,
            self.gravity,
            self.eye_height,
            self.collision_world,
            self.ground_y_base,
        )

    def render(self) -> None:
        self.screen.fill(self.clear_color)
        draw_grid(
            self.ctx,
            self.cam,
            self.slopes,
            self.ground_y_base,
            self.grid_offset,
            self.grid_size,
            self.grid_range,
        )
        render_scene(self.ctx, self.cam, self.world_objects)
        draw_debug(
            self.ctx,
            self.cam,
            self.state,
            self.fps,
            self.jump_velocity,
            self.gravity,
            self.slopes,
        )
        pygame.display.flip()

    def shutdown(self) -> None:
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        pygame.quit()
