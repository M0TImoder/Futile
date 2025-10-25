from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Type

import pygame

from .engine import Engine, EngineConfig, Scene
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
    draw_ground_fill,
    render_scene,
)
from .resources import MeshManager
from .world import DimensionSettings, Material, WorldObject, load_dimension


class DefaultScene(Scene):
    """標準的なウォークスルーシーン。"""

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
    grid_range: int = 1500
    eye_height: float = 40.0
    max_step: float = 30.0
    pitch_min: float = -math.pi / 2 + 0.01
    pitch_max: float = math.pi / 2 - 0.01
    clear_color: tuple[int, int, int] = (15, 15, 20)

    def __init__(self, engine: Engine, dimension: DimensionSettings | None = None) -> None:
        super().__init__(engine)
        if dimension is None:
            dimension = load_dimension("basic")
        self.dimension = dimension
        self.screen: Optional[pygame.Surface] = None
        self.ctx: Optional[RenderContext] = None
        self.lighting: Optional[LightingSetup] = None
        self.cam: dict[str, float] = {}
        self.slopes: list[dict[str, Any]] = []
        self.collision_world: Optional[CollisionWorld] = None
        self.state: Optional[PhysicsState] = None
        self.asset_directory = self.get_asset_directory()
        self.mesh_manager: Optional[MeshManager] = None
        self.world_objects: list[WorldObject] = []
        self.jump_velocity = self.get_default_jump_velocity()
        self.jump_default = self.jump_velocity
        self.fps = 0.0
        self.gravity = dimension.gravity
        self.clear_color = dimension.sky_color
        self.friction = dimension.base_friction
        self.ground_color = dimension.ground_color
        self.draw_grid = dimension.draw_grid

    def load(self) -> None:
        """リソースとシミュレーション状態を初期化する。"""

        self.screen = self.engine.screen
        size = (self.engine.config.width, self.engine.config.height)
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
        collision_world = self.create_collision_world(self.slopes)
        self.collision_world = collision_world
        state = self.create_physics_state()
        self.state = state
        self.jump_velocity = self.get_default_jump_velocity()
        self.jump_default = self.jump_velocity
        mesh_manager = self.create_mesh_manager(self.asset_directory)
        self.mesh_manager = mesh_manager
        self.world_objects = self.create_world_objects(mesh_manager)
        self.cam["y"] = ground_height(
            self.cam["x"],
            self.cam["z"],
            self.ground_y_base,
            collision_world,
        ) + self.eye_height
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

    def handle_event(self, event: pygame.event.Event) -> None:
        """入力イベントを処理する。"""

        if event.type == pygame.QUIT:
            self.engine.stop()
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.engine.stop()
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

    def update(self, dt: float) -> None:
        """タイムステップ分だけ物理状態を前進させる。"""

        if self.state is None or self.collision_world is None:
            raise RuntimeError("シーンが初期化されていない")
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
        """描画コンテキストを用いてシーンを表示する。"""

        if self.ctx is None or self.screen is None or self.state is None:
            raise RuntimeError("シーンが初期化されていない")
        self.fps = self.engine.fps
        self.screen.fill(self.clear_color)
        if self.draw_grid:
            draw_grid(
                self.ctx,
                self.cam,
                self.slopes,
                self.ground_y_base,
                self.grid_offset,
                self.grid_size,
                self.grid_range,
                self.ground_color,
            )
        else:
            draw_ground_fill(
                self.ctx,
                self.cam,
                self.slopes,
                self.ground_y_base,
                self.grid_offset,
                self.grid_size,
                self.grid_range,
                self.ground_color,
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

    def unload(self) -> None:
        """カーソル状態を戻しリソース参照を破棄する。"""

        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        self.screen = None
        self.ctx = None
        self.lighting = None
        self.collision_world = None
        self.state = None
        self.mesh_manager = None
        self.world_objects = []

    def get_asset_directory(self) -> Path:
        """アセットディレクトリのパスを返す。"""

        return Path(__file__).resolve().parent.parent / "assets"

    def create_camera(self) -> dict[str, float]:
        """カメラの初期状態を生成する。"""

        return {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
        }

    def create_slopes(self) -> list[dict[str, Any]]:
        """地形スロープ情報を生成する。"""

        return []

    def create_collision_world(self, slopes: list[dict[str, Any]]) -> CollisionWorld:
        """衝突判定用のワールドを構築する。"""

        return build_default_collision_world(self.ground_y_base, slopes)

    def create_physics_state(self) -> PhysicsState:
        """プレイヤー物理状態を生成する。"""

        return PhysicsState()

    def create_mesh_manager(self, asset_directory: Path) -> MeshManager:
        """メッシュリソースマネージャーを構築する。"""

        return MeshManager(asset_directory)

    def material_from_rgb(self, rgb: tuple[int, int, int], shininess: float, rim: float) -> Material:
        """RGB値からマテリアルを生成する。"""

        return Material(
            diffuse_color=tuple(max(0.0, min(1.0, c / 255.0)) for c in rgb),
            specular_color=(0.9, 0.9, 0.9),
            shininess=shininess,
            ambient_factor=1.0,
            rim_strength=rim,
        )

    def create_world_objects(self, mesh_manager: MeshManager) -> list[WorldObject]:
        """シーン内に配置するオブジェクトを構築する。"""

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
        """ライティング構成を生成する。"""

        directional_lights = [
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
        ]
        if self.dimension.identifier == "basic":
            directional_lights.insert(
                0,
                DirectionalLight(
                    direction=(-0.35, -0.9, -0.18),
                    color=(1.0, 0.94, 0.82),
                    intensity=1.15,
                    specular_intensity=0.55,
                ),
            )
        return LightingSetup(
            ambient_color=(0.18, 0.19, 0.22),
            directional_lights=directional_lights,
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

    def get_default_jump_velocity(self) -> float:
        """ジャンプ初速度の既定値を返す。"""

        return self.default_jump_velocity


class GameApplication:
    """エンジンとシーンをまとめて起動するためのヘルパー。"""

    width: int = 1280
    height: int = 720
    window_title: str = "Futile - Commented"
    target_fps: int = 60
    time_step: float = 0.0
    config_class: Type[EngineConfig] = EngineConfig
    scene_class: Type[Scene] = DefaultScene
    dimension_name: str = "basic"

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        if config is None:
            config = self.create_config()
        self.config = config
        self.engine = self.create_engine(self.config)
        self.dimension = self.create_dimension()
        self.scene = self.create_scene(self.engine, self.dimension)

    def create_config(self) -> EngineConfig:
        """アプリケーションに必要なエンジン設定を構築する。"""

        return self.config_class(
            width=self.width,
            height=self.height,
            window_title=self.window_title,
            target_fps=self.target_fps,
            time_step=self.time_step,
        )

    def create_engine(self, config: EngineConfig) -> Engine:
        """設定を基にエンジンを生成する。"""

        return Engine(config)

    def create_dimension(self) -> DimensionSettings:
        """選択されたディメンション設定を読み込む。"""

        return load_dimension(self.dimension_name)

    def create_scene(self, engine: Engine, dimension: DimensionSettings) -> Scene:
        """ゲームで使用するシーンを組み立てる。"""

        return self.scene_class(engine, dimension=dimension)

    def run(self) -> None:
        """シーンをエンジンに登録して実行する。"""

        self.engine.set_scene(self.scene)
        self.engine.run()
