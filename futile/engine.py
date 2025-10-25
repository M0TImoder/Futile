from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pygame


@dataclass(slots=True)
class EngineConfig:
    """レンダリングと更新処理に用いる基本設定を保持する。"""

    width: int = 1280
    height: int = 720
    window_title: str = "Futile - Commented"
    target_fps: int = 60
    time_step: float = 0.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("widthとheightは正の整数でなければならない")
        if self.target_fps <= 0:
            raise ValueError("target_fpsは正の整数でなければならない")
        if self.time_step <= 0.0:
            self.time_step = 1.0 / float(self.target_fps)


class Scene:
    """ゲーム内シーンのライフサイクルを定義する基本クラス。"""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def load(self) -> None:
        """リソースを読み込み初期化を行う。"""

    def handle_event(self, event: pygame.event.Event) -> None:
        """イベントを処理する。"""

    def update(self, dt: float) -> None:
        """固定タイムステップで状態を更新する。"""

    def render(self) -> None:
        """描画処理を実行する。"""

    def unload(self) -> None:
        """保持しているリソースを解放する。"""


class Engine:
    """シーンを管理しゲームループを駆動する。"""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._scene: Optional[Scene] = None
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._running = False
        self._fps = 0.0

    @property
    def screen(self) -> pygame.Surface:
        """描画先の画面サーフェスを取得する。"""

        if self._screen is None:
            raise RuntimeError("画面サーフェスはまだ初期化されていない")
        return self._screen

    @property
    def fps(self) -> float:
        """直近フレームの実効FPSを返す。"""

        return self._fps

    def set_scene(self, scene: Scene) -> None:
        """現在のシーンを差し替える。"""

        self._scene = scene

    def stop(self) -> None:
        """メインループを停止する。"""

        self._running = False

    def run(self) -> None:
        """メインループを開始し、シーンに制御を委譲する。"""

        if self._scene is None:
            raise RuntimeError("シーンが設定されていない")
        pygame.init()
        try:
            self._screen = pygame.display.set_mode((self.config.width, self.config.height))
            pygame.display.set_caption(self.config.window_title)
            self._clock = pygame.time.Clock()
            self._running = True
            accumulator = 0.0
            self._scene.load()
            while self._running:
                assert self._clock is not None
                ms = self._clock.tick(self.config.target_fps)
                frame_time = ms / 1000.0
                accumulator += frame_time
                max_accumulator = self.config.time_step * 5.0
                if accumulator > max_accumulator:
                    accumulator = max_accumulator
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.stop()
                    self._scene.handle_event(event)
                    if not self._running:
                        break
                if not self._running:
                    break
                while accumulator >= self.config.time_step:
                    self._scene.update(self.config.time_step)
                    accumulator -= self.config.time_step
                if accumulator > 0.0:
                    self._scene.update(accumulator)
                    accumulator = 0.0
                self._scene.render()
                pygame.display.flip()
                self._fps = float(self._clock.get_fps())
        finally:
            try:
                if self._scene is not None:
                    self._scene.unload()
            finally:
                pygame.quit()
                self._screen = None
                self._clock = None
                self._running = False
                self._fps = 0.0
