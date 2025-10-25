from __future__ import annotations

from futile import EngineConfig, GameApplication


class FutileApplication(GameApplication):
    """デフォルトシーンを起動するサンプルアプリケーション。"""

    window_title = "Futile - Commented"
    dimension_name = "basic"


def main() -> None:
    """エンジン設定を構築しアプリケーションを実行する。"""

    config = EngineConfig(
        width=FutileApplication.width,
        height=FutileApplication.height,
        window_title=FutileApplication.window_title,
        target_fps=FutileApplication.target_fps,
        time_step=FutileApplication.time_step,
    )
    app = FutileApplication(config=config)
    app.run()


if __name__ == "__main__":
    main()
