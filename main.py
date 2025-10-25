from futile import GameApplication


class FutileApplication(GameApplication):
    window_title = "Futile - Commented"


def main() -> None:
    app = FutileApplication()
    app.run()


if __name__ == "__main__":
    main()
