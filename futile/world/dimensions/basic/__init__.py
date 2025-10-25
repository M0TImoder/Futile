"""Basic gameplay dimension configuration."""

from futile.world import DimensionSettings

DIMENSION = DimensionSettings(
    identifier="basic",
    display_name="Basic",
    gravity=-300.0,
    sky_color=(135, 196, 255),
    ground_color=(120, 96, 72),
    base_friction=8.0,
    draw_grid=False,
)
