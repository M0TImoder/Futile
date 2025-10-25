"""Basic gameplay dimension configuration."""

from futile.world import DimensionSettings

DIMENSION = DimensionSettings(
    identifier="basic",
    display_name="Basic",
    gravity=-300.0,
    sky_color=(15, 15, 20),
    ground_color=(70, 70, 70),
    base_friction=8.0,
)
