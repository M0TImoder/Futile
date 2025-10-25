"""Alternative test dimension configuration."""

from futile.world import DimensionSettings

DIMENSION = DimensionSettings(
    identifier="test_dimension",
    display_name="Test Dimension",
    gravity=-120.0,
    sky_color=(30, 60, 110),
    ground_color=(40, 90, 80),
    base_friction=5.0,
)
