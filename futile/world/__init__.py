"""World object structuresとディメンション設定をまとめたモジュール。"""

from __future__ import annotations

import math
import pkgutil
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

Vector3 = Tuple[float, float, float]
ColorRGB = Tuple[int, int, int]


@dataclass(frozen=True)
class DimensionSettings:
    identifier: str
    display_name: str
    gravity: float
    sky_color: ColorRGB
    ground_color: ColorRGB
    base_friction: float
    draw_grid: bool = True


@dataclass
class Material:
    diffuse_color: Tuple[float, float, float]
    specular_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    shininess: float = 16.0
    ambient_factor: float = 1.0
    rim_strength: float = 1.0


@dataclass
class MeshGeometry:
    vertices: List[Vector3]
    normals: List[Vector3]
    indices: List[Tuple[int, int, int]]


@dataclass(order=True)
class MeshLOD:
    max_distance: float
    geometry: MeshGeometry = field(compare=False)


@dataclass
class Mesh:
    lod_levels: List[MeshLOD]
    name: str = ""

    def __post_init__(self) -> None:
        self.lod_levels.sort(key=lambda lod: lod.max_distance)
        self.bounding_radius = self._compute_radius()

    def _compute_radius(self) -> float:
        radius = 0.0
        for lod in self.lod_levels:
            for vx, vy, vz in lod.geometry.vertices:
                radius = max(radius, math.sqrt(vx * vx + vy * vy + vz * vz))
        return radius

    def select_geometry(self, distance: float) -> MeshGeometry:
        for lod in self.lod_levels:
            if distance <= lod.max_distance:
                return lod.geometry
        return self.lod_levels[-1].geometry


@dataclass
class Collider:
    radius: float


@dataclass
class WorldObject:
    mesh: Mesh
    position: Vector3
    rotation: Vector3
    scale: Vector3 | float
    color: Tuple[int, int, int] = (170, 170, 190)
    material: Material | None = None
    collider: Collider | None = None
    visible: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.scale, (int, float)):
            self.scale = (float(self.scale), float(self.scale), float(self.scale))
        else:
            self.scale = tuple(float(v) for v in self.scale)
        if self.material is None:
            base = tuple(max(0.0, min(1.0, c / 255.0)) for c in self.color)
            self.material = Material(diffuse_color=base)
        else:
            diffuse = tuple(max(0.0, min(1.0, channel)) for channel in self.material.diffuse_color)
            specular = tuple(max(0.0, min(1.0, channel)) for channel in self.material.specular_color)
            self.material = Material(
                diffuse_color=diffuse,
                specular_color=specular,
                shininess=self.material.shininess,
                ambient_factor=self.material.ambient_factor,
                rim_strength=self.material.rim_strength,
            )
            self.color = tuple(int(channel * 255.0) for channel in diffuse)
        if self.collider is None:
            largest_scale = max(self.scale)
            self.collider = Collider(radius=self.mesh.bounding_radius * largest_scale)

    def world_vertices(self, geometry: MeshGeometry) -> Iterable[Vector3]:
        sx, sy, sz = self.scale
        pitch, yaw, roll = self.rotation
        sin_pitch = math.sin(pitch)
        cos_pitch = math.cos(pitch)
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)
        sin_roll = math.sin(roll)
        cos_roll = math.cos(roll)
        # オブジェクトの姿勢を適用
        for vx, vy, vz in geometry.vertices:
            x = vx * sx
            y = vy * sy
            z = vz * sz
            rx = cos_yaw * x + sin_yaw * z
            rz = -sin_yaw * x + cos_yaw * z
            ry = y
            ty = cos_pitch * ry - sin_pitch * rz
            tz = sin_pitch * ry + cos_pitch * rz
            fx = cos_roll * rx - sin_roll * ty
            fy = sin_roll * rx + cos_roll * ty
            yield (
                fx + self.position[0],
                fy + self.position[1],
                tz + self.position[2],
            )

    def distance_to(self, point: Sequence[float]) -> float:
        dx = self.position[0] - point[0]
        dy = self.position[1] - point[1]
        dz = self.position[2] - point[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)


_DIMENSION_CACHE: Dict[str, DimensionSettings] = {}


def _dimension_module_path(name: str) -> str:
    return f"{__name__}.dimensions.{name}"


def load_dimension(name: str) -> DimensionSettings:
    if name not in _DIMENSION_CACHE:
        module = import_module(_dimension_module_path(name))
        settings = getattr(module, "DIMENSION", None)
        if not isinstance(settings, DimensionSettings):
            raise ValueError(f"ディメンション'{name}'の設定が正しく定義されていない")
        if settings.identifier != name:
            raise ValueError("ディメンション識別子がモジュール名と一致していない")
        _DIMENSION_CACHE[name] = settings
    return _DIMENSION_CACHE[name]


def available_dimensions() -> Mapping[str, DimensionSettings]:
    package = import_module(f"{__name__}.dimensions")
    discovered: Dict[str, DimensionSettings] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        name = module_info.name
        try:
            discovered[name] = load_dimension(name)
        except Exception:
            continue
    return dict(discovered)
