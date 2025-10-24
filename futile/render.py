import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pygame

from .physics import PhysicsState, ground_height
from .world import MeshGeometry, WorldObject


@dataclass
class RenderContext:
    screen: pygame.Surface
    cx: float
    cy: float
    fov_d: float


def proj(ctx: RenderContext, px: float, py: float, pz: float) -> tuple[float, float]:
    pz_clip = max(pz, 0.05)
    factor = ctx.fov_d / (pz_clip + ctx.fov_d)
    return px * factor + ctx.cx, -py * factor + ctx.cy


def rot_cam(px: float, py: float, pz: float, cam: Dict[str, float]) -> tuple[float, float, float]:
    yaw = cam["yaw"]
    pitch = cam["pitch"]
    rx = px * math.cos(yaw) - pz * math.sin(yaw)
    rz = px * math.sin(yaw) + pz * math.cos(yaw)
    ry = py * math.cos(pitch) - rz * math.sin(pitch)
    zz = py * math.sin(pitch) + rz * math.cos(pitch)
    return rx, ry, zz


def draw_grid(
    ctx: RenderContext,
    cam: Dict[str, float],
    slopes: List[Dict[str, float]],
    ground_y_base: float,
    grid_off: float,
    g_size: int,
    g_range: int,
) -> None:
    sx = int(cam["x"] // g_size) * g_size - g_range
    ex = int(cam["x"] // g_size) * g_size + g_range
    sz = int(cam["z"] // g_size) * g_size - g_range
    ez = int(cam["z"] // g_size) * g_size + g_range
    col = (70, 70, 70)
    inner_r = 200

    for x in range(sx, ex + g_size, g_size):
        for z in range(sz, ez + g_size, g_size):
            dxp = x - cam["x"]
            dzp = z - cam["z"]
            dist = math.hypot(dxp, dzp)
            if dist > g_range:
                continue

            y1 = ground_height(x, z, ground_y_base, slopes) + grid_off
            y2 = ground_height(x + g_size, z, ground_y_base, slopes) + grid_off
            y3 = ground_height(x + g_size, z + g_size, ground_y_base, slopes) + grid_off
            y4 = ground_height(x, z + g_size, ground_y_base, slopes) + grid_off

            pts = []
            corners = (
                (x, y1, z),
                (x + g_size, y2, z),
                (x + g_size, y3, z + g_size),
                (x, y4, z + g_size),
            )
            for vx_, vy_, vz_ in corners:
                px = vx_ - cam["x"]
                py = vy_ - cam["y"]
                pz = vz_ - cam["z"]
                rx, ry, rz = rot_cam(px, py, pz, cam)
                if dist < inner_r and rz <= 0.05:
                    rz = 0.05
                if rz > 0.05:
                    pts.append(proj(ctx, rx, ry, rz))
            if len(pts) == 4:
                pygame.draw.lines(ctx.screen, col, True, pts, 1)


def draw_debug(
    ctx: RenderContext,
    cam: Dict[str, float],
    state: PhysicsState,
    fps: float,
    jump_v: float,
    grav: float,
    slopes: List[Dict[str, float]],
) -> None:
    font = pygame.font.SysFont("consolas", 18)
    info = (
        f"pos: ({cam['x']:.1f},{cam['y']:.1f},{cam['z']:.1f}) "
        f"yaw:{math.degrees(cam['yaw']):.1f} pitch:{math.degrees(cam['pitch']):.1f} "
        f"on_g:{state.on_ground} vx:{state.vx:.1f} vz:{state.vz:.1f} fps:{int(fps)} "
        f"slopes:{len(slopes)} jump_v:{jump_v:.1f} grav:{grav:.1f}"
    )
    ctx.screen.blit(font.render(info, True, (200, 200, 200)), (10, 10))


def render_scene(
    ctx: RenderContext,
    cam: Dict[str, float],
    objects: Sequence[WorldObject],
    enable_wire: bool = False,
) -> None:
    cam_pos = (cam["x"], cam["y"], cam["z"])
    visible_objects = [obj for obj in objects if obj.visible]
    visible_objects.sort(key=lambda obj: obj.distance_to(cam_pos), reverse=True)
    for obj in visible_objects:
        geometry = obj.mesh.select_geometry(obj.distance_to(cam_pos))
        _render_object(ctx, cam, obj, geometry, enable_wire)


def _render_object(
    ctx: RenderContext,
    cam: Dict[str, float],
    obj: WorldObject,
    geometry: MeshGeometry,
    enable_wire: bool,
) -> None:
    world_vertices = list(obj.world_vertices(geometry))
    cam_vertices: List[Tuple[float, float, float]] = []
    for vx, vy, vz in world_vertices:
        px = vx - cam["x"]
        py = vy - cam["y"]
        pz = vz - cam["z"]
        cam_vertices.append(rot_cam(px, py, pz, cam))

    for a, b, c in geometry.indices:
        v0 = cam_vertices[a]
        v1 = cam_vertices[b]
        v2 = cam_vertices[c]
        if v0[2] <= 0.05 and v1[2] <= 0.05 and v2[2] <= 0.05:
            continue
        # カメラ空間での裏面判定
        normal = _triangle_normal(v0, v1, v2)
        if normal[2] >= 0.0:
            continue
        # 法線からライティングを決定
        shade = _shade(normal, v0, obj.color)
        pts2d = [proj(ctx, *v) for v in (v0, v1, v2)]
        pygame.draw.polygon(ctx.screen, shade, pts2d)
        if enable_wire:
            pygame.draw.lines(ctx.screen, (40, 40, 40), True, pts2d, 1)


def _triangle_normal(v0: Tuple[float, float, float], v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    edge1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    edge2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
    return (
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
    )


def _shade(normal: Tuple[float, float, float], view_point: Tuple[float, float, float], color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    nx, ny, nz = normal
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0.0:
        return color
    nx /= length
    ny /= length
    nz /= length
    light = (0.3, 0.8, 0.5)
    lambert = max(0.0, min(1.0, nx * light[0] + ny * light[1] + nz * light[2]))
    view = _normalize((-view_point[0], -view_point[1], -view_point[2]))
    reflect = _reflect((-light[0], -light[1], -light[2]), (nx, ny, nz))
    # レイトレーシング風の反射寄与
    spec = max(0.0, reflect[0] * view[0] + reflect[1] * view[1] + reflect[2] * view[2]) ** 16
    rim = max(0.0, 1.0 - max(0.0, nz))
    # リムライトで輪郭を強調
    intensity = min(1.0, 0.2 + 0.6 * lambert + 0.15 * spec + 0.05 * rim)
    return tuple(min(255, max(0, int(channel * intensity))) for channel in color)


def _normalize(vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    vx, vy, vz = vec
    length = math.sqrt(vx * vx + vy * vy + vz * vz)
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return vx / length, vy / length, vz / length


def _reflect(light: Tuple[float, float, float], normal: Tuple[float, float, float]) -> Tuple[float, float, float]:
    dot = light[0] * normal[0] + light[1] * normal[1] + light[2] * normal[2]
    return (
        light[0] - 2.0 * dot * normal[0],
        light[1] - 2.0 * dot * normal[1],
        light[2] - 2.0 * dot * normal[2],
    )


__all__ = [
    "RenderContext",
    "draw_grid",
    "draw_debug",
    "proj",
    "rot_cam",
    "render_scene",
]
