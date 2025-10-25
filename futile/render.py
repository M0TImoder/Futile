import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import pygame

from .physics import PhysicsState, ground_height
from .world import Material, MeshGeometry, WorldObject


_NEAR_CLIP: float = 0.05
_CLIP_EPS: float = 1e-4


@dataclass
class DirectionalLight:
    # 平行光の方向と色を保持する
    direction: Tuple[float, float, float]
    color: Tuple[float, float, float]
    intensity: float = 1.0
    specular_intensity: float = 0.0


@dataclass
class PointLight:
    # 点光源の位置と減衰を保持する
    position: Tuple[float, float, float]
    color: Tuple[float, float, float]
    intensity: float = 1.0
    radius: float = 0.0
    attenuation: float = 0.0
    specular_intensity: float = 0.0
    _inv_radius: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        if self.radius > 0.0:
            self._inv_radius = 1.0 / self.radius
        else:
            self._inv_radius = 0.0


@dataclass
class LightingSetup:
    # シーン全体の環境光とライトを保持する
    ambient_color: Tuple[float, float, float]
    directional_lights: List[DirectionalLight] = field(default_factory=list)
    point_lights: List[PointLight] = field(default_factory=list)
    rim_intensity: float = 0.05


@dataclass
class RenderContext:
    screen: pygame.Surface
    cx: float
    cy: float
    fov_d: float
    lighting: LightingSetup | None = None


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


def _clip_polygon_near(
    vertices: Sequence[Tuple[float, float, float]],
    near: float = _NEAR_CLIP,
) -> List[Tuple[float, float, float]]:
    # 近クリップ面でポリゴンを切り詰める
    if not vertices:
        return []
    clipped: List[Tuple[float, float, float]] = []
    prev = vertices[-1]
    prev_inside = prev[2] > near
    for curr in vertices:
        curr_inside = curr[2] > near
        if prev_inside != curr_inside:
            denom = curr[2] - prev[2]
            if abs(denom) < 1e-8:
                ix = curr[0]
                iy = curr[1]
            else:
                t = (near - prev[2]) / denom
                ix = prev[0] + (curr[0] - prev[0]) * t
                iy = prev[1] + (curr[1] - prev[1]) * t
            clipped.append((ix, iy, near + _CLIP_EPS))
        if curr_inside:
            clipped.append((curr[0], curr[1], curr[2]))
        prev = curr
        prev_inside = curr_inside
    return clipped


def draw_grid(
    ctx: RenderContext,
    cam: Dict[str, float],
    slopes: List[Dict[str, float]],
    ground_y_base: float,
    grid_off: float,
    g_size: int,
    g_range: int,
    grid_color: Tuple[int, int, int],
) -> None:
    sx = int(cam["x"] // g_size) * g_size - g_range
    ex = int(cam["x"] // g_size) * g_size + g_range
    sz = int(cam["z"] // g_size) * g_size - g_range
    ez = int(cam["z"] // g_size) * g_size + g_range
    col = grid_color
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


def draw_ground_fill(
    ctx: RenderContext,
    cam: Dict[str, float],
    slopes: List[Dict[str, float]],
    ground_y_base: float,
    grid_off: float,
    g_size: int,
    g_range: int,
    ground_color: Tuple[int, int, int],
) -> None:
    sx = int(cam["x"] // g_size) * g_size - g_range
    ex = int(cam["x"] // g_size) * g_size + g_range
    sz = int(cam["z"] // g_size) * g_size - g_range
    ez = int(cam["z"] // g_size) * g_size + g_range

    near_bridge: Dict[Tuple[float, float], Tuple[float, float]] = {}
    # 近クリップに接する頂点を集めて足元を補間する
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

            corners = (
                (x, y1, z),
                (x + g_size, y2, z),
                (x + g_size, y3, z + g_size),
                (x, y4, z + g_size),
            )
            cam_poly: List[Tuple[float, float, float]] = []
            for vx_, vy_, vz_ in corners:
                px = vx_ - cam["x"]
                py = vy_ - cam["y"]
                pz = vz_ - cam["z"]
                cam_poly.append(rot_cam(px, py, pz, cam))
            clipped = _clip_polygon_near(cam_poly)
            if len(clipped) < 3:
                continue
            pts: List[Tuple[float, float]] = []
            for rx, ry, rz in clipped:
                screen_pt = proj(ctx, rx, ry, rz)
                pts.append(screen_pt)
                if rz <= _NEAR_CLIP + 1e-3:
                    key = (round(screen_pt[0], 2), round(screen_pt[1], 2))
                    near_bridge[key] = screen_pt
            pygame.draw.polygon(ctx.screen, ground_color, pts)

    if near_bridge:
        height = ctx.screen.get_height()
        bridge_pts = sorted(near_bridge.values(), key=lambda pt: pt[0])
        if len(bridge_pts) >= 2:
            left_x = bridge_pts[0][0]
            right_x = bridge_pts[-1][0]
        else:
            left_x = right_x = bridge_pts[0][0]
        clip_poly = [
            (left_x, float(height)),
            *bridge_pts,
            (right_x, float(height)),
        ]
        pygame.draw.polygon(ctx.screen, ground_color, clip_poly)


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
    lighting: LightingSetup | None = None,
    enable_wire: bool = False,
) -> None:
    if lighting is None:
        lighting = ctx.lighting
    if lighting is None:
        lighting = _default_lighting_setup()
    cam_pos = (cam["x"], cam["y"], cam["z"])
    visible_objects = [obj for obj in objects if obj.visible]
    visible_objects.sort(key=lambda obj: obj.distance_to(cam_pos), reverse=True)
    for obj in visible_objects:
        geometry = obj.mesh.select_geometry(obj.distance_to(cam_pos))
        _render_object(ctx, cam, obj, geometry, lighting, enable_wire)


def _render_object(
    ctx: RenderContext,
    cam: Dict[str, float],
    obj: WorldObject,
    geometry: MeshGeometry,
    lighting: LightingSetup,
    enable_wire: bool,
) -> None:
    world_vertices = list(obj.world_vertices(geometry))
    cam_vertices: List[Tuple[float, float, float]] = []
    cam_pos = (cam["x"], cam["y"], cam["z"])
    for vx, vy, vz in world_vertices:
        px = vx - cam["x"]
        py = vy - cam["y"]
        pz = vz - cam["z"]
        cam_vertices.append(rot_cam(px, py, pz, cam))

    for a, b, c in geometry.indices:
        v0 = cam_vertices[a]
        v1 = cam_vertices[b]
        v2 = cam_vertices[c]
        if v0[2] <= _NEAR_CLIP and v1[2] <= _NEAR_CLIP and v2[2] <= _NEAR_CLIP:
            continue
        normal_cam = _triangle_normal(v0, v1, v2)
        tri_center_cam = (
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0,
        )
        view_vec = (
            -tri_center_cam[0],
            -tri_center_cam[1],
            -tri_center_cam[2],
        )
        # カメラ視点で表面の向きを確認
        facing = (
            normal_cam[0] * view_vec[0]
            + normal_cam[1] * view_vec[1]
            + normal_cam[2] * view_vec[2]
        )
        if facing <= 0.0:
            continue
        # 法線からライティングを決定
        w0 = world_vertices[a]
        w1 = world_vertices[b]
        w2 = world_vertices[c]
        normal_world = _triangle_normal(w0, w1, w2)
        tri_center = (
            (w0[0] + w1[0] + w2[0]) / 3.0,
            (w0[1] + w1[1] + w2[1]) / 3.0,
            (w0[2] + w1[2] + w2[2]) / 3.0,
        )
        material = obj.material
        if material is None or lighting is None:
            shade = _shade_legacy(normal_cam, v0, w0, obj.color, lighting or _default_lighting_setup())
        else:
            shade = _shade(normal_world, tri_center, cam_pos, material, lighting)
        clipped_tri = _clip_polygon_near((v0, v1, v2))
        if len(clipped_tri) < 3:
            continue
        pts2d = [proj(ctx, vx, vy, vz) for vx, vy, vz in clipped_tri]
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


def _shade(
    normal_world: Tuple[float, float, float],
    tri_center_world: Tuple[float, float, float],
    cam_pos: Tuple[float, float, float],
    material: Material,
    lighting: LightingSetup,
) -> Tuple[int, int, int]:
    lighting = lighting or _default_lighting_setup()
    normal = _normalize(normal_world)
    if normal == (0.0, 0.0, 0.0):
        return tuple(int(max(0.0, min(1.0, c)) * 255.0) for c in material.diffuse_color)
    view_dir = _normalize(
        (
            cam_pos[0] - tri_center_world[0],
            cam_pos[1] - tri_center_world[1],
            cam_pos[2] - tri_center_world[2],
        )
    )
    # リムライトの視線依存成分を算出
    rim_view = max(0.0, 1.0 - max(0.0, normal[0] * view_dir[0] + normal[1] * view_dir[1] + normal[2] * view_dir[2]))
    ambient = [
        lighting.ambient_color[i] * material.ambient_factor * material.diffuse_color[i]
        for i in range(3)
    ]
    diffuse = [0.0, 0.0, 0.0]
    specular = [0.0, 0.0, 0.0]
    shininess = max(0.0, material.shininess)
    for light in lighting.directional_lights:
        light_dir = _normalize(light.direction)
        lambert = max(0.0, normal[0] * light_dir[0] + normal[1] * light_dir[1] + normal[2] * light_dir[2])
        lambert *= light.intensity
        if lambert > 0.0:
            for i in range(3):
                diffuse[i] += lambert * light.color[i] * material.diffuse_color[i]
            if light.specular_intensity > 0.0 and shininess > 0.0:
                half_vec = _normalize(
                    (
                        light_dir[0] + view_dir[0],
                        light_dir[1] + view_dir[1],
                        light_dir[2] + view_dir[2],
                    )
                )
                if half_vec != (0.0, 0.0, 0.0):
                    spec_power = max(
                        0.0,
                        normal[0] * half_vec[0]
                        + normal[1] * half_vec[1]
                        + normal[2] * half_vec[2],
                    )
                    spec = (spec_power ** shininess) * light.specular_intensity
                    for i in range(3):
                        specular[i] += spec * light.color[i] * material.specular_color[i]
    for light in lighting.point_lights:
        to_light = (
            light.position[0] - tri_center_world[0],
            light.position[1] - tri_center_world[1],
            light.position[2] - tri_center_world[2],
        )
        dist_sq = to_light[0] * to_light[0] + to_light[1] * to_light[1] + to_light[2] * to_light[2]
        if dist_sq == 0.0:
            continue
        dist = math.sqrt(dist_sq)
        light_dir = (to_light[0] / dist, to_light[1] / dist, to_light[2] / dist)
        range_factor = 1.0
        if light.radius > 0.0 and light._inv_radius > 0.0:
            # 半径に応じて線形減衰を計算
            range_factor = max(0.0, 1.0 - dist * light._inv_radius)
            range_factor *= range_factor
        lambert = max(0.0, normal[0] * light_dir[0] + normal[1] * light_dir[1] + normal[2] * light_dir[2])
        atten_mul = light.attenuation if light.attenuation > 0.0 else 1.0
        lambert *= light.intensity * range_factor * atten_mul
        if lambert > 0.0:
            for i in range(3):
                diffuse[i] += lambert * light.color[i] * material.diffuse_color[i]
            if light.specular_intensity > 0.0 and shininess > 0.0:
                half_vec = _normalize(
                    (
                        light_dir[0] + view_dir[0],
                        light_dir[1] + view_dir[1],
                        light_dir[2] + view_dir[2],
                    )
                )
                if half_vec != (0.0, 0.0, 0.0):
                    spec_power = max(
                        0.0,
                        normal[0] * half_vec[0]
                        + normal[1] * half_vec[1]
                        + normal[2] * half_vec[2],
                    )
                    spec = (spec_power ** shininess) * light.specular_intensity * range_factor * atten_mul
                    for i in range(3):
                        specular[i] += spec * light.color[i] * material.specular_color[i]
    rim_component = lighting.rim_intensity * material.rim_strength * rim_view
    shaded = []
    for i in range(3):
        intensity = ambient[i] + diffuse[i] + specular[i] + rim_component * material.diffuse_color[i]
        intensity = max(0.0, min(1.0, intensity))
        shaded.append(int(intensity * 255.0))
    return tuple(shaded)


def _shade_legacy(
    normal: Tuple[float, float, float],
    view_point: Tuple[float, float, float],
    world_point: Tuple[float, float, float],
    color: Tuple[int, int, int],
    lighting: LightingSetup,
) -> Tuple[int, int, int]:
    nx, ny, nz = normal
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0.0:
        return color
    nx /= length
    ny /= length
    nz /= length
    view = _normalize((-view_point[0], -view_point[1], -view_point[2]))
    rim = max(0.0, 1.0 - max(0.0, nz))
    diffuse = [0.0, 0.0, 0.0]
    specular = [0.0, 0.0, 0.0]
    for light in lighting.directional_lights:
        light_dir = _normalize(light.direction)
        lambert = max(0.0, nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2])
        lambert *= light.intensity
        reflect = _reflect((-light_dir[0], -light_dir[1], -light_dir[2]), (nx, ny, nz))
        spec = max(0.0, reflect[0] * view[0] + reflect[1] * view[1] + reflect[2] * view[2]) ** 16
        spec *= light.specular_intensity
        for i in range(3):
            diffuse[i] += lambert * light.color[i]
            specular[i] += spec * light.color[i]
    for light in lighting.point_lights:
        to_light = (
            light.position[0] - world_point[0],
            light.position[1] - world_point[1],
            light.position[2] - world_point[2],
        )
        dist = math.sqrt(to_light[0] * to_light[0] + to_light[1] * to_light[1] + to_light[2] * to_light[2])
        if dist == 0.0:
            continue
        light_dir = (to_light[0] / dist, to_light[1] / dist, to_light[2] / dist)
        atten = 1.0
        if light.attenuation > 0.0:
            atten = 1.0 / (1.0 + light.attenuation * dist * dist)
        lambert = max(0.0, nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2])
        lambert *= light.intensity * atten
        reflect = _reflect((-light_dir[0], -light_dir[1], -light_dir[2]), (nx, ny, nz))
        spec = max(0.0, reflect[0] * view[0] + reflect[1] * view[1] + reflect[2] * view[2]) ** 16
        spec *= light.specular_intensity * atten
        for i in range(3):
            diffuse[i] += lambert * light.color[i]
            specular[i] += spec * light.color[i]
    rim_component = rim * lighting.rim_intensity
    shaded = []
    for i, channel in enumerate(color):
        intensity = lighting.ambient_color[i]
        intensity += diffuse[i] + specular[i] + rim_component
        intensity = max(0.0, min(1.0, intensity))
        shaded.append(min(255, max(0, int(channel * intensity))))
    return tuple(shaded)


def _default_lighting_setup() -> LightingSetup:
    return LightingSetup(
        ambient_color=(0.2, 0.2, 0.2),
        directional_lights=[
            DirectionalLight(
                direction=(0.3, 0.8, 0.5),
                color=(1.0, 1.0, 1.0),
                intensity=0.6,
                specular_intensity=0.15,
            )
        ],
        rim_intensity=0.05,
    )


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
    "DirectionalLight",
    "PointLight",
    "LightingSetup",
    "RenderContext",
    "draw_grid",
    "draw_debug",
    "proj",
    "rot_cam",
    "render_scene",
]
