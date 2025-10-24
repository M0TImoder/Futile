import math
from dataclasses import dataclass
from typing import Dict, List

import pygame

from .physics import PhysicsState, ground_height


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
