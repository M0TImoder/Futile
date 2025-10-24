import math
from typing import Dict, Tuple

import pygame


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def read_input(cam: Dict[str, float], m_sens: float, pitch_min: float, pitch_max: float) -> Tuple[float, float, bool, bool]:
    try:
        mx, my = pygame.mouse.get_rel()
    except Exception:
        mx, my = 0, 0

    cam["yaw"] += mx * m_sens
    cam["pitch"] -= my * m_sens
    cam["pitch"] = clamp(cam["pitch"], pitch_min, pitch_max)

    keys = pygame.key.get_pressed()
    yaw = cam["yaw"]
    fx = math.sin(yaw)
    fz = math.cos(yaw)
    rx = math.cos(yaw)
    rz = -math.sin(yaw)

    mvx = 0.0
    mvz = 0.0

    if keys[pygame.K_w]:
        mvx += fx
        mvz += fz
    if keys[pygame.K_s]:
        mvx -= fx
        mvz -= fz
    if keys[pygame.K_a]:
        mvx -= rx
        mvz -= rz
    if keys[pygame.K_d]:
        mvx += rx
        mvz += rz

    dash = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
    jump = keys[pygame.K_SPACE]

    return mvx, mvz, dash, jump
