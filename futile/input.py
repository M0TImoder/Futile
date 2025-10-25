import math
from dataclasses import dataclass
from typing import Dict, Tuple

import pygame


@dataclass(frozen=True)
class InputBindings:
    forward: int = pygame.K_w
    backward: int = pygame.K_s
    left: int = pygame.K_a
    right: int = pygame.K_d
    dash: Tuple[int, ...] = (pygame.K_LSHIFT, pygame.K_RSHIFT)
    jump: Tuple[int, ...] = (pygame.K_SPACE,)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def read_input(
    cam: Dict[str, float],
    m_sens: float,
    pitch_min: float,
    pitch_max: float,
    bindings: InputBindings = InputBindings(),
) -> Tuple[float, float, bool, bool]:
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

    if keys[bindings.forward]:
        mvx += fx
        mvz += fz
    if keys[bindings.backward]:
        mvx -= fx
        mvz -= fz
    if keys[bindings.left]:
        mvx -= rx
        mvz -= rz
    if keys[bindings.right]:
        mvx += rx
        mvz += rz

    dash = any(keys[key_code] for key_code in bindings.dash)
    jump = any(keys[key_code] for key_code in bindings.jump)

    return mvx, mvz, dash, jump
