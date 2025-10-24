import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PhysicsState:
    vx: float = 0.0
    vz: float = 0.0
    vel_y: float = 0.0
    on_ground: bool = True


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def ground_height(x: float, z: float, ground_y_base: float, slopes: List[Dict[str, float]]) -> float:
    y = ground_y_base
    for slope in slopes:
        dx = x - slope["x"]
        dz = z - slope["z"]
        if abs(dx) < slope["w"] * 0.5 and abs(dz) < slope["d"] * 0.5:
            y += dx * slope["sx"] + dz * slope["sz"]
    return y


def handle_movement(
    cam: Dict[str, float],
    state: PhysicsState,
    mvx: float,
    mvz: float,
    dash: bool,
    jump: bool,
    dt: float,
    spd: float,
    dash_mul: float,
    acc: float,
    fric: float,
    slopes: List[Dict[str, float]],
    eye_h: float,
    max_step: float,
    ground_y_base: float,
    jump_v: float,
) -> None:
    mag = math.hypot(mvx, mvz)
    desire_x = 0.0
    desire_z = 0.0

    if mag > 0.001:
        desire_x = mvx / mag
        desire_z = mvz / mag

    if jump and state.on_ground:
        state.vel_y = jump_v
        state.on_ground = False

    target_speed = spd * (dash_mul if dash else 1.0)
    tar_vx = desire_x * target_speed
    tar_vz = desire_z * target_speed

    state.vx += (tar_vx - state.vx) * clamp(acc * dt, 0.0, 1.0)
    state.vz += (tar_vz - state.vz) * clamp(acc * dt, 0.0, 1.0)

    if not state.on_ground:
        state.vx *= 1.0 - clamp(6.0 * dt, 0.0, 0.8)
        state.vz *= 1.0 - clamp(6.0 * dt, 0.0, 0.8)

    dx = state.vx * dt
    dz = state.vz * dt

    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        if state.on_ground:
            state.vx -= state.vx * clamp(fric * dt, 0.0, 1.0)
            state.vz -= state.vz * clamp(fric * dt, 0.0, 1.0)
        return

    cur_gy = ground_height(cam["x"], cam["z"], ground_y_base, slopes)
    nx = cam["x"] + dx
    nz = cam["z"] + dz
    nxt_gy = ground_height(nx, nz, ground_y_base, slopes)
    step = nxt_gy - cur_gy

    if state.on_ground:
        if step <= max_step:
            cam["x"] = nx
            cam["z"] = nz
            cam["y"] = nxt_gy + eye_h
        else:
            if mag > 0.001:
                dot = state.vx * desire_x + state.vz * desire_z
                if dot > 0:
                    state.vx -= dot * desire_x
                    state.vz -= dot * desire_z
            state.vx *= 0.6
            state.vz *= 0.6
    else:
        cam["x"] = nx
        cam["z"] = nz


def upd_phys(
    cam: Dict[str, float],
    state: PhysicsState,
    dt: float,
    grav: float,
    eye_h: float,
    slopes: List[Dict[str, float]],
    ground_y_base: float,
) -> None:
    state.vel_y += grav * dt
    cam["y"] += state.vel_y * dt

    base_h = ground_height(cam["x"], cam["z"], ground_y_base, slopes) + eye_h
    if cam["y"] <= base_h:
        cam["y"] = base_h
        state.vel_y = 0.0
        state.on_ground = True
    else:
        state.on_ground = False
