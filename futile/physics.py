import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Union


GROUND_REATTACH_EPSILON = 0.5


Vector3 = Tuple[float, float, float]


@dataclass
class PhysicsState:
    vx: float = 0.0
    vz: float = 0.0
    vel_y: float = 0.0
    on_ground: bool = True
    ground_normal: Vector3 = (0.0, 1.0, 0.0)
    ground_velocity: Vector3 = (0.0, 0.0, 0.0)
    ground_surface: "SurfaceProperties | None" = None


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass
class AABB:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    def expand(self, amount: float) -> "AABB":
        return AABB(
            self.min_x - amount,
            self.min_y - amount,
            self.min_z - amount,
            self.max_x + amount,
            self.max_y + amount,
            self.max_z + amount,
        )

    def merge(self, other: "AABB") -> "AABB":
        return AABB(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            min(self.min_z, other.min_z),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
            max(self.max_z, other.max_z),
        )

    def intersects(self, other: "AABB") -> bool:
        return not (
            other.max_x < self.min_x
            or other.min_x > self.max_x
            or other.max_y < self.min_y
            or other.min_y > self.max_y
            or other.max_z < self.min_z
            or other.min_z > self.max_z
        )

    def contains_horizontal(self, x: float, z: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_z <= z <= self.max_z


@dataclass
class SurfaceProperties:
    friction: float = 0.8
    slope_limit: float = math.radians(50.0)
    step_height: float = 30.0
    surface_kind: str = "default"
    slide: bool = True
    navigation_hint: str = "walk"


@dataclass
class HeightField:
    origin: Vector3
    width: int
    depth: int
    cell_size: float
    heights: List[float]

    def index(self, ix: int, iz: int) -> int:
        return iz * self.width + ix

    def height_at(self, ix: int, iz: int) -> float:
        ix = clamp(ix, 0, self.width - 1)
        iz = clamp(iz, 0, self.depth - 1)
        idx = int(self.index(int(ix), int(iz)))
        return self.heights[idx]

    def sample(self, x: float, z: float) -> Tuple[float, Vector3]:
        local_x = (x - self.origin[0]) / self.cell_size
        local_z = (z - self.origin[2]) / self.cell_size
        ix = math.floor(local_x)
        iz = math.floor(local_z)
        fx = local_x - ix
        fz = local_z - iz
        ix0 = max(0, min(self.width - 1, ix))
        iz0 = max(0, min(self.depth - 1, iz))
        ix1 = max(0, min(self.width - 1, ix + 1))
        iz1 = max(0, min(self.depth - 1, iz + 1))
        h00 = self.height_at(ix0, iz0)
        h10 = self.height_at(ix1, iz0)
        h01 = self.height_at(ix0, iz1)
        h11 = self.height_at(ix1, iz1)
        h0 = h00 * (1.0 - fx) + h10 * fx
        h1 = h01 * (1.0 - fx) + h11 * fx
        h = h0 * (1.0 - fz) + h1 * fz
        sx = (h10 - h00) / self.cell_size
        sz = (h01 - h00) / self.cell_size
        normal = normalize((-sx, 1.0, -sz))
        return h, normal

    def bounds(self) -> AABB:
        min_y = min(self.heights) if self.heights else self.origin[1]
        max_y = max(self.heights) if self.heights else self.origin[1]
        size_x = self.width * self.cell_size
        size_z = self.depth * self.cell_size
        return AABB(
            self.origin[0],
            min_y,
            self.origin[2],
            self.origin[0] + size_x,
            max_y,
            self.origin[2] + size_z,
        )


@dataclass
class VoxelVolume:
    origin: Vector3
    size: Tuple[int, int, int]
    cell_size: float
    solid: List[bool]

    def index(self, ix: int, iy: int, iz: int) -> int:
        w, h, d = self.size
        return (iy * d + iz) * w + ix

    def is_solid(self, ix: int, iy: int, iz: int) -> bool:
        w, h, d = self.size
        if not (0 <= ix < w and 0 <= iy < h and 0 <= iz < d):
            return False
        return self.solid[self.index(ix, iy, iz)]

    def bounds(self) -> AABB:
        w, h, d = self.size
        max_x = self.origin[0] + w * self.cell_size
        max_y = self.origin[1] + h * self.cell_size
        max_z = self.origin[2] + d * self.cell_size
        return AABB(self.origin[0], self.origin[1], self.origin[2], max_x, max_y, max_z)


@dataclass
class TriangleMesh:
    vertices: List[Vector3]
    indices: List[Tuple[int, int, int]]

    def bounds(self) -> AABB:
        min_x = min((v[0] for v in self.vertices), default=0.0)
        min_y = min((v[1] for v in self.vertices), default=0.0)
        min_z = min((v[2] for v in self.vertices), default=0.0)
        max_x = max((v[0] for v in self.vertices), default=0.0)
        max_y = max((v[1] for v in self.vertices), default=0.0)
        max_z = max((v[2] for v in self.vertices), default=0.0)
        return AABB(min_x, min_y, min_z, max_x, max_y, max_z)


@dataclass
class BoundingVolume:
    center: Vector3
    half_extents: Vector3

    def bounds(self) -> AABB:
        hx, hy, hz = self.half_extents
        cx, cy, cz = self.center
        return AABB(
            cx - hx,
            cy - hy,
            cz - hz,
            cx + hx,
            cy + hy,
            cz + hz,
        )


CollisionShape = Union[HeightField, VoxelVolume, TriangleMesh, BoundingVolume]


@dataclass
class CollisionObject:
    shape: CollisionShape
    surface: SurfaceProperties = field(default_factory=SurfaceProperties)
    dynamic: bool = False
    velocity: Vector3 = (0.0, 0.0, 0.0)

    def bounds(self) -> AABB:
        if isinstance(self.shape, HeightField):
            return self.shape.bounds()
        if isinstance(self.shape, VoxelVolume):
            return self.shape.bounds()
        if isinstance(self.shape, TriangleMesh):
            return self.shape.bounds()
        if isinstance(self.shape, BoundingVolume):
            return self.shape.bounds()
        raise TypeError("unknown collision shape")


@dataclass
class AABBTreeNode:
    bounds: AABB
    left: Optional["AABBTreeNode"] = None
    right: Optional["AABBTreeNode"] = None
    obj: Optional[CollisionObject] = None

    def is_leaf(self) -> bool:
        return self.obj is not None


class AABBTree:
    def __init__(self, root: Optional[AABBTreeNode]) -> None:
        self.root = root

    @staticmethod
    def build(objects: List[CollisionObject]) -> "AABBTree":
        def build_nodes(items: List[CollisionObject]) -> Optional[AABBTreeNode]:
            if not items:
                return None
            if len(items) == 1:
                bounds = items[0].bounds()
                return AABBTreeNode(bounds=bounds, obj=items[0])
            bounds = items[0].bounds()
            for obj in items[1:]:
                bounds = bounds.merge(obj.bounds())
            size_x = bounds.max_x - bounds.min_x
            size_y = bounds.max_y - bounds.min_y
            size_z = bounds.max_z - bounds.min_z
            axis = 0
            if size_y >= size_x and size_y >= size_z:
                axis = 1
            elif size_z >= size_x and size_z >= size_y:
                axis = 2
            items.sort(key=lambda o: (o.bounds().min_x, o.bounds().min_y, o.bounds().min_z)[axis])
            mid = len(items) // 2
            left = build_nodes(items[:mid])
            right = build_nodes(items[mid:])
            left_bounds = left.bounds if left else None
            right_bounds = right.bounds if right else None
            if left_bounds and right_bounds:
                merged = left_bounds.merge(right_bounds)
            elif left_bounds:
                merged = left_bounds
            elif right_bounds:
                merged = right_bounds
            else:
                merged = bounds
            return AABBTreeNode(bounds=merged, left=left, right=right)

        return AABBTree(build_nodes(list(objects)))

    def query(self, area: AABB) -> Iterable[CollisionObject]:
        stack: List[AABBTreeNode] = []
        if self.root:
            stack.append(self.root)
        while stack:
            node = stack.pop()
            if not node.bounds.intersects(area):
                continue
            if node.is_leaf():
                if node.obj is not None:
                    yield node.obj
                continue
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)


@dataclass
class SupportInfo:
    hit: bool
    height: float
    normal: Vector3
    obj: Optional[CollisionObject]


@dataclass
class ControllerSettings:
    radius: float = 25.0
    eye_height: float = 40.0
    step_height: float = 30.0
    slope_limit: float = math.radians(50.0)


class CollisionWorld:
    def __init__(self) -> None:
        self.objects: List[CollisionObject] = []
        self.tree: Optional[AABBTree] = None

    def add_object(self, obj: CollisionObject) -> None:
        self.objects.append(obj)
        self.tree = None

    def rebuild(self) -> None:
        if self.tree is None:
            self.tree = AABBTree.build(self.objects)

    def query(self, area: AABB) -> Iterable[CollisionObject]:
        self.rebuild()
        if self.tree:
            yield from self.tree.query(area)

    def sample_support(self, x: float, z: float) -> SupportInfo:
        self.rebuild()
        best = SupportInfo(hit=False, height=-math.inf, normal=(0.0, 1.0, 0.0), obj=None)
        if not self.tree:
            return best
        area = AABB(x - 1.0, -math.inf, z - 1.0, x + 1.0, math.inf, z + 1.0)
        for obj in self.tree.query(area):
            bounds = obj.bounds()
            if not bounds.contains_horizontal(x, z):
                continue
            # 広域結果を詳細化して最上支持面を確定
            height, normal = self._support_for_object(obj, x, z)
            if height > best.height:
                best = SupportInfo(hit=True, height=height, normal=normal, obj=obj)
        return best

    def _support_for_object(
        self, obj: CollisionObject, x: float, z: float
    ) -> Tuple[float, Vector3]:
        if isinstance(obj.shape, HeightField):
            height, normal = obj.shape.sample(x, z)
            return height, normal
        if isinstance(obj.shape, VoxelVolume):
            return self._support_voxel(obj.shape, x, z)
        if isinstance(obj.shape, TriangleMesh):
            return self._support_mesh(obj.shape, x, z)
        if isinstance(obj.shape, BoundingVolume):
            return obj.shape.center[1] + obj.shape.half_extents[1], (0.0, 1.0, 0.0)
        raise TypeError("unknown shape")

    def _support_voxel(self, vox: VoxelVolume, x: float, z: float) -> Tuple[float, Vector3]:
        lx = (x - vox.origin[0]) / vox.cell_size
        lz = (z - vox.origin[2]) / vox.cell_size
        ix = math.floor(lx)
        iz = math.floor(lz)
        max_y = -math.inf
        for iy in range(vox.size[1]):
            if vox.is_solid(ix, iy, iz):
                top = vox.origin[1] + (iy + 1) * vox.cell_size
                max_y = max(max_y, top)
        if max_y == -math.inf:
            return -math.inf, (0.0, 1.0, 0.0)
        return max_y, (0.0, 1.0, 0.0)

    def _support_mesh(self, mesh: TriangleMesh, x: float, z: float) -> Tuple[float, Vector3]:
        best_y = -math.inf
        best_n = (0.0, 1.0, 0.0)
        for i0, i1, i2 in mesh.indices:
            v0 = mesh.vertices[i0]
            v1 = mesh.vertices[i1]
            v2 = mesh.vertices[i2]
            if not triangle_contains(v0, v1, v2, x, z):
                continue
            y = barycentric_height(v0, v1, v2, x, z)
            if y > best_y:
                n = triangle_normal(v0, v1, v2)
                best_y = y
                best_n = n
        return best_y, best_n

    def build_navigation_graph(self) -> "NavigationGraph":
        nodes: List[NavNode] = []
        for obj in self.objects:
            if isinstance(obj.shape, HeightField):
                nodes.extend(self._nav_nodes_heightfield(obj))
            elif isinstance(obj.shape, TriangleMesh):
                nodes.extend(self._nav_nodes_mesh(obj))
            elif isinstance(obj.shape, VoxelVolume):
                nodes.extend(self._nav_nodes_voxel(obj))
        graph = NavigationGraph(nodes=nodes)
        graph.build_edges()
        return graph

    def _nav_nodes_heightfield(self, obj: CollisionObject) -> List["NavNode"]:
        hf = obj.shape
        nodes: List[NavNode] = []
        for iz in range(hf.depth):
            for ix in range(hf.width):
                x = hf.origin[0] + ix * hf.cell_size
                z = hf.origin[2] + iz * hf.cell_size
                y = hf.height_at(ix, iz)
                nodes.append(NavNode(position=(x, y, z), surface=obj.surface))
        return nodes

    def _nav_nodes_mesh(self, obj: CollisionObject) -> List["NavNode"]:
        nodes: List[NavNode] = []
        mesh = obj.shape
        for i0, i1, i2 in mesh.indices:
            v0, v1, v2 = mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2]
            cx = (v0[0] + v1[0] + v2[0]) / 3.0
            cy = (v0[1] + v1[1] + v2[1]) / 3.0
            cz = (v0[2] + v1[2] + v2[2]) / 3.0
            nodes.append(NavNode(position=(cx, cy, cz), surface=obj.surface))
        return nodes

    def _nav_nodes_voxel(self, obj: CollisionObject) -> List["NavNode"]:
        vox = obj.shape
        nodes: List[NavNode] = []
        w, h, d = vox.size
        for ix in range(w):
            for iz in range(d):
                top = -math.inf
                for iy in range(h):
                    if vox.is_solid(ix, iy, iz):
                        top = max(top, vox.origin[1] + (iy + 1) * vox.cell_size)
                if top > -math.inf:
                    x = vox.origin[0] + (ix + 0.5) * vox.cell_size
                    z = vox.origin[2] + (iz + 0.5) * vox.cell_size
                    nodes.append(NavNode(position=(x, top, z), surface=obj.surface))
        return nodes


@dataclass
class NavNode:
    position: Vector3
    surface: SurfaceProperties
    neighbors: List[int] = field(default_factory=list)


@dataclass
class NavigationGraph:
    nodes: List[NavNode]

    def build_edges(self) -> None:
        for idx, node in enumerate(self.nodes):
            for jdx, other in enumerate(self.nodes):
                if idx == jdx:
                    continue
                if node.surface.navigation_hint != other.surface.navigation_hint:
                    continue
                dx = other.position[0] - node.position[0]
                dy = other.position[1] - node.position[1]
                dz = other.position[2] - node.position[2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist <= 120.0:
                    # 近接ノード間に移動候補エッジを追加
                    node.neighbors.append(jdx)


def build_default_collision_world(
    ground_y_base: float, slopes: List[Dict[str, float]]
) -> CollisionWorld:
    world = CollisionWorld()
    base_size = 2000.0
    heights = [ground_y_base for _ in range(4)]
    base_origin = (-base_size * 0.5, ground_y_base, -base_size * 0.5)
    base_field = HeightField(
        origin=base_origin,
        width=2,
        depth=2,
        cell_size=base_size,
        heights=heights,
    )
    base_surface = SurfaceProperties(surface_kind="ground", slope_limit=math.radians(89.0))
    world.add_object(CollisionObject(shape=base_field, surface=base_surface))
    for slope in slopes:
        mesh = _slope_to_mesh(ground_y_base, slope)
        surf = SurfaceProperties(
            surface_kind="slope",
            slope_limit=math.radians(70.0),
            step_height=slope.get("step", 30.0),
        )
        # 傾斜面定義をメッシュとして登録
        world.add_object(CollisionObject(shape=mesh, surface=surf))
    return world


def _slope_to_mesh(base: float, data: Dict[str, float]) -> TriangleMesh:
    cx = data["x"]
    cz = data["z"]
    hw = data["w"] * 0.5
    hd = data["d"] * 0.5
    sx = data["sx"]
    sz = data["sz"]
    corners = [
        (cx - hw, cz - hd),
        (cx + hw, cz - hd),
        (cx + hw, cz + hd),
        (cx - hw, cz + hd),
    ]
    vertices: List[Vector3] = []
    for px, pz in corners:
        dx = px - cx
        dz = pz - cz
        y = base + dx * sx + dz * sz
        vertices.append((px, y, pz))
    indices = [(0, 1, 2), (0, 2, 3)]
    return TriangleMesh(vertices=vertices, indices=indices)


def normalize(v: Vector3) -> Vector3:
    x, y, z = v
    length = math.sqrt(x * x + y * y + z * z)
    if length < 1e-5:
        return (0.0, 1.0, 0.0)
    inv = 1.0 / length
    return (x * inv, y * inv, z * inv)


def triangle_normal(a: Vector3, b: Vector3, c: Vector3) -> Vector3:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    nx = ab[1] * ac[2] - ab[2] * ac[1]
    ny = ab[2] * ac[0] - ab[0] * ac[2]
    nz = ab[0] * ac[1] - ab[1] * ac[0]
    return normalize((nx, ny, nz))


def triangle_contains(a: Vector3, b: Vector3, c: Vector3, x: float, z: float) -> bool:
    det = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2])
    if abs(det) < 1e-6:
        return False
    l1 = ((b[2] - c[2]) * (x - c[0]) + (c[0] - b[0]) * (z - c[2])) / det
    l2 = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / det
    l3 = 1.0 - l1 - l2
    return l1 >= 0.0 and l2 >= 0.0 and l3 >= 0.0


def barycentric_height(a: Vector3, b: Vector3, c: Vector3, x: float, z: float) -> float:
    det = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2])
    if abs(det) < 1e-6:
        return -math.inf
    l1 = ((b[2] - c[2]) * (x - c[0]) + (c[0] - b[0]) * (z - c[2])) / det
    l2 = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / det
    l3 = 1.0 - l1 - l2
    return a[1] * l1 + b[1] * l2 + c[1] * l3


def ground_height(
    x: float,
    z: float,
    ground_y_base: float,
    collision_data: Union[CollisionWorld, List[Dict[str, float]]],
) -> float:
    if isinstance(collision_data, CollisionWorld):
        support = collision_data.sample_support(x, z)
        if support.hit:
            return support.height
        return ground_y_base
    y = ground_y_base
    for slope in collision_data:
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
    collision_data: Union[CollisionWorld, List[Dict[str, float]]],
    eye_h: float,
    max_step: float,
    ground_y_base: float,
    jump_v: float,
    controller: Optional[ControllerSettings] = None,
) -> None:
    if controller is None:
        controller = ControllerSettings(eye_height=eye_h, step_height=max_step)
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

    if isinstance(collision_data, CollisionWorld):
        _handle_movement_world(
            cam,
            state,
            dx,
            dz,
            desire_x,
            desire_z,
            controller,
            collision_data,
        )
    else:
        _handle_movement_legacy(
            cam,
            state,
            dx,
            dz,
            desire_x,
            desire_z,
            fric,
            collision_data,
            eye_h,
            max_step,
            ground_y_base,
        )


def _handle_movement_world(
    cam: Dict[str, float],
    state: PhysicsState,
    dx: float,
    dz: float,
    desire_x: float,
    desire_z: float,
    controller: ControllerSettings,
    world: CollisionWorld,
) -> None:
    foot_y = cam["y"] - controller.eye_height
    ascending = state.vel_y > 0.0
    support = world.sample_support(cam["x"], cam["z"])
    if support.hit:
        ground = support.obj.surface
        slope_angle = math.acos(clamp(support.normal[1], -1.0, 1.0))
        # 斜面角度で接地継続を判定
        if slope_angle > min(controller.slope_limit, ground.slope_limit):
            if ground.slide:
                # 急斜面上では速度を面方向へ投影して滑落を誘発
                dot_n = state.vx * support.normal[0] + state.vz * support.normal[2]
                slip_x = state.vx - support.normal[0] * dot_n
                slip_z = state.vz - support.normal[2] * dot_n
                state.vx = slip_x
                state.vz = slip_z
            state.on_ground = False
        else:
            if ascending:
                # 上昇中は地面拘束を外す
                state.on_ground = False
            else:
                gap = foot_y - support.height
                if gap <= GROUND_REATTACH_EPSILON:
                    state.on_ground = True
                    state.ground_normal = support.normal
                    state.ground_surface = ground
                    state.ground_velocity = ground_velocity(support.obj)
                    foot_y = max(foot_y, support.height)
                    cam["y"] = foot_y + controller.eye_height
                else:
                    # 空中維持のための耐性
                    state.on_ground = False
    else:
        state.on_ground = False

    nx = cam["x"] + dx
    nz = cam["z"] + dz
    next_support = world.sample_support(nx, nz)
    if not next_support.hit:
        cam["x"] = nx
        cam["z"] = nz
        return
    height_diff = next_support.height - foot_y
    surf = next_support.obj.surface
    allowed_step = min(controller.step_height, surf.step_height)
    if state.on_ground and height_diff <= allowed_step:
        cam["x"] = nx
        cam["z"] = nz
        cam["y"] = next_support.height + controller.eye_height
        state.ground_normal = next_support.normal
        state.ground_surface = surf
        state.ground_velocity = ground_velocity(next_support.obj)
        return
    if height_diff > allowed_step:
        if desire_x != 0.0 or desire_z != 0.0:
            dot = state.vx * desire_x + state.vz * desire_z
            if dot > 0.0:
                state.vx -= dot * desire_x
                state.vz -= dot * desire_z
        state.vx *= 0.5
        state.vz *= 0.5
        return
    cam["x"] = nx
    cam["z"] = nz


def ground_velocity(obj: CollisionObject | None) -> Vector3:
    if obj is None:
        return (0.0, 0.0, 0.0)
    if obj.dynamic:
        return obj.velocity
    return (0.0, 0.0, 0.0)


def _handle_movement_legacy(
    cam: Dict[str, float],
    state: PhysicsState,
    dx: float,
    dz: float,
    desire_x: float,
    desire_z: float,
    fric: float,
    slopes: List[Dict[str, float]],
    eye_h: float,
    max_step: float,
    ground_y_base: float,
) -> None:
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
            if desire_x != 0.0 or desire_z != 0.0:
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
    collision_data: Union[CollisionWorld, List[Dict[str, float]]],
    ground_y_base: float,
) -> None:
    state.vel_y += grav * dt
    cam["y"] += state.vel_y * dt

    if isinstance(collision_data, CollisionWorld):
        controller = ControllerSettings(eye_height=eye_h)
        foot_y = cam["y"] - controller.eye_height
        support = collision_data.sample_support(cam["x"], cam["z"])
        if support.hit:
            target_y = support.height + controller.eye_height
            if cam["y"] <= target_y:
                cam["y"] = target_y
                state.vel_y = 0.0
                state.on_ground = True
                state.ground_normal = support.normal
                state.ground_surface = support.obj.surface
                state.ground_velocity = ground_velocity(support.obj)
                if support.obj.dynamic:
                    gx, gy, gz = state.ground_velocity
                    cam["x"] += gx * dt
                    cam["z"] += gz * dt
            else:
                state.on_ground = False
        else:
            state.on_ground = False
    else:
        base_h = ground_height(cam["x"], cam["z"], ground_y_base, collision_data) + eye_h
        if cam["y"] <= base_h:
            cam["y"] = base_h
            state.vel_y = 0.0
            state.on_ground = True
        else:
            state.on_ground = False
