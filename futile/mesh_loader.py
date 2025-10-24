"""Mesh loader implementations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

from .world import Mesh, MeshGeometry, MeshLOD


class MeshLoaderError(Exception):
    pass


def load_mesh(path: str | Path) -> Mesh:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".obj":
        return _load_obj(file_path)
    if suffix in {".fsm", ".mesh"}:
        return _load_simple(file_path)
    raise MeshLoaderError(f"Unsupported mesh format: {file_path.suffix}")


def _load_obj(path: Path) -> Mesh:
    vertices: List[tuple[float, float, float]] = []
    normals: List[tuple[float, float, float]] = []
    indices: List[tuple[int, int, int]] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                _, x, y, z = line.split(maxsplit=3)
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("vn "):
                _, x, y, z = line.split(maxsplit=3)
                normals.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                parts = line.split()[1:]
                face = []
                for part in parts:
                    idx = part.split("/")[0]
                    face.append(int(idx) - 1)
                if len(face) < 3:
                    continue
                for i in range(1, len(face) - 1):
                    indices.append((face[0], face[i], face[i + 1]))

    if not vertices or not indices:
        raise MeshLoaderError(f"OBJ mesh incomplete: {path}")

    geometry = MeshGeometry(vertices=vertices, normals=normals, indices=indices)
    return Mesh([MeshLOD(max_distance=math.inf, geometry=geometry)], name=path.stem)


def _load_simple(path: Path) -> Mesh:
    lods: List[MeshLOD] = []
    cur_vertices: List[tuple[float, float, float]] = []
    cur_normals: List[tuple[float, float, float]] = []
    cur_indices: List[tuple[int, int, int]] = []
    cur_limit = math.inf

    def flush() -> None:
        if cur_vertices and cur_indices:
            geometry = MeshGeometry(
                vertices=list(cur_vertices),
                normals=list(cur_normals),
                indices=list(cur_indices),
            )
            lods.append(MeshLOD(max_distance=cur_limit, geometry=geometry))

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            head = tokens[0].lower()
            if head == "lod":
                flush()
                cur_vertices.clear()
                cur_normals.clear()
                cur_indices.clear()
                if len(tokens) < 2:
                    raise MeshLoaderError(f"LODの距離が不足しています: {path}")
                if tokens[1].lower() == "inf":
                    cur_limit = math.inf
                else:
                    cur_limit = float(tokens[1])
            elif head == "v":
                if len(tokens) != 4:
                    raise MeshLoaderError(f"頂点の記述が不正です: {path}")
                cur_vertices.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
            elif head == "vn":
                if len(tokens) != 4:
                    raise MeshLoaderError(f"法線の記述が不正です: {path}")
                cur_normals.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
            elif head == "f":
                if len(tokens) != 4:
                    raise MeshLoaderError(f"三角形数が不正です: {path}")
                a, b, c = (int(tok) - 1 for tok in tokens[1:4])
                cur_indices.append((a, b, c))
            elif head == "end":
                flush()
                cur_vertices.clear()
                cur_normals.clear()
                cur_indices.clear()
                cur_limit = math.inf
            else:
                raise MeshLoaderError(f"未知のトークンです: {tokens[0]}")

    flush()

    if not lods:
        raise MeshLoaderError(f"メッシュデータが存在しません: {path}")

    return Mesh(lods, name=path.stem)
