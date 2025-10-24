"""Resource management utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .mesh_loader import MeshLoaderError, load_mesh
from .world import Mesh


class MeshManager:
    def __init__(self, base_path: str | Path | None = None) -> None:
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._cache: Dict[Path, Mesh] = {}

    def _resolve(self, path: str | Path) -> Path:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        return file_path

    def load(self, path: str | Path) -> Mesh:
        file_path = self._resolve(path).resolve()
        if file_path in self._cache:
            return self._cache[file_path]
        mesh = load_mesh(file_path)
        self._cache[file_path] = mesh
        return mesh

    def get(self, path: str | Path) -> Mesh:
        return self.load(path)

    def clear(self) -> None:
        self._cache.clear()


__all__ = ["MeshManager", "MeshLoaderError"]
