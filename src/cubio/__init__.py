"""
Welcome to `cubio`!
---
### Available Classes
- CubeContext // Metadata for reading in geospatial cubes.
- CubeData // Memory-mapped pointer for performing processing with `xarray`.
"""

from .cube_context import CubeContext
from .cube_data import CubeData
from .cube_writer import write_envi, write_zarr
from .cube_reader import read_cube_data

__all__ = [
    "CubeContext",
    "CubeData",
    "write_envi",
    "write_zarr",
    "read_cube_data",
]
