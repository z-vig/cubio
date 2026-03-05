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
from .cube_reader import cubedata_from_json_file

__all__ = [
    "CubeContext",
    "CubeData",
    "write_envi",
    "write_zarr",
    "cubedata_from_json_file",
]
