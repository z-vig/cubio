"""
Welcome to `cubio`!
---
### Available Classes
- CubeContext // Metadata for reading in geospatial cubes.
- CubeData // Memory-mapped pointer for performing processing with `xarray`.
"""

from .convenience_functions.cube_from_numpy import cube_from_numpy
from .cube_context import CubeContext
from .cube_data import CubeData
from .cube_dims import CubeDims
from .cube_reader import cube_from_envi, cube_from_gtif, cube_from_json
from .cube_writer import write_envi, write_zarr

__all__ = [
    "CubeContext",
    "CubeData",
    "CubeDims",
    "cube_from_envi",
    "cube_from_gtif",
    "cube_from_json",
    "cube_from_numpy",
    "write_envi",
    "write_zarr",
]
