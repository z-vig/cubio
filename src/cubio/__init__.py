"""
Welcome to `cubio`!
---
### Available Classes
- CubeContext // Metadata for reading in geospatial cubes.
- CubeData // Memory-mapped pointer for performing processing with `xarray`.
"""

from .cube_context import CubeContext

__all__ = ["CubeContext"]
