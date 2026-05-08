"""
### CloneMixIn

Adds cloning functionality to the CubeData class.
"""

# Built-Ins
from typing import Self
from copy import deepcopy

import xarray as xr


from .core import CubeDataCore


class CloneMixIn(CubeDataCore):
    """
    # CloneMixIn

    Adds cloning functionality to the CubeData class.
    """

    def clone(self) -> Self:
        return deepcopy(self)

    def with_data(self, new_data: xr.DataArray) -> Self:
        new_cubedata = deepcopy(self)
        new_cubedata.array = new_data
        return new_cubedata
