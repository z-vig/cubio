"""
### CloneMixIn

Adds cloning functionality to the CubeData class.
"""

# Built-Ins
from typing import Self, Optional
from copy import deepcopy

import xarray as xr

from cubio.geotools.models import GeotransformModel

from .core import CubeDataCore


class CloneMixIn(CubeDataCore):
    """
    # CloneMixIn

    Adds cloning functionality to the CubeData class.
    """

    def clone(self) -> Self:
        return deepcopy(self)

    def with_data(
        self,
        new_data: Optional[xr.DataArray] = None,
        new_gtrans: Optional[GeotransformModel] = None,
        new_crs: Optional[str] = None,
    ) -> Self:
        new_cubedata = deepcopy(self)
        if new_gtrans is not None:
            pass
        if new_data is not None:
            new_cubedata.array = new_data
        return new_cubedata
