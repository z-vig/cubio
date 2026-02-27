# Built-Ins
from typing import Protocol

# Dependencies
import xarray as xr

# Local
from cubio.types import CubeArrayFormat, MaskType
from cubio.cube_size_tools import CubeSize
from cubio.cube_mask import CubeMask, MaskBuilder
from cubio.geotools.models import GeotransformModel
from .data_transfer_classes import CoordinateArrays


class CubeDataProtocol(Protocol):
    name: str
    _array: xr.DataArray | None
    _gtrans: GeotransformModel
    nodata: float
    coord_arrays: CoordinateArrays
    xdim_name: str
    ydim_name: str
    zdim_name: str

    @property
    def fmt(self) -> CubeArrayFormat: ...
    @fmt.setter
    def fmt(self, value: CubeArrayFormat) -> None: ...

    @property
    def shape(self) -> CubeSize: ...

    @property
    def array(self) -> xr.DataArray: ...
    @array.setter
    def array(self, value: xr.DataArray) -> None: ...


class MaskableCubeDataProtocol(CubeDataProtocol, Protocol):
    _mask: CubeMask
    _builder: MaskBuilder

    @property
    def mask(self) -> CubeMask: ...
    @mask.setter
    def mask(self, value: CubeMask): ...

    def reset_mask(self, which: MaskType = "both") -> None: ...

    def _apply_mask(
        self, which: MaskType = "both", drop: bool = False
    ) -> xr.DataArray: ...
