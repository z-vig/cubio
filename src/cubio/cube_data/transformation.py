from .core import CubeDataCore
from cubio.types import CubeArrayFormat
from cubio.cube_size_tools import transpose_cube
import xarray as xr


class TransformationMixIn(CubeDataCore):
    def transpose_to(self, format: CubeArrayFormat) -> None:
        old_format = self.fmt
        self.fmt = format
        new_arr = transpose_cube(old_format, format, self.array)
        self.array = new_arr

    def transpose_to_rasterio(self) -> xr.DataArray:
        return transpose_cube(self.fmt, "RASTERIO", self.array)
