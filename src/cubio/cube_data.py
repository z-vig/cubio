# Built-Ins
from typing import Optional
from warnings import warn

# Dependencies
import xarray as xr
import numpy as np

# Local Imports
from cubio.types import LabelLike, CubeArrayFormat
from cubio.geotransform import GeotransformModel
from cubio.cube_size_tools import get_cube_size, transpose_cube


class CubeData:
    def __init__(
        self,
        name: str,
        format: CubeArrayFormat,
        *,
        x_labels: Optional[LabelLike] = None,
        y_labels: Optional[LabelLike] = None,
        z_labels: Optional[LabelLike] = None,
        x_name: str = "XAxis",
        y_name: str = "YAxis",
        z_name: str = "ZAxis",
        geotransform: Optional[GeotransformModel] = None,
    ) -> None:
        self.name: str = name
        self.fmt: CubeArrayFormat = format
        self._gtrans = geotransform

        self._array: xr.DataArray | None = None

        self._xlbl = x_labels
        self._ylbl = y_labels
        self._zlbl = z_labels

        self._xname = x_name
        self._yname = y_name
        self._zname = z_name

        if self._gtrans is not None:
            if (self._xlbl is not None) | (self._ylbl is not None):
                warn(
                    "A geotransform was provided, so any provided x and y"
                    f" labels will be overwritten for {self.name}"
                )
            self._xname = "Longitude"
            self._yname = "Latitude"

    @property
    def array(self) -> xr.DataArray:
        if self._array is None:
            raise NotImplementedError(
                f"A data array has not been set for {self.name}."
            )
        return self._array

    @array.setter
    def array(self, value: xr.DataArray) -> None:
        s = get_cube_size(value, self.fmt)

        # Setting label values as indices if none are provided.
        if self._xlbl is None:
            self._xlbl = np.arange(0, s.ncolumns)
        if self._ylbl is None:
            self._ylbl = np.arange(0, s.nrows)
        if self._zlbl is None:
            self._zlbl = np.arange(0, s.nbands)

        # X and Y labels should be replaced with coordinates, if possible.
        if self.geotransform is not None:
            self._xlbl, self._ylbl = self.geotransform.generate_coords(
                width=s.ncolumns, height=s.nrows
            )

        dims: tuple[str, str, str]
        if self.fmt == "BIL":
            dims = (self._yname, self._zname, self._xname)
        elif self.fmt == "BIP":
            dims = (self._yname, self._xname, self._zname)
        elif self.fmt == "BSQ":
            dims = (self._zname, self._xname, self._yname)
        else:
            raise ValueError("Invalid axis names.")

        self._array = xr.DataArray(
            value.data,
            coords={
                self._xname: self._xlbl,
                self._yname: self._ylbl,
                self._zname: self._zlbl,
            },
            dims=dims,
        )

    @property
    def geotransform(self) -> GeotransformModel:
        if self._gtrans is None:
            raise ValueError("Geotransform has not been set.")
        return self._gtrans

    @geotransform.setter
    def geotransform(self, value: GeotransformModel) -> None:
        self._gtrans = value
        self.array = self.array

    def transpose_to(self, format: CubeArrayFormat) -> None:
        old_format = self.fmt
        self.fmt = format
        new_arr = transpose_cube(old_format, format, self.array)
        self.array = new_arr
