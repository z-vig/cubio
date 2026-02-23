# Built-Ins
from typing import Optional
from typing_extensions import Self
from warnings import warn

# Dependencies
import xarray as xr
import numpy as np

# Local Imports
from cubio.types import LabelLike, CubeArrayFormat
from cubio.geotools.models import GeotransformModel
from cubio.cube_size_tools import get_cube_size, transpose_cube, CubeSize


class CubeMask:
    def __init__(
        self,
        *,
        shape: CubeSize,
        xy_mask: np.ndarray | xr.DataArray | None = None,
        z_mask: np.ndarray | xr.DataArray | None = None,
    ) -> None:
        self.shape = shape
        if (xy_mask is not None) and (xy_mask.dtype is not bool):
            raise ValueError("XY Mask must be of dtype: bool")
        if (z_mask is not None) and (z_mask.dtype is not bool):
            raise ValueError("Z Mask must be of dtype: bool")

        self._xymask = xy_mask
        self._zmask = z_mask

    @classmethod
    def transparent(cls, shape: CubeSize) -> Self:
        return cls(
            shape=shape,
            xy_mask=np.ones((shape.nrows, shape.ncolumns), dtype=bool),
            z_mask=np.ones(shape.nbands, dtype=bool),
        )

    @property
    def xymask(self) -> np.ndarray | xr.DataArray:
        if self._xymask is None:
            raise ValueError("XY Mask not set yet.")
        return self._xymask

    @xymask.setter
    def xymask(self, value: np.ndarray | xr.DataArray) -> None:
        if value.dtype is not bool:
            raise ValueError("XY Mask must be of type: bool")
        self._xymask = value

    @property
    def zmask(self) -> np.ndarray | xr.DataArray:
        if self._zmask is None:
            raise ValueError("XY Mask not set yet.")
        return self._zmask

    @zmask.setter
    def zmask(self, value: np.ndarray | xr.DataArray) -> None:
        if value.dtype is not bool:
            raise ValueError("Z Mask must be of type: bool")
        self._zmask = value


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

        self.xname = x_name
        self.yname = y_name
        self.zname = z_name

        if self._gtrans is not None:
            if (self._xlbl is not None) | (self._ylbl is not None):
                warn(
                    "A geotransform was provided, so any provided x and y"
                    f" labels will be overwritten for {self.name}"
                )
            self.xname = "Longitude"
            self.yname = "Latitude"

            self._mask: CubeMask | None = None

    @property
    def mask(self) -> CubeMask:
        if self._mask is None:
            return CubeMask.transparent(self.shape)
        return self._mask

    @mask.setter
    def mask(self, value: CubeMask) -> None:
        self._mask = value

    @property
    def shape(self) -> CubeSize:
        if (
            (self._ylbl is None)
            or (self._xlbl is None)
            or (self._zlbl is None)
        ):
            raise ValueError("Cube Data is not set yet.")
        return CubeSize(
            nrows=len(self._ylbl),
            ncolumns=len(self._xlbl),
            nbands=len(self._zlbl),
        )

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
            dims = (self.yname, self.zname, self.xname)
        elif self.fmt == "BIP":
            dims = (self.yname, self.xname, self.zname)
        elif self.fmt == "BSQ":
            dims = (self.zname, self.xname, self.yname)
        else:
            raise ValueError("Invalid axis names.")

        self._array = xr.DataArray(
            value.data,
            coords={
                self.xname: self._xlbl,
                self.yname: self._ylbl,
                self.zname: self._zlbl,
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

    def transpose_to_rasterio(self) -> xr.DataArray:
        return transpose_cube(self.fmt, "RASTERIO", self.array)
