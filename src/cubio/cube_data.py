from __future__ import annotations

# Built-Ins
from typing import Optional
from warnings import warn


# Dependencies
import xarray as xr
import numpy as np

# Local Imports
from cubio.types import LabelLike, CubeArrayFormat, FORMAT_INDICES, MaskType
from cubio.geotools.models import GeotransformModel, PointModel
from cubio.cube_size_tools import get_cube_size, transpose_cube, CubeSize
from cubio.cube_mask import CubeMask


class CubeData:
    def __init__(
        self,
        name: str,
        format: CubeArrayFormat,
        *,
        xcoord_label: Optional[LabelLike] = None,
        ycoord_label: Optional[LabelLike] = None,
        zcoord_label: Optional[LabelLike] = None,
        x_name: str = "XAxis",
        y_name: str = "YAxis",
        z_name: str = "ZAxis",
        geotransform: Optional[GeotransformModel] = None,
        nodata: float | int = -999,
    ) -> None:
        """
        Class for storing the data and metadata of an image cube.

        Parameters
        ----------
        name: str
            Name of the cube.
        format: CubeArrayFormat
            Format of the cube array, one of ["BSQ", "BIL", "BIP"].
        x_labels: Optional[LabelLike]
            Optional labels for the x dimension. If not provided, integer
            indexing will be used.
        y_labels: Optional[LabelLike]
            Optional labels for the y dimension. If not provided, integer
            indexing will be used.
        z_labels: Optional[LabelLike]
            Optional labels for the z dimension. If not provided, integer
            indexing will be used.
        x_name: str
            Name of the x dimension. Default is "XAxis".
        y_name: str
            Name of the y dimension. Default is "YAxis".
        z_name: str
            Name of the z dimension. Default is "ZAxis".
        geotransform: Optional[GeotransformModel]
            Optional geotransform for the cube. If provided, it will be used to
            generate the x and y coordinate arrays, and any provided x and y
            labels will be overwritten.
        nodata: float | int
            Value to use for nodata. Default is -999.
        """
        self.name: str = name
        self._fmt: CubeArrayFormat = format
        self._gtrans = geotransform
        self.nodata = nodata

        self._array: xr.DataArray | None = None

        self.xcoord_array: Optional[LabelLike] = xcoord_label
        self.ycoord_array: Optional[LabelLike] = ycoord_label
        self.zcoord_array: Optional[LabelLike] = zcoord_label

        self.xdim_name = x_name
        self.ydim_name = y_name
        self.zdim_name = z_name

        if self._gtrans is not None:
            if (self.xcoord_array is not None) | (
                self.ycoord_array is not None
            ):
                warn(
                    "A geotransform was provided, so any provided x and y"
                    f" labels will be overwritten for {self.name}"
                )
            self.xdim_name = "Longitude"
            self.ydim_name = "Latitude"

        self._mask: CubeMask | None = None

    @property
    def fmt(self) -> CubeArrayFormat:
        return self._fmt

    @fmt.setter
    def fmt(self, value: CubeArrayFormat) -> None:
        self._fmt = value
        idx = FORMAT_INDICES[self._fmt]
        self.rowindex = idx.row
        self.colindex = idx.col
        self.bandindex = idx.band

    @property
    def mask(self) -> CubeMask:
        if self._mask is None:
            return CubeMask.transparent(self)
        return self._mask

    @mask.setter
    def mask(self, value: CubeMask) -> None:
        self._mask = value

    def reset_mask(self, which: MaskType = "both") -> None:
        """
        Resets the current cube mask.

        Parameters
        ----------
        which: MaskType
            Which mask(s) to reset: "both", "xy" or "z".
        """
        if which == "both":
            self.mask = CubeMask.transparent(self)
        elif which == "xy":
            old_zmask = self.mask.zmask
            self.mask = CubeMask.transparent(self)
            self.mask.add_to_zmask(old_zmask)
        elif which == "z":
            old_xymask = self.mask.xymask
            self.mask = CubeMask.transparent(self)
            self.mask.add_to_xymask(old_xymask)

    def add_nodata_mask(self) -> None:
        """Adds a mask to the current cube mask based on the nodata value."""
        self._array = self._array_is_set()
        nodata = self._array[:, :, 0].drop_vars(self.zdim_name) == self.nodata
        self.mask.add_to_xymask(nodata)

    @property
    def shape(self) -> CubeSize:
        if (
            (self.ycoord_array is None)
            or (self.xcoord_array is None)
            or (self.zcoord_array is None)
        ):
            raise ValueError("Cube Data is not set yet.")
        return CubeSize(
            nrows=len(self.ycoord_array),
            ncolumns=len(self.xcoord_array),
            nbands=len(self.zcoord_array),
        )

    @property
    def array(self) -> xr.DataArray:
        self._array = self._array_is_set()
        self._array.name = "data"
        return self._apply_mask(drop=False)

    @array.setter
    def array(self, value: xr.DataArray) -> None:
        if value.ndim == 2:
            value = value.expand_dims(dim={self.xdim_name: 1}, axis=2)
        cubesize = get_cube_size(value, self.fmt)
        self._set_coordinate_arrays(cubesize)
        self._array = self._create_xarray_dataarray(value)

    def _set_coordinate_arrays(self, cubesize: CubeSize) -> None:
        """
        Sets the default axis labels as index arrays based on the cube size.
        """
        # Integer indexing arrays are used if none of the arrays are set yet.
        self.xcoord_array, self.ycoord_array, self.zcoord_array = (
            self._coord_arrays_are_set()
        )
        self.xcoord_array = np.arange(0, cubesize.ncolumns)
        self.ycoord_array = np.arange(0, cubesize.nrows)
        self.zcoord_array = np.arange(0, cubesize.nbands)

        # X and Y labels should be replaced with coordinates, if possible.
        if self.geotransform is not None:
            self.xcoord_array, self.ycoord_array = (
                self.geotransform.generate_coords(
                    width=cubesize.ncolumns, height=cubesize.nrows
                )
            )

        return

    def _create_dims_tuple(self) -> tuple[str, str, str]:
        """
        Creates a tuple of string dim names in the correct order for the
        current array format.
        """
        _n = np.array(
            [self.ydim_name, self.xdim_name, self.zdim_name]
        )  # names
        dims = (_n[self.rowindex], _n[self.colindex], _n[self.bandindex])
        return dims

    def _create_xarray_dataarray(self, value: xr.DataArray) -> xr.DataArray:
        """
        Creates a new xarray dataarray from an existing dataarray but with the
        correct coords and dimension names in the correct order for the current
        cube format.
        """
        dims = self._create_dims_tuple()
        return xr.DataArray(
            value.data,
            coords={
                self.xdim_name: self.xcoord_array,
                self.ydim_name: self.ycoord_array,
                self.zdim_name: self.zcoord_array,
            },
            dims=dims,
        )

    @property
    def geotransform(self) -> GeotransformModel:
        if self._gtrans is None:
            return GeotransformModel.null()
        if self._array is None:
            return self._gtrans
        return self._get_masked_geotransform()

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

    def _apply_mask(
        self, which: MaskType = "both", drop: bool = False
    ) -> xr.DataArray:
        if self._array is None:
            raise ValueError("Array is not set for applying mask.")

        masks = {
            "both": ~self.mask.xymask & ~self.mask.zmask,
            "xy": ~self.mask.xymask,
            "z": ~self.mask.zmask,
        }
        return self._array.where(masks[which], np.nan, drop=drop)

    def _get_masked_geotransform(self) -> GeotransformModel:
        if self._gtrans is None:
            raise ValueError("Geotransform is not set yet.")
        return GeotransformModel(
            upperleft=PointModel(
                x=min(self.array.coords["Longitude"]),
                y=max(self.array.coords["Latitude"]),
            ),
            xres=self._gtrans.xres,
            row_rotation=self._gtrans.row_rotation,
            yres=self._gtrans.yres,
            col_rotation=self._gtrans.col_rotation,
        )

    def get_unmasked_array(self, ignore: MaskType = "both") -> xr.DataArray:
        """
        Get the unmasked version of the data cube.

        Parameters
        ----------
        ignore: MaskType
            Which mask(s) to ignore: "both", "xy" or "z".
        """
        if self._array is None:
            raise ValueError("Array is not set for applying mask.")

        if ignore == "both":
            return self._array
        elif ignore == "xy":
            return self._apply_mask("z")
        elif ignore == "z":
            return self._apply_mask("xy")

    def _array_is_set(self) -> xr.DataArray:
        if self._array is None:
            raise ValueError("Array is not set.")
        return self._array

    def _coord_arrays_are_set(self) -> tuple[LabelLike, LabelLike, LabelLike]:
        if self.xcoord_array is None:
            raise ValueError("X Coordinates are not set.")
        if self.ycoord_array is None:
            raise ValueError("Y Coordinates are not set.")
        if self.zcoord_array is None:
            raise ValueError("Z Coordinates are not set.")
        return self.xcoord_array, self.ycoord_array, self.zcoord_array
