# Built-Ins
from warnings import warn

# Dependencies
import xarray as xr

# Local Imports
from cubio.types import LabelLike
from cubio.geotools.models import (
    GeotransformModel,
    PointModel,
    BoundingBoxModel,
)
from .core import CubeDataCore
from .validation import array_is_set


class GeospatialMixIn(CubeDataCore):
    @property
    def geotransform(self) -> GeotransformModel:
        if self._gtrans is None:
            return GeotransformModel.null()
        if self._array is None:
            return self._gtrans
        return self._get_current_geotransform()

    @geotransform.setter
    def geotransform(self, value: GeotransformModel) -> None:
        self._gtrans = value
        self.array = self.array

    def _get_current_geotransform(self) -> GeotransformModel:
        if self._gtrans is None:
            raise ValueError("Geotransform is not set yet.")
        return GeotransformModel(
            upperleft=PointModel(
                x=self.array.coords["Longitude"][0],
                y=self.array.coords["Latitude"][0],
            ),
            xres=self._gtrans.xres,
            row_rotation=self._gtrans.row_rotation,
            yres=self._gtrans.yres,
            col_rotation=self._gtrans.col_rotation,
        )

    def _create_coords_dict(self) -> dict[str, LabelLike]:
        # X and Y labels should be replaced with coordinates, if geotransform
        # is set, otherwise, fall back on core behavior.
        if self._gtrans is not None:
            self._xcoords, self._ycoords = self.geotransform.generate_coords(
                width=self.shape.ncolumns, height=self.shape.nrows
            )
        return super()._create_coords_dict()

    def _create_dims_tuple(self) -> tuple[str, str, str]:
        # X and Y dimension names should be Longitude and Latitude if the
        # geotransform is set.
        if self._gtrans is not None:
            if self.xdim_name not in ["XAxis", "Longitude"]:
                warn("Custom X dimension name is replaced by `Longitude`.")
            if self.ydim_name not in ["YAxis", "Latitude"]:
                warn("Custom Y dimension name is replaced by `Latitude`.")
            self.xdim_name = "Longitude"
            self.ydim_name = "Latitude"
        return super()._create_dims_tuple()

    def read_bbox(
        self, bbox: BoundingBoxModel
    ) -> tuple[xr.DataArray, GeotransformModel]:
        bottom_left_pixel = self.geotransform.map_to_pixel(
            xmap=bbox.bottom_left.x, ymap=bbox.bottom_left.y
        )
        top_right_pixel = self.geotransform.map_to_pixel(
            xmap=bbox.top_right.x, ymap=bbox.top_right.y
        )
        row_slice = slice(int(top_right_pixel.y), int(bottom_left_pixel.y))
        col_slice = slice(int(bottom_left_pixel.x), int(top_right_pixel.x))

        self._array = array_is_set(self._array)

        bbox_gtrans = GeotransformModel(
            upperleft=PointModel(x=bbox.top_left.x, y=bbox.top_left.y),
            xres=self.geotransform.xres,
            yres=self.geotransform.yres,
            row_rotation=self.geotransform.row_rotation,
            col_rotation=self.geotransform.col_rotation,
        )

        return self._array[row_slice, col_slice, :], bbox_gtrans
