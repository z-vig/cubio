from __future__ import annotations

# Dependencies
import xarray as xr

# Local Imports
from cubio.geotools.models import (
    GeotransformModel,
    PointModel,
    BoundingBoxModel,
)
from .core import CubeDataCore
from .validation import array_is_set


class GeospatialMixIn(CubeDataCore):
    """
    # GeospatialMixIn
    Adds geospatial manipulation to the CubeData class.
    """

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

    @property
    def bounds(self) -> BoundingBoxModel:
        return self.geotransform.get_bbox(
            self.shape.nrows, self.shape.ncolumns
        )

    def set_array_coords(self, value: xr.DataArray) -> xr.DataArray:
        super().set_array_coords(value)
        xcrds, ycrds = self.geotransform.generate_coords(
            width=self.shape.ncolumns, height=self.shape.nrows
        )
        crd_dict = {
            self.cube_dims.vdim: ycrds,
            self.cube_dims.hdim: xcrds,
            self.cube_dims.zdim: value.coords[self.cube_dims.zdim],
        }
        self._xcoords = xcrds
        self._ycoords = ycrds
        value = value.assign_coords(crd_dict)
        return value

    def _post_array_setting_config(self) -> None:
        super()._post_array_setting_config()
        self.update_cube_dims(vdim_name="Latitude", hdim_name="Longitude")
        return None

    def _get_current_geotransform(self) -> GeotransformModel:
        if self._gtrans is None:
            raise ValueError("Geotransform is not set yet.")
        return self._gtrans

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

        bboxdata = self._array[row_slice, col_slice, :]

        return bboxdata, bbox_gtrans
