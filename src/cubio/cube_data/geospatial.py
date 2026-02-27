# Built-Ins
from warnings import warn

# Local Imports
from cubio.types import LabelLike
from cubio.geotools.models import GeotransformModel, PointModel
from .core import CubeDataCore


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
                x=min(self.array.coords["Longitude"]),
                y=max(self.array.coords["Latitude"]),
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
