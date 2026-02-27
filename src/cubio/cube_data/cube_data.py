# Built-Ins
from pathlib import Path

# Mixins
from .masking import MaskingMixIn
from .geospatial import GeospatialMixIn
from .transformation import TransformationMixIn

from cubio.geotools.generate_raster_from_shapefile import raster_from_shapefile


class CubeData(MaskingMixIn, GeospatialMixIn, TransformationMixIn):
    """
    # CubeData
    Class for storing and manipulating the data of an Image Cube.

    ### Key Features
    #### `mask`
     - Creates two main masks for the data: an `xymask` and a `zmask`.
     - `xymask` is for the spatial dimension.
     - `zmask` is for the measurement dimension.
    #### `geospatial`
    - Handles the geotransform of the data cube.
    """

    def add_shapefile_mask(self, shapefile_fp: str | Path) -> None:
        if self._gtrans is None:
            raise ValueError(
                "Cannot mask from shapefile without a Geotransform"
            )
        shapefile_raster = raster_from_shapefile(
            self.ycoords, self.xcoords, shapefile_fp
        )
        self.mask.add_to_xymask(~shapefile_raster)
