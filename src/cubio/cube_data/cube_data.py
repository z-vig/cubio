# Built-Ins
from pathlib import Path

import xarray as xr

# Mixins
from .masking import MaskingMixIn
from .geospatial import GeospatialMixIn
from .transformation import TransformationMixIn
from .clone import CloneMixIn

from cubio.geotools.generate_raster_from_shapefile import raster_from_shapefile


class CubeData(MaskingMixIn, GeospatialMixIn, TransformationMixIn, CloneMixIn):
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

    def add_shapefile_mask(self, shapefile_fp: str | Path) -> xr.DataArray:
        if self._gtrans is None:
            raise ValueError(
                "Cannot mask from shapefile without a Geotransform"
            )
        shapefile_raster = raster_from_shapefile(
            (self.cube_dims.vdim, self.ycoords),
            (self.cube_dims.hdim, self.xcoords),
            shapefile_fp,
        )
        self.mask.add_to_xymask(~shapefile_raster)
        return shapefile_raster
