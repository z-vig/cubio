from .masking import MaskingMixIn
from .geospatial import GeospatialMixIn
from .transformation import TransformationMixIn


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

    pass
