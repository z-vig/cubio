from typing import TypedDict, NotRequired
from uuid import UUID

from cubio.types import NumpyDType, CubeArrayFormat
from cubio.geotools.models import GeotransformModel


class ContextBuilder(TypedDict):
    """
    Builder dictionary for constructing a CubeContext object. This is used for
    the builder pattern in the CubeContext class.
    """

    name: str
    description: str
    data_filename: str
    xymask_filename: NotRequired[str | None]
    ncols: int
    nrows: int
    nbands: int
    hdr_off: NotRequired[int]
    data_type: NumpyDType
    interleave: NotRequired[CubeArrayFormat]
    crs: str
    geotransform: GeotransformModel
    band_names: NotRequired[list[str]]
    nodata: float | int
    measurement_name: NotRequired[str]
    measurement_units: NotRequired[str]
    measurement_values: NotRequired[list[float]]
    bad_bands: NotRequired[list[int]]
    id: NotRequired[UUID]
