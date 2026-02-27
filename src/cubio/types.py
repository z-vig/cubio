from dataclasses import dataclass
from typing import TypeAlias, Literal, TypeGuard, TypedDict, NotRequired
from enum import StrEnum
from rasterio.crs import CRS  # type: ignore
from affine import Affine  # type: ignore
import numpy as np

CubeArrayFormat: TypeAlias = Literal["BIL", "BIP", "BSQ"]
cube_array_formats: list[CubeArrayFormat] = ["BIL", "BIP", "BSQ"]
cube_array_suffix_map: dict[CubeArrayFormat, str] = {
    "BIL": ".bil",
    "BIP": ".bip",
    "BSQ": ".bsq",
}
suffix_to_format_map: dict[str, CubeArrayFormat] = {
    v: k for k, v in cube_array_suffix_map.items()
}


@dataclass(frozen=True)
class FormatIndices:
    row: int
    col: int
    band: int


FORMAT_INDICES: dict[CubeArrayFormat, FormatIndices] = {
    "BIL": FormatIndices(0, 2, 1),
    "BIP": FormatIndices(0, 1, 2),
    "BSQ": FormatIndices(2, 1, 0),
}


def is_valid_cubearrayformat(value: str) -> TypeGuard[CubeArrayFormat]:
    return value in cube_array_formats


class NumpyDType(StrEnum):
    UINT8 = "uint8"  # unsigned 8-bit integer
    INT16 = "int16"  # signed 16-bit integer
    INT32 = "int32"  # signed 32-bit integer
    FLOAT32 = "float32"  # 32-bit floating point
    FLOAT64 = "float64"  # 64-bit floating point
    CPLX64 = "complex64"  # 64-bit complex number (2 x 32-bit floats)
    CPLX128 = "complex128"  # 128-bit complex number (2 x 64-bit floats)
    UINT16 = "uint16"  # unsigned 16-bit integer
    UINT32 = "uint32"  # unsigned 32-bit integer
    INT64 = "int64"  # signed 64-bit integer
    UINT64 = "uint64"  # unsigned 64-bit integer


NumpyDTypeLiteral: TypeAlias = Literal[
    "UINT8",
    "INT16",
    "INT32",
    "FLOAT32",
    "FLOAT64",
    "CPLX64",
    "CPLX128",
    "UINT16",
    "UINT32",
    "INT64",
    "UINT64",
]

dtype_to_hdr_integer: dict[NumpyDType, int] = {
    NumpyDType.UINT8: 1,
    NumpyDType.INT16: 2,
    NumpyDType.INT32: 3,
    NumpyDType.FLOAT32: 4,
    NumpyDType.FLOAT64: 5,
    NumpyDType.CPLX64: 6,
    NumpyDType.CPLX128: 9,
    NumpyDType.UINT16: 12,
    NumpyDType.UINT32: 13,
    NumpyDType.INT64: 14,
    NumpyDType.UINT64: 15,
}
hdr_integer_to_dtype = {v: k for k, v in dtype_to_hdr_integer.items()}

RasterioDriver: TypeAlias = Literal["ENVI", "GTiff", "ISIS3"]
rasterio_drivers: list[RasterioDriver] = ["ENVI", "GTiff", "ISIS3"]


class RasterioProfile(TypedDict):
    width: int
    height: int
    count: int
    driver: RasterioDriver
    interleave: NotRequired[CubeArrayFormat]
    crs: CRS
    transform: Affine
    dtype: NumpyDType
    nodata: float | int


LabelLike: TypeAlias = np.ndarray | list[float] | list[str]


ImageSuffix: TypeAlias = Literal[
    ".tif", ".tiff", ".zarr", ".bsq", ".bil", ".bip", ".img", ".hdf5"
]
valid_image_suffixes: list[ImageSuffix] = [
    ".tif",
    ".tiff",
    ".zarr",
    ".bsq",
    ".bil",
    ".bip",
    ".img",
    ".hdf5",
]
image_suffix_priority: dict[ImageSuffix, int] = {
    ".zarr": 0,
    ".bil": 1,
    ".bip": 2,
    ".bsq": 3,
    ".img": 4,
    ".hdf5": 5,
    ".tif": 6,
    ".tiff": 7,
}


def is_valid_image_suffix(value: str) -> TypeGuard[ImageSuffix]:
    return value in valid_image_suffixes


MaskType: TypeAlias = Literal["both", "xy", "z"]
valid_mask_types: list[MaskType] = ["both", "xy", "z"]


def is_valid_mask_type(value: str) -> TypeGuard[ImageSuffix]:
    return value in valid_mask_types


TrimDirection: TypeAlias = Literal["NoTrim", "Both", "x", "y"]
