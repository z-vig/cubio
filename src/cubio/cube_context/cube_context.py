# Built-ins
from typing import Literal, Union, overload
from typing_extensions import Self
from pathlib import Path
from uuid import UUID, uuid4

# Dependencies
import numpy as np
from pydantic import (
    BaseModel,
    field_serializer,
    model_validator,
    Field,
    field_validator,
    PrivateAttr,
)
import xarray as xr

# Local Imports
from cubio.types import (
    NumpyDType,
    CubeArrayFormat,
    is_valid_cubearrayformat,
    RasterioProfile,
)
from cubio.geotools.models import GeotransformModel
from cubio.cube_size_tools import CubeSize

# Sub-package Imports
from .builder import ContextBuilder


class CubeContext(BaseModel):
    """
    A context object for managing the metadata and properties of a data cube.
    """

    name: str = Field(..., description="A short name for the cube.")
    description: str = Field(
        ..., description="A description of the data cube."
    )
    data_filename: str = Field(
        ...,
        description="Filename of the data that is described by this file. It"
        " should be in the same directory as this file.",
    )
    xymask_filename: str | None = Field(
        default=None,
        description="Filename of the spatial mask for the dataset.",
    )
    ncols: int = Field(..., description="Number of columns in the data cube.")
    nrows: int = Field(..., description="Number of rows in the data cube.")
    nbands: int = Field(..., description="Number of bands in the data cube.")
    hdr_off: int = Field(default=0, description="Header offset in bytes.")
    data_type: NumpyDType = Field(
        ..., description="The data type of the cube."
    )
    interleave: CubeArrayFormat = Field(
        default="BIP",
        description=(
            "The interleave format of the cube. Either BIL, BIP, or BSQ."
        ),
    )
    crs: str = Field(
        ..., description="Coordinate reference system of the cube."
    )
    geotransform: GeotransformModel = Field(
        ..., description="Geotransform model for the cube."
    )
    band_names: list[str] = Field(default_factory=list)
    nodata: float | int = Field(..., description="The nodata value.")
    measurement_name: str = Field(
        default="Measurement",
        description="Name that describes the nature of the measurement along "
        "the cube z-axis.",
    )
    measurement_units: str = Field(
        default="unitless", description="The measurement units."
    )
    measurement_values: list[float] = Field(
        default_factory=list, description="The measurement values."
    )
    bad_bands: list[int] = Field(
        default_factory=list, description="List of bad band flags."
    )
    id: UUID = Field(
        default_factory=uuid4, description="Unique ID of the cube object."
    )
    _retrieval_path: Path | Literal["NoRetrieval"] = PrivateAttr(
        default="NoRetrieval"
    )

    # ==== Properties =====
    @property
    def shape(self) -> CubeSize:
        """
        Returns the shape of the data cube as a CubeSize object.
        """
        return CubeSize(
            nrows=self.nrows, ncolumns=self.ncols, nbands=self.nbands
        )

    @property
    def shape_tuple(self) -> tuple[int, int, int]:
        """
        Returns the shape of the data cube as a tuple in the order specified by
        the interleave format.
        """
        return self.shape.as_tuple(self.interleave)

    @property
    def builder(self) -> ContextBuilder:
        """
        Returns a builder dictionary for constructing a CubeContext object.
        This is used for the builder pattern in the CubeContext class.
        """
        _builder: ContextBuilder = {
            "name": self.name,
            "description": self.description,
            "data_filename": self.data_filename,
            "xymask_filename": self.xymask_filename,
            "ncols": self.ncols,
            "nrows": self.nrows,
            "nbands": self.nbands,
            "hdr_off": self.hdr_off,
            "data_type": self.data_type,
            "interleave": self.interleave,
            "crs": self.crs,
            "geotransform": self.geotransform,
            "band_names": self.band_names,
            "nodata": self.nodata,
            "measurement_name": self.measurement_name,
            "measurement_units": self.measurement_units,
            "measurement_values": self.measurement_values,
            "bad_bands": self.bad_bands,
            "id": self.id,
        }
        return _builder

    @property
    def retrieval_path(self) -> Path:
        if self._retrieval_path == "NoRetrieval":
            raise AttributeError("Retrieval Path was not set.")
        return self._retrieval_path

    @retrieval_path.setter
    def retrieval_path(self, retrieval_path: Path) -> None:
        self._retrieval_path = Path(retrieval_path)

    # ==== Measurement Queries =====
    def get_bbl_mask(
        self, measurement_name: str | None = None
    ) -> xr.DataArray:
        """Returns a mask for the bad bands based on the bad_bands list."""
        if measurement_name is None:
            name = self.measurement_name
        else:
            name = measurement_name
        return xr.DataArray(
            [not bool(i) for i in self.bad_bands],
            coords={name: self.measurement_values},
        )

    def get_measurement_mask(
        self, min_val: float, max_val: float
    ) -> xr.DataArray:
        """
        Returns a mask for the measurement values based on the provided min and
        max values. Values outside the range will be masked.
        """
        measarr = np.array(self.measurement_values)
        mask = (measarr < min_val) | (measarr > max_val)
        return xr.DataArray(
            mask, coords={self.measurement_name: self.measurement_values}
        )

    @overload
    def get_measurement_idx(
        self, value: Union[list[float], list[int]]
    ) -> list[int]: ...
    @overload
    def get_measurement_idx(self, value: Union[float, int]) -> int: ...

    def get_measurement_idx(
        self, value: Union[float, int, list[float], list[int]]
    ) -> Union[int, list[int]]:
        """
        Returns the index or indices of the measurement value(s) that are\
        closest to the provided value(s).
        """
        measarr = np.array(self.measurement_values)
        if isinstance(value, list):
            idxs: list[int] = []
            for i in value:
                idxs.append(int(np.argmin(abs(measarr - i))))
            return idxs
        else:
            return int(np.argmin(abs(measarr - value)))

    # ==== Inner Constructors ====
    @classmethod
    def from_builder(cls, builder_dict: ContextBuilder) -> Self:
        return cls(**builder_dict)

    @classmethod
    def from_json(cls, savefp: str | Path) -> Self:
        """Convenience method for reading in the model from json file."""
        with open(savefp, "r") as f:
            json = f.read()
        valid_model = cls.model_validate_json(json)
        valid_model.retrieval_path = Path(savefp)
        return valid_model

    @classmethod
    def from_rasterio_profile(
        cls,
        name: str,
        description: str,
        data_filename: str,
        band_names: list[str],
        measurement_name: str,
        measurement_units: str,
        measurement_values: list[float],
        bbl: list[int],
        rasterio_profile: RasterioProfile,
    ) -> Self:
        _builder: ContextBuilder = {
            "name": name,
            "description": description,
            "data_filename": data_filename,
            "ncols": rasterio_profile["width"],
            "nrows": rasterio_profile["height"],
            "nbands": rasterio_profile["count"],
            "hdr_off": 0,
            "data_type": NumpyDType(rasterio_profile["dtype"]),
            "interleave": rasterio_profile.get("interleave", "BIP"),
            "crs": rasterio_profile["crs"],
            "geotransform": GeotransformModel.fromaffine(
                rasterio_profile["transform"]
            ),
            "band_names": band_names,
            "nodata": rasterio_profile["nodata"],
            "measurement_name": measurement_name,
            "measurement_units": measurement_units,
            "measurement_values": measurement_values,
            "bad_bands": bbl,
            "id": uuid4(),
        }
        return cls.from_builder(_builder)

    @classmethod
    def from_single_band_data(
        cls,
        name: str,
        description: str,
        data_filename: str,
        height: int,
        width: int,
        crs: str,
        geotransform: GeotransformModel,
        nodata: float | int,
        interleave: CubeArrayFormat = "BIL",
        dtype: NumpyDType = NumpyDType.FLOAT32,
    ) -> Self:
        _builder: ContextBuilder = {
            "name": name,
            "description": description,
            "data_filename": data_filename,
            "ncols": width,
            "nrows": height,
            "nbands": 1,
            "hdr_off": 0,
            "data_type": NumpyDType(dtype),
            "interleave": interleave,
            "crs": crs,
            "geotransform": geotransform,
            "band_names": [name],
            "nodata": nodata,
            "measurement_name": name,
            "measurement_units": "na",
            "measurement_values": [0.0],
            "bad_bands": [1],
            "id": uuid4(),
        }
        return cls.from_builder(_builder)

    @field_serializer("interleave", mode="plain")
    def lowercase(self, value: CubeArrayFormat) -> str:
        return value.lower()

    @field_validator("interleave", mode="before")
    @classmethod
    def uppercase(cls, value: str) -> CubeArrayFormat:
        ustr = value.upper()
        if is_valid_cubearrayformat(ustr):
            return ustr
        else:
            raise ValueError(f"Invalid interleave: {ustr}")

    @model_validator(mode="after")
    def set_measurement_values(self) -> Self:
        current_measval = self.measurement_values
        if len(current_measval) == 0:
            self.measurement_values = [float(i) for i in range(self.nbands)]

        if len(self.measurement_values) != self.nbands:
            raise ValueError(
                "Length of measurement values must match the number of bands."
            )
        return self

    @model_validator(mode="after")
    def set_default_bbl(self) -> Self:
        current_bbl = self.bad_bands
        if len(current_bbl) == 0:
            self.bad_bands = [1] * len(self.measurement_values)
        else:
            self.bad_bands = current_bbl

        if len(self.bad_bands) != self.nbands:
            print(self.bad_bands)
            raise ValueError(
                "Length of bad band list must match the number of bands."
            )

        return self

    def write_json(self, savefp: str | Path) -> None:
        """Convenience function for dumping the model to a json file."""
        json_str = self.model_dump_json(indent=2)
        with open(Path(savefp).with_suffix(".json"), "w") as f:
            f.write(json_str)
