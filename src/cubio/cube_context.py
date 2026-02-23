# Built-ins
from typing import TypedDict, Literal
from typing_extensions import Self
from tempfile import NamedTemporaryFile
from pathlib import Path
import shutil
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
import rasterio as rio  # type: ignore
import xarray as xr

# Local Imports
from cubio.types import (
    NumpyDType,
    CubeArrayFormat,
    is_valid_cubearrayformat,
    suffix_to_format_map,
    RasterioProfile,
    ImageSuffix,
    is_valid_image_suffix,
    image_suffix_priority,
)
from cubio.geotools.models import GeotransformModel
from cubio.envi_hdr_tools import (
    replace_hdr_band_names,
    replace_hdr_description,
    replace_shape_fields,
)
from cubio.cube_size_tools import CubeSize
from cubio.cube_data import CubeData


class ContextBuilder(TypedDict):
    name: str
    description: str
    data_filename: Path
    ncols: int
    nrows: int
    nbands: int
    hdr_off: int
    data_type: NumpyDType
    interleave: CubeArrayFormat
    crs: str
    geotransform: GeotransformModel
    band_names: list[str]
    nodata: float | int
    measurement_name: str
    measurement_units: str
    measurement_values: list[float]
    bad_bands: list[int]
    id: UUID


class CubeContext(BaseModel):
    name: str = Field(..., description="A short name for the cube.")
    description: str = Field(
        ..., description="A description of the data cube."
    )
    data_filename: Path = Field(
        ...,
        description="Filename of the data that is described by this file. It"
        " should be in the same directory as this file.",
    )
    ncols: int = Field(..., description="Number of columns in the data cube.")
    nrows: int = Field(..., description="Number of rows in the data cube.")
    nbands: int = Field(..., description="Number of bands in the data cube.")
    hdr_off: int = Field(default=0, description="Header offset in bytes.")
    data_type: NumpyDType = Field(
        ..., description="The data type of the cube."
    )
    interleave: CubeArrayFormat = Field(
        ...,
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
    nodata: float | int = Field(default=-999, description="The nodata value.")
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

    @property
    def shape(self) -> CubeSize:
        return CubeSize(
            nrows=self.nrows, ncolumns=self.ncols, nbands=self.nbands
        )

    @property
    def shape_tuple(self) -> tuple[int, int, int]:
        return self.shape.as_tuple(self.interleave)

    @property
    def builder(self) -> ContextBuilder:
        _builder: ContextBuilder = {
            "name": self.name,
            "description": self.description,
            "data_filename": self.data_filename,
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

    @classmethod
    def from_builder(cls, builder_dict: ContextBuilder) -> Self:
        return cls(**builder_dict)

    @classmethod
    def from_json(cls, savefp: str | Path) -> Self:
        """Convenience method for reading in the model from json file."""
        with open(savefp, "r") as f:
            json = f.read()
        valid_model = cls.model_validate_json(json)
        valid_model._retrieval_path = Path(savefp)
        return valid_model

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

    @model_validator(mode="after")
    def check_measurement_values(self) -> Self:
        if len(self.measurement_values) != self.nbands:
            raise ValueError(
                "Length of measurement values must match the number of bands."
            )
        return self

    def write_envi_hdr(
        self, dst: Path | str | None = None, use_image_name: bool = True
    ) -> None:
        """
        Writes the model to a properly formatted ENVI header file.

        Parameters
        ----------
        dst : Path | str | None, optional
            Destination file path, by default None.

        Notes
        -----
        The file suffix is set internally, so the dst file does not need to
        include one.
        """

        # A temporary file should be used if no path is provided.
        if dst is None:
            tf = NamedTemporaryFile(suffix=".hdr", delete=False)
            savefp = Path(tf.name)
        else:
            savefp = Path(dst)

        # Adding suffix based on the use_image_name option.
        if use_image_name:
            if savefp.is_dir():
                savefp = Path(savefp, self.data_filename).with_suffix(".hdr")
            else:
                savefp = savefp.with_name(str(self.data_filename)).with_suffix(
                    ".hdr"
                )
        else:
            savefp = savefp.with_suffix(".hdr")

        # Need a temporary image and hdr file to write to. These will be empty.
        temp_img = savefp.with_name(f"_{savefp.name}").with_suffix(".img")
        temp_hdr = temp_img.with_suffix(".hdr")

        # Need a rasterio profile to do a proper write.
        envi_profile: RasterioProfile = {
            "driver": "ENVI",
            "width": 1,
            "height": 1,
            "count": 1,
            "dtype": self.data_type,
            "crs": self.crs,
            "transform": self.geotransform.toaffine(),
            "interleave": self.interleave,
            "nodata": -999,
        }

        # A blank image write is needed. We just use the .hdr file.
        with rio.open(temp_img, "w", **envi_profile) as f:
            f.write(np.empty((1, 1, 1)))

        temp_img.unlink()  # Discarding temporary image.

        # Modifying rasterio-written .hdr file.
        # These are existing fields.
        replace_shape_fields(
            temp_hdr,
            samples=self.ncols,
            lines=self.nrows,
            bands=self.nbands,
        )
        replace_hdr_band_names(temp_hdr, self.band_names)
        replace_hdr_description(temp_hdr, self.description)

        with open(temp_hdr, "a") as f:
            # These are all new fields.
            f.write(f"data ignore value = {self.nodata}\n")
            f.write(f"wavelenth units = {self.measurement_units}\n")
            f.write(
                "wavelength = "
                f"{{{",".join([str(i) for i in self.measurement_values])}}}\n"
            )
            f.write(
                "bbl = " f"{{{",".join([str(i) for i in self.bad_bands])}}}\n"
            )
        # Renaming the temporary hdr that was created with the temp image.
        shutil.move(temp_hdr, temp_hdr.with_name(temp_hdr.name[1:]))

        # Writing out the pydantic model in json form for easy reading.
        self.write_json(savefp)

    def write_json(self, savefp: str | Path) -> None:
        """Convenience function for dumping the model to a json file."""
        json_str = self.model_dump_json(indent=2)
        with open(Path(savefp).with_suffix(".json"), "w") as f:
            f.write(json_str)

    def lazy_load_data(self, search_dir: str | Path | None = None) -> CubeData:
        load_from: Path
        if (search_dir is not None) and (
            self._retrieval_path == "NoRetrieval"
        ):
            load_from = Path(search_dir)
        elif (self._retrieval_path != "NoRetrieval") and (search_dir is None):
            load_from = self._retrieval_path.parent
        else:
            raise ValueError(
                "If `savefp` is specified, the object must not have been "
                "validated from disk."
            )

        if not load_from.is_dir():
            raise ValueError(f"Search directory does not exist: {load_from}")

        dat = CubeData(
            self.name,
            self.interleave,
            z_labels=self.measurement_values,
            z_name=self.measurement_name,
            geotransform=self.geotransform,
        )
        candidate_image_data_files: list[tuple[Path, ImageSuffix]] = []
        for i in load_from.iterdir():
            if i.stem == str(self.data_filename.stem):
                suff = i.suffix.lower()
                if is_valid_image_suffix(suff):
                    candidate_image_data_files.append((i, suff))

        candidate_image_data_files.sort(
            key=lambda item: image_suffix_priority[item[1]]
        )

        image_data_file = candidate_image_data_files[0][0]

        interleave_test = suffix_to_format_map.get(image_data_file.suffix)

        if (interleave_test is not None) and (
            interleave_test != self.interleave
        ):
            raise ValueError(
                f"Loaded data ({image_data_file}) does not match the"
                f"registered interleave ({self.interleave})"
            )

        if image_data_file.suffix.lower() in [".img", ".bsq", ".bil", ".bip"]:
            mmap = np.memmap(
                image_data_file,
                dtype=np.dtype(self.data_type),
                shape=self.shape_tuple,
            )
            dat.array = xr.DataArray(mmap)
        elif image_data_file.suffix.lower() == ".hdf5":
            raise NotImplementedError("HDF5 file not implemented yet.")
        elif image_data_file.suffix.lower() == ".zarr":
            arr = xr.open_dataarray(image_data_file, engine="zarr")
            dat.yname = str(arr.dims[0])
            dat.xname = str(arr.dims[1])
            dat.zname = str(arr.dims[2])
            dat.array = arr

        return dat
