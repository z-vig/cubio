from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
import xarray as xr
from pydantic import BaseModel, Field, PrivateAttr

from cubio.spectral_domain import SpectralDomain
from cubio.spectral_domain.units import SpectralUnit, SpectralUnitLiteral
from cubio.spectrum.loaders import (
    CollectionFileType,
    CollectionFileTypeLiteral,
    load_collection,
)
from cubio.spectrum.writer import ZarrWriter

from .chunk_handling import get_spectral_collection_chunks

type ArrayEncoding = dict[str, dict[str, tuple[int, ...]]]


class SpectrumBase(BaseModel):
    domain_name: str
    value_name: str
    collection_name: str = Field(default="collection")
    nspectra: int = Field(default=0)
    collection_type: CollectionFileType = Field(default=CollectionFileType.ZARR)
    domain: SpectralDomain

    _retrieval_directory: Path | None = PrivateAttr(None)
    _writer: ZarrWriter | None = PrivateAttr(None)

    @classmethod
    def from_values(
        cls,
        domain_values: npt.NDArray,
        domain_unit: SpectralUnit | SpectralUnitLiteral,
        domain_name: str,
        value_name: str,
        collection_name: str = "collection",
        storage_directory: Path | str | None = None,
    ) -> Self:
        if storage_directory is None:
            storage_directory = Path.cwd()
        domain = SpectralDomain.from_values(domain_values, domain_unit)

        specbase = cls(
            domain=domain,
            domain_name=domain_name,
            value_name=value_name,
            collection_name=collection_name,
        )
        specbase.to_json(storage_directory, collection_name)

        filename = Path(storage_directory, collection_name).with_suffix(".json")
        loaded = cls.from_json(filename)
        loaded.writer.create(specbase.template, specbase.encoding, overwrite=True)
        return loaded

    @classmethod
    def from_json(cls, filename: str | Path) -> Self:
        with open(filename, "r") as f:
            specbase = cls.model_validate_json(f.read())
        specbase.retrieval_directory = Path(filename).parent
        return specbase

    def to_json(self, directory: Path | str, name: str) -> None:
        filename = Path(directory, name).with_suffix(".json")
        with open(filename, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @property
    def template(self) -> xr.Dataset:
        nbands = self.domain.nbands
        return xr.Dataset(
            data_vars={
                "spectra": (
                    ("index", "channel"),
                    np.empty((0, nbands), dtype=np.float32),
                )
            },
            coords={"index": np.zeros(0), "channel": self.domain.values},
        )

    @property
    def encoding(self) -> ArrayEncoding:
        chunk_sizes = get_spectral_collection_chunks(
            self.domain.nbands, np.dtype(np.float32), 0.75
        )
        return {"spectra": {"chunks": chunk_sizes}}

    @property
    def retrieval_directory(self) -> Path:
        if self._retrieval_directory is None:
            raise RuntimeError("Retrieval directory was not initiated.")
        return self._retrieval_directory

    @retrieval_directory.setter
    def retrieval_directory(self, value: Path) -> None:
        self._retrieval_directory = value

    @property
    def collection_path(self) -> Path:
        return Path(self.retrieval_directory, self.collection_name).with_suffix(
            self.collection_type
        )

    @property
    def collection(self) -> xr.Dataset:
        return load_collection(self.retrieval_directory)

    @property
    def writer(self) -> ZarrWriter:
        if self._writer is None:
            self._writer = ZarrWriter(self.collection_path, "index")
        return self._writer

    def set_collection_file_type(
        self, type: CollectionFileType | CollectionFileTypeLiteral
    ) -> None:
        type = CollectionFileType(type)
        self.collection_type = type
