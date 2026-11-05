from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from astropy import units as u
from numpy import ma
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
)

from .units import SpectralUnit, SpectralUnitLiteral
from .validation import NumpyArrayMask1D, QuantityType


class DomainMasker(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    mask: NumpyArrayMask1D = Field(default=np.zeros(1, dtype=bool))
    _domain: "SpectralDomain" = PrivateAttr()

    def bind(self, domain: "SpectralDomain") -> None:
        self._domain = domain
        if self.mask.size != self._domain.nbands:
            self.mask = np.zeros(self._domain.nbands, dtype=bool)

    def add_to_mask(self, mask: npt.NDArray[np.bool]) -> None:
        self.mask |= mask

    def crop_values(self, min: float, max: float) -> None:
        crop_mask = (self._domain.quantity.value < min) | (
            self._domain.quantity.value > max
        )
        self.add_to_mask(crop_mask)

    def exclude_slice(self, min: float, max: float) -> None:
        slice_mask = (self._domain.quantity.value > min) & (
            self._domain.quantity.value < max
        )
        self.add_to_mask(slice_mask)


class SpectralDomain(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    quantity: QuantityType
    masker: DomainMasker = Field(default_factory=DomainMasker)

    def model_post_init(self, context: Any) -> None:
        self.masker.bind(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpectralDomain):
            return NotImplemented

        return ma.allequal(self.values, other.values)

    @classmethod
    def from_values(
        cls, values: npt.ArrayLike, unit: SpectralUnit | SpectralUnitLiteral
    ) -> Self:
        unit = SpectralUnit(unit)
        quantity = np.asarray(values) * unit.ap_unit
        return cls(quantity=quantity)

    @classmethod
    def from_json(cls, filepath: Path | str) -> Self:
        with open(filepath, "r") as f:
            return cls.model_validate_json(f.read(), context="serialized")

    @property
    def values(self) -> ma.MaskedArray:
        return ma.array(self.quantity.value, mask=self.masker.mask)

    @computed_field
    @property
    def nbands(self) -> int:
        return self.quantity.value.size

    def to(self, unit: SpectralUnit | SpectralUnitLiteral) -> None:
        unit = SpectralUnit(unit)
        self.quantity = self.quantity.to(unit.ap_unit, equivalencies=u.spectral())

    def to_json(self, save_directory: Path | str, save_name: str) -> None:
        savepath = Path(save_directory, save_name).with_suffix(".json")
        json_str = self.model_dump_json(indent=2)
        with open(savepath, "w") as f:
            f.write(json_str)

    # === Exposed Masker Methods ===
    def add_to_mask(self, mask: npt.NDArray[np.bool]) -> None:
        self.masker.add_to_mask(mask)

    def crop_values(self, min: float, max: float) -> None:
        self.masker.crop_values(min, max)

    def exclude_slice(self, min: float, max: float) -> None:
        self.masker.exclude_slice(min, max)
