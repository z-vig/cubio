from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt
from astropy import units as u
from numpy import ma

from .units import SpectralUnit, SpectralUnitLiteral
from .validation import validate_spectral_domain


@dataclass
class SpectralDomain:
    quantity: u.Quantity

    def __post_init__(self):
        validate_spectral_domain(self.quantity)
        self.nbands = self.quantity.size
        self.masker = DomainMasker(self)

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
        return cls(quantity)

    @property
    def values(self) -> ma.MaskedArray:
        return ma.array(self.quantity.value, mask=self.masker._mask)

    def to(self, unit: SpectralUnit | SpectralUnitLiteral) -> None:
        unit = SpectralUnit(unit)
        self.quantity = self.quantity.to(unit.ap_unit, equivalencies=u.spectral())


class DomainMasker:
    def __init__(self, spectral_domain: SpectralDomain) -> None:
        self.domain = spectral_domain
        self._mask: npt.NDArray[np.bool] = np.zeros(self.domain.quantity.size, np.bool)

    def add_to_mask(self, mask: npt.NDArray) -> None:
        self._mask |= mask

    def crop_values(self, min: float, max: float) -> None:
        crop_mask = (self.domain.quantity.value < min) | (
            self.domain.quantity.value > max
        )
        self.add_to_mask(crop_mask)

    def exclude_slice(self, min: float, max: float) -> None:
        slice_mask = (self.domain.quantity.value > min) & (
            self.domain.quantity.value < max
        )
        self.add_to_mask(slice_mask)
