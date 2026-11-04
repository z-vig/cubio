from enum import StrEnum
from typing import Literal

from astropy import units as u

type SpectralUnitLiteral = Literal["m", "um", "nm", "cm", "1/cm", "1/m", "Hz", "eV"]


class SpectralUnit(StrEnum):
    METERS = "m"
    MICRONS = "um"
    NANOMETERS = "nm"
    CENTIMETERS = "cm"
    INVERSE_CENTIMETERS = "1/cm"
    INVERSE_METER = "1/m"
    HERTZ = "Hz"
    ELECTRONVOLT = "eV"

    @property
    def ap_unit(self) -> u.Unit:
        return u.Unit(self.value)
