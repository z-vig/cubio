from typing import Self
from dataclasses import dataclass


@dataclass
class CubeDims:
    vdim: str
    hdim: str
    zdim: str

    @classmethod
    def default(cls) -> Self:
        return cls("Ydim", "Xdim", "Zdim")

    @classmethod
    def hyperspectral(cls) -> Self:
        return cls("Latitude", "Longitude", "Wavelengths")

    @classmethod
    def geo_time_series(cls) -> Self:
        return cls("Latitude", "Longitude", "Time")

    def as_list(self) -> list[str]:
        return [self.vdim, self.hdim, self.zdim]
