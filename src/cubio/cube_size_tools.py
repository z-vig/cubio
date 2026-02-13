# Built-ins
from dataclasses import dataclass

# Dependencies
import numpy as np

# Local imports
from cubio.types import CubeArrayFormat


@dataclass
class CubeSize:
    rows: int
    columns: int
    bands: int


def get_cube_size(
    arr: np.ndarray | np.memmap, format: CubeArrayFormat
) -> CubeSize:
    if format == "BIL":
        lines, bands, samples = arr.shape
    elif format == "BIP":
        lines, samples, bands = arr.shape
    elif format == "BSQ":
        bands, samples, lines = arr.shape

    return CubeSize(lines, samples, bands)
