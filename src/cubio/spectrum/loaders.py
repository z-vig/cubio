from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Literal

import xarray as xr

type CollectionFileTypeLiteral = Literal[".zarr"]


class CollectionFileType(StrEnum):
    ZARR = ".zarr"


type LoaderFunc = Callable[[Path], xr.Dataset]


def load_zarr(filename: Path) -> xr.Dataset:
    return xr.open_zarr(filename)


LOAD_DISPATCH: dict[str, LoaderFunc] = {".zarr": load_zarr}


def load_collection(filename: Path | str) -> xr.Dataset:
    func = LOAD_DISPATCH.get(Path(filename).suffix)
    if func is None:
        raise ValueError(f"Wrong file type for collection: {filename}")
    return func(Path(filename))
