from pathlib import Path
from typing import Literal, TypeAlias
from collections.abc import Callable

import numpy as np
import xarray as xr
import tifffile as tiff
import dask.array as dsk_array

from cubio.cube_data import CubeData
from cubio.types import (
    ImageSuffix,
    is_valid_image_suffix,
    image_suffix_priority,
    suffix_to_format_map,
)

from .cube_context import CubeContext

DataLoaderFunction: TypeAlias = Callable[[Path, CubeData, CubeContext], None]


def load_envi_compatible(
    data_fp: Path, empty_cube_data: CubeData, cube_context: CubeContext
) -> None:
    mmap = np.memmap(
        data_fp,
        dtype=np.dtype(cube_context.data_type),
        shape=cube_context.shape_tuple,
    )
    empty_cube_data.array = xr.DataArray(mmap)


def load_zarr(
    data_fp: Path, empty_cube_data: CubeData, cube_context: CubeContext
) -> None:
    arr = xr.open_zarr(data_fp).data
    empty_cube_data.ydim_name = str(arr.dims[0])
    empty_cube_data.xdim_name = str(arr.dims[1])
    empty_cube_data.zdim_name = str(arr.dims[2])
    empty_cube_data.array = arr


def load_gtiff(
    data_fp: Path, empty_cube_data: CubeData, cube_context: CubeContext
) -> None:
    zarr = tiff.imread(data_fp, aszarr=True)
    print("READING TIFF")
    darr = dsk_array.from_zarr(zarr, chunks="auto")
    if darr.ndim == 2:
        da = xr.DataArray(darr, dims=("y", "x"))
    elif darr.ndim == 3:
        da = xr.DataArray(darr, dims=("y", "x", "z"))
    else:
        raise ValueError(
            "Loaded data has an invalid number of dimensions: " f"{darr.ndim}."
        )
    empty_cube_data.array = da


def load_hdf5(
    data_fp: Path, empty_cube_data: CubeData, cube_context: CubeContext
) -> None:
    raise NotImplementedError("HDF5 loading has not been implemented yet.")


LOAD_DISPATCH: dict[ImageSuffix, DataLoaderFunction] = {
    ".img": load_envi_compatible,
    ".bil": load_envi_compatible,
    ".bip": load_envi_compatible,
    ".bsq": load_envi_compatible,
    ".tif": load_gtiff,
    ".tiff": load_gtiff,
    ".zarr": load_zarr,
    ".hdf5": load_hdf5,
}


class CubeDataLoaderValidator:
    def __init__(self, cube_context: CubeContext) -> None:
        self.cc = cube_context

    def search_dir(self, search_dir: str | Path | None) -> Path:
        """
        Validates retrieval path, returning the directory to look in for the
        data file path.
        """
        load_from: Path
        if (search_dir is not None) and (
            self.cc._retrieval_path == "NoRetrieval"
        ):
            load_from = Path(search_dir)
        elif (self.cc.retrieval_path != "NoRetrieval") and (
            search_dir is None
        ):
            load_from = Path(self.cc.retrieval_path).parent
        else:
            raise ValueError(
                "If `savefp` is specified, the object must not have been "
                "validated from disk."
            )

        if not load_from.is_dir():
            raise ValueError(f"Search directory does not exist: {load_from}")

        return load_from

    def image_suffix(self, image_data_file: Path) -> ImageSuffix:
        interleave_test = suffix_to_format_map.get(image_data_file.suffix)

        if (interleave_test is not None) and (
            interleave_test != self.cc.interleave
        ):
            raise ValueError(
                f"Loaded data ({image_data_file}) does not match the"
                f"registered interleave ({self.cc.interleave})"
            )

        suffix = image_data_file.suffix.lower()
        if not is_valid_image_suffix(suffix):
            raise ValueError(f"Invalid image type: {suffix}")

        return suffix


class CubeDataLoader:
    """Handles lazy-loading of cube data object."""

    def __init__(
        self,
        cube_context: CubeContext,
        json_fp: Path | Literal["NoRetrieval"] = "NoRetrieval",
    ) -> None:
        self.cc = cube_context
        self._retrieval_path = json_fp
        self._validator = CubeDataLoaderValidator(cube_context)

    def _get_empty_cubedata(self) -> CubeData:
        """
        Using the cube context attribute, return an empty CubeData object.
        """
        return CubeData(
            self.cc.name,
            self.cc.interleave,
            zcoord_label=self.cc.measurement_values,
            z_name=self.cc.measurement_name,
            geotransform=self.cc.geotransform,
            nodata=self.cc.nodata,
        )

    def _find_image_data(self, search_dir: Path) -> Path:
        candidate_image_data_files: list[tuple[Path, ImageSuffix]] = []
        for i in search_dir.iterdir():
            if i.stem == Path(self.cc.data_filename).stem:
                suff = i.suffix.lower()
                if is_valid_image_suffix(suff):
                    candidate_image_data_files.append((i, suff))

        candidate_image_data_files.sort(
            key=lambda item: image_suffix_priority[item[1]]
        )

        image_data_file = candidate_image_data_files[0][0]

        return image_data_file

    def lazy_load_data(self, search_dir: str | Path | None = None) -> CubeData:
        search_dir = self._validator.search_dir(search_dir)
        dat = self._get_empty_cubedata()
        image_data_file = self._find_image_data(search_dir)
        print(f"Lazy Loading from: {image_data_file}")
        suffix = self._validator.image_suffix(image_data_file)

        LOAD_DISPATCH[suffix](image_data_file, dat, self.cc)

        return dat
