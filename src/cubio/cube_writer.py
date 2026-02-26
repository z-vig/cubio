from pathlib import Path
from typing import Literal

from cubio.cube_context import CubeContext
from cubio.cube_data import CubeData
from cubio.types import (
    CubeArrayFormat,
    cube_array_suffix_map,
    RasterioProfile,
)

import rasterio as rio  # type: ignore


def get_save_directory(
    cube_context: CubeContext, dst_fp: str | Path | None = None
) -> Path:
    save_dir: Path
    if dst_fp is None:
        if cube_context._retrieval_path != "NoRetrieval":
            save_dir = Path(cube_context._retrieval_path).parent
        else:
            raise ValueError(
                "Cube Context retrieval path is not set. The context object"
                " was likely created in a script, rather than loaded from "
                "json."
            )
    else:
        save_dir = Path(dst_fp).parent
    return save_dir


def write_envi(
    cube_context: CubeContext,
    cube_data: CubeData,
    interleave: CubeArrayFormat,
    dst_fp: Path | str | None = None,
) -> None:
    """
    Writes an ENVI-compatible file.

    Parameters
    ----------
    cube_context: CubeContext:
        CubeContext object containing relevant metadata about the cube.
    cube_data: CubeData
        CubeData object containing the data to be written.
    interleave: CubeArrayFormat
        Desired interleave format for the output file. Must be one of "BIP",
        "BIL", or "BSQ".
    dst_fp: Path | str | None, optional.
        Path to save directory. File name is automatically set by cube context.
        File directory is either set by the function arg or by the retrieval
        path of the Cube Context, if it is set. If this value is not set,
        an error will be returned.
    """
    prf: RasterioProfile = {
        "height": cube_context.shape.nrows,
        "width": cube_context.shape.ncolumns,
        "count": cube_context.shape.nbands,
        "crs": cube_context.crs,
        "driver": "ENVI",
        "dtype": cube_context.data_type,
        "interleave": interleave,
        "nodata": -999,
        "transform": cube_context.geotransform.toaffine(),
    }

    save_dir = get_save_directory(cube_context, dst_fp)
    save_fp = Path(
        save_dir,
        cube_context.data_filename.with_suffix(
            cube_array_suffix_map[interleave]
        ),
    )
    r_cube = cube_data.transpose_to_rasterio()
    with rio.open(save_fp, "w", **prf) as f:
        f.write(r_cube)
    Path(save_fp.with_suffix(".hdr")).unlink()

    cube_context.interleave = interleave
    cube_context.write_envi_hdr(dst=dst_fp, use_image_name=True)


def write_zarr(
    cube_context: CubeContext,
    cube_data: CubeData,
    dst_fp: Path | str | None = None,
    mode: Literal["w", "r"] = "r",
) -> None:
    """
    Writes an .zarr directory.

    Parameters
    ----------
    cube_context: CubeContext:
        CubeContext object containing relevant metadata about the cube.
    cube_data: CubeData
        CubeData object containing the data to be written.
    interleave: CubeArrayFormat
        Desired interleave format for the output file. Must be one of "BIP",
        "BIL", or "BSQ".
    dst_fp: Path | str | None, optional.
        Path to save directory. File name is automatically set by cube context.
        File directory is either set by the function arg or by the retrieval
        path of the Cube Context, if it is set. If this value is not set,
        an error will be returned.
    """
    save_dir = get_save_directory(cube_context, dst_fp)
    save_fp = Path(save_dir, cube_context.data_filename.with_suffix(".zarr"))
    print(save_fp)
    if not save_fp.exists():
        cube_data.array.to_zarr(
            save_fp, zarr_format=2, consolidated=True, mode="w"
        )
    else:
        cube_data.array.to_zarr(
            save_fp, zarr_format=2, consolidated=True, mode=mode
        )
