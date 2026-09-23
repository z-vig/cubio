from pathlib import Path
from typing import Literal

import rasterio as rio  # type: ignore

from cubio.cube_context import CubeContext, EnviHeaderWriter
from cubio.cube_data import CubeData
from cubio.types import CubeArrayFormat, RasterioProfile, cube_array_suffix_map


def write_envi(
    cube_context: CubeContext,
    cube_data: CubeData,
    interleave: CubeArrayFormat,
    save_directory: Path | str,
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
    dst_fp: Path | str.
        Path to save directory.
    """
    prf: RasterioProfile = {
        "height": cube_context.shape.nrows,
        "width": cube_context.shape.ncolumns,
        "count": cube_context.shape.nbands,
        "crs": cube_context.crs,
        "driver": "ENVI",
        "dtype": cube_context.data_type,
        "interleave": interleave,
        "nodata": cube_context.nodata,
        "transform": cube_context.geotransform.toaffine(),
    }

    if not Path(save_directory).exists():
        Path(save_directory).mkdir(parents=True)
    save_fp = Path(
        save_directory,
        Path(cube_context.data_filename).with_suffix(
            cube_array_suffix_map[interleave]
        ),
    )
    cube_data.transpose_to("BSQ")
    with rio.open(save_fp, "w", **prf) as f:
        f.write(cube_data.array)
    Path(save_fp.with_suffix(".hdr")).unlink()

    cube_context.interleave = interleave
    envi_writer = EnviHeaderWriter(cube_context)
    envi_writer.to_file(dst=save_fp.with_suffix(".hdr"), use_image_name=True)


def write_zarr(
    cube_context: CubeContext,
    cube_data: CubeData,
    save_directory: Path | str,
    mode: Literal["w"] = "w",
) -> None:
    """
    Writes an .zarr directory.

    Parameters
    ----------
    cube_context: CubeContext:
        CubeContext object containing relevant metadata about the cube.
    cube_data: CubeData
        CubeData object containing the data to be written.
    dst_fp: Path | str | None, optional.
        Path to save directory. File name is automatically set by cube context.
        File directory is either set by the function arg or by the retrieval
        path of the Cube Context, if it is set. If this value is not set,
        an error will be returned.
    """
    save_fp = Path(
        save_directory, Path(cube_context.data_filename).with_suffix(".zarr")
    )
    cube_context.interleave = "BIP"
    cube_context.write_json(cube_context.retrieval_path)
    print(f"Saving zarr: {save_fp}")
    if not save_fp.exists():
        cube_data.array.to_zarr(
            save_fp, zarr_format=2, consolidated=True, mode="w"
        )
    else:
        cube_data.array.to_zarr(
            save_fp, zarr_format=2, consolidated=True, mode=mode
        )
