from pathlib import Path

from cubio.cube_context import CubeContext
from cubio.cube_data import CubeData
from cubio.types import (
    CubeArrayFormat,
    cube_array_suffix_map,
    RasterioProfile,
)

import rasterio as rio  # type: ignore


def write_envi(
    dst_fp: Path | str,
    cube_context: CubeContext,
    cube_data: CubeData,
    interleave: CubeArrayFormat,
) -> None:
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
    save_fp = Path(
        dst_fp,
        cube_context.data_filename.with_suffix(
            cube_array_suffix_map[interleave]
        ),
    )
    r_cube = cube_data.transpose_to_rasterio()
    with rio.open(save_fp, "w", **prf) as f:
        f.write(r_cube)
    cube_context.write_envi_hdr(dst=dst_fp, use_image_name=True)
