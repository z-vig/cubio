# Built-Ins
from pathlib import Path
from uuid import uuid4

# Dependencies
import spectralio as sio
import rasterio as rio

# Local Imports
from cubio.types import RasterioProfile, NumpyDType
from cubio.cube_context import CubeContext, ContextBuilder
from cubio.geotransform import GeotransformModel
from cubio.data.crs_wkt_strings import GeographicCRS


def read_spectral_envi_file_context(
    fp: Path | str, spectralio_wvl_path: Path | str
) -> CubeContext:
    wvls = sio.read_wvl(spectralio_wvl_path)

    with rio.open(fp, "r") as f:
        prf: RasterioProfile = f.profile

    if prf["crs"] is None:
        crs = GeographicCRS.GCS_MOON_2000
    else:
        crs = prf["crs"]

    context_dict: ContextBuilder = {
        "description": "Testing a cube.",
        "data_filename": Path(fp),
        "nrows": prf["height"],
        "ncols": prf["width"],
        "nbands": prf["count"],
        "crs": crs,
        "geotransform": GeotransformModel.fromaffine(prf["transform"]),
        "hdr_off": 0,
        "data_type": NumpyDType.FLOAT32,
        "interleave": "BIL",
        "nodata": -999,
        "band_names": [f"Band{n+1}({i}nm)" for n, i in enumerate(wvls.values)],
        "measurement_units": "nm",
        "measurement_values": wvls.values,
        "bad_bands": list(wvls.bbl),
        "id": uuid4(),
    }

    return CubeContext.from_builder(context_dict)


if __name__ == "__main__":
    import tracemalloc
    import sys
    import numpy as np

    def print_memory():
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current: {current*1e-6:.4f}MB\nPeak: {peak*1e-6:.4f}")

    def list_local_memory(local_inventory: dict):
        all_bytes = 0
        print("Items in Local Memory:")
        for k, v in local_inventory.items():
            obj_size = sys.getsizeof(v)
            print(f"    ---> {k}: {obj_size} Bytes")
            all_bytes += obj_size
        print(f"Local Object Total Size: {all_bytes*1e-6:.4f} MB")

    tracemalloc.start()
    fp = Path(
        "D:/moon_data/m3/Gruithuisen_Region/M3G20090208T160125/pds_data/L1/"
        "M3G20090208T160125_V03_RDN.IMG"
    )
    wvl_fp = "D:/moon_data/m3/M3G.wvl"
    arr = np.zeros((1000, 1000, 100))
    cb = read_spectral_envi_file_context(fp, wvl_fp)
    cb.write_envi_hdr("D:/cube_testing/refl_cube_big")

    list_local_memory({k: v for k, v in locals().items()})
    # print_memory()
    tracemalloc.stop()
