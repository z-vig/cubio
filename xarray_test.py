# flake8: noqa
import xarray as xr
import tracemalloc
import numpy as np
from cubio.cube_context import CubeContext


def print_memory():
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current: {current*1e-6:.4f}MB\nPeak: {peak*1e-6:.4f}")


if __name__ == "__main__":
    tracemalloc.start()
    meta = CubeContext.from_json("D:/cube_testing/refl_cube.json")
    test = np.memmap(
        "D:/cube_testing/refl_cube.bil", dtype=meta.data_type, shape=meta.shape
    )
    test.resize((meta.nrows, meta.ncols, meta.nbands))
    xcoords, ycoords = meta.geotransform.generate_coords(
        xsize=meta.ncols, ysize=meta.nrows
    )
    test_xr = xr.DataArray(
        test,
        coords={
            "longitude": xcoords,
            "latitude": ycoords,
            "wavelength": meta.measurement_values,
        },
        dims=("latitude", "longitude", "wavelength"),
    )
    print_memory()
    tracemalloc.stop()
