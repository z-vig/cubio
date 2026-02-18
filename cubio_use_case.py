# flake8: noqa
import xarray as xr
import tracemalloc
import numpy as np
from cubio.cube_context import CubeContext
from cubio.cube_data import CubeData
import sys
from pathlib import Path


def print_memory(name: str, size: int) -> None:
    if 1e3 < size < 1e6:
        print(f"{name} --> {size*1e-3:.3f} kB")
    elif 1e6 < size < 1e9:
        print(f"{name} --> {size*1e-6:.3f} MB")
    elif size > 1e9:
        print(f"{name} --> {size*1e-9:.3f} GB")
    else:
        print(f"{name} --> {size} bytes")


def list_current_peak():
    current, peak = tracemalloc.get_traced_memory()
    print_memory("Current", current)
    print_memory("Peak", peak)


def list_locals(local_call: dict[str, object]):
    local_list = {k: v for k, v in local_call.items()}
    for i, j in local_list.items():
        print_memory(i, sys.getsizeof(j))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from cubio.geotransform import GeotransformModel
    from scipy.interpolate import LinearNDInterpolator

    tracemalloc.start()

    cc = CubeContext.from_json("D:/cube_testing/M3G20090208T160125_LOC.json")
    cub = cc.lazy_load_data()
    cub.transpose_to("BIP")
    ul_long, ul_lat = cub.array[0, 0, :2]
    lr_long, lr_lat = cub.array[-1, -1, :2]
    ur_long, ur_lat = cub.array[0, -1, :2]

    # print(ul_long, ur_long)
    # print((ul_long - ur_long) / cc.ncols)
    # print(ul_lat, lr_lat)
    gtrans = GeotransformModel.fromarraysize(
        ul_lat, ul_long, lr_lat, lr_long, cc.nrows, cc.ncols
    )

    long_vals = np.array(cub.array[:, :, 0])
    long_vals = long_vals.reshape(long_vals.size)
    lat_vals = np.array(cub.array[:, :, 1])
    lat_vals = lat_vals.reshape(lat_vals.size)

    pts = np.column_stack((long_vals, lat_vals))

    interp = LinearNDInterpolator(
        pts, 
    )

    # print(gtrans)
    cub.geotransform = gtrans

    diff_long = abs(cub.array[:, 0, 0] - cub.array[:, -1, 0])
    plt.plot(cub.array[:, 0, 1], diff_long)
    plt.xlabel("Latitude")
    plt.ylabel("Width of Raster (In Longitude)")

    # plt.plot(cub.array["Longitude"])

    # list_locals(locals())
    print("\n===================\nMemory Allocation:")
    list_current_peak()
    tracemalloc.stop()

    plt.show()
