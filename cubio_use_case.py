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
    # flake8: noqa

    import matplotlib.pyplot as plt
    from cubio.conevnience_functions import (
        read_measurement_envi_file_context,
        read_spectral_envi_file_context,
    )

    tracemalloc.start()

    rdn_fp = "D:/moon_data/m3/Gruithuisen_Region/M3G20090208T194335/pds_data/L1/M3G20090208T194335_V03_RDN.IMG"
    loc_fp = "D:/moon_data/m3/Gruithuisen_Region/M3G20090208T194335/pds_data/L1/M3G20090208T194335_V03_LOC.IMG"
    obs_fp = "D:/moon_data/m3/Gruithuisen_Region/M3G20090208T194335/pds_data/L1/M3G20090208T194335_V03_OBS.IMG"

    rdn_ctxt = read_spectral_envi_file_context(rdn_fp, "RDN")
    loc_ctxt = read_measurement_envi_file_context(loc_fp, "LOC")
    obs_ctxt = read_measurement_envi_file_context(obs_fp, "OBS")

    for i in [rdn_ctxt, loc_ctxt, obs_ctxt]:
        i.write_envi_hdr(Path(rdn_fp).with_suffix(".hdr"))

    # list_locals(locals())
    print("\n===================\nMemory Allocation:")
    list_current_peak()
    tracemalloc.stop()

    plt.show()
