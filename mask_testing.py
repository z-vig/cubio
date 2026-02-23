import xarray as xr
from cubio import CubeContext
import matplotlib.pyplot as plt

fp = "D:/moon_data/m3/Gruithuisen_Region/M3G20090208T160125/M3G20090208T160125_RFL_georef.json"  # noqa
cc = CubeContext.from_json(fp)
cb = cc.lazy_load_data()
cb.transpose_to("BIP")
data = cb.array
# mask = xr.

plt.figure()
plt.imshow(data[:, :, 10])
plt.show()
