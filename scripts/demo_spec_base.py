import numpy as np

from cubio.spectrum import SpectrumBase

sb = SpectrumBase.from_values(
    np.linspace(500, 3000),
    "nm",
    "wavelength",
    "reflectance",
    storage_directory="D:/",
)

# sb = SpectrumBase.from_json("D:/")
