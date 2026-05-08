"""
Masking operations for CubeData class.
"""

# Dependencies
import xarray as xr
import numpy as np

# Local
from cubio.cube_mask import CubeMask
from cubio.cube_dims import CubeDims
from cubio.types import MaskType
from .core import CubeDataCore
from .validation import array_is_set


class MaskingMixIn(CubeDataCore):
    """
    # MaskingMixIn
    Adds masking capabilities to the core `CubeData` class.

    Notes
    -----
    Adds in the mask property, which is a `CubeMask` type object. This
    object has two different masks, the "xymask", which applies over
    spatial dimensions (the "front" of the cube) and the "zmask", which
    applies over the measured dimension (the "back" of the cube).
    """

    @property
    def masked_array(self) -> xr.DataArray:
        return self._apply_mask()

    @property
    def mask(self) -> CubeMask:
        """Mask property of cube data."""
        if not hasattr(self, "_mask"):
            self._mask = CubeMask.transparent(
                self.array,
                cube_dims=CubeDims(
                    self.ydim_name, self.xdim_name, self.zdim_name
                ),
            )
        return self._mask

    @mask.setter
    def mask(self, value: CubeMask) -> None:
        self._mask = value

    def reset_mask(self, which: MaskType = "both") -> None:
        """
        Resets the current cube mask.

        Parameters
        ----------
        which: MaskType
            Which mask(s) to reset: "both", "xy" or "z".
        """
        if which == "both":
            self.mask = CubeMask.transparent(self.array)
        elif which == "xy":
            old_zmask = self.mask.get_zmask().data
            self.mask = CubeMask.transparent(self.array)
            self.mask.add_to_zmask(old_zmask)
        elif which == "z":
            old_xymask = self.mask.get_xymask().data
            self.mask = CubeMask.transparent(self.array)
            self.mask.add_to_xymask(old_xymask)

    def add_nodata_mask(self) -> None:
        """Adds a mask to the current cube mask based on the nodata value."""
        nodata_here = (self.array == self.nodata).any(dim=self.zdim_name)
        self.mask.add_to_xymask(nodata_here.data)

    def _apply_mask(
        self,
        which: MaskType = "both",
    ) -> xr.DataArray:
        """
        Applies mask to the data cube.

        Parameters
        ----------
        which: MaskType, default="both"
            Which mask to apply: {"both", "xy", "z"}.
        drop: bool, default=False
            Whether to drop the masked coordinates from the dataarray.
        """
        self._array = array_is_set(self._array)  # Validation
        masks = {
            "both": ~self.mask.get_xymask() & ~self.mask.get_zmask(),
            "xy": ~self.mask.get_xymask(),
            "z": ~self.mask.get_zmask(),
        }
        return self._array.where(masks[which], np.nan, drop=False)
