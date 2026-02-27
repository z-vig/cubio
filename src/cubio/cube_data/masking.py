# Dependencies
import xarray as xr
import numpy as np

# Local
from cubio.cube_mask import CubeMask, MaskBuilder
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
    def mask(self) -> CubeMask:
        if not hasattr(self, "_mask"):
            self._builder: MaskBuilder = {
                "shape": self.shape,
                "xdim_name": self.xdim_name,
                "ydim_name": self.ydim_name,
                "zdim_name": self.zdim_name,
            }
            self._mask = CubeMask.transparent(**self._builder)
        return self._mask

    @mask.setter
    def mask(self, value: CubeMask) -> None:
        self._mask = value

    @property
    def array(self) -> xr.DataArray:
        super().array
        return self._apply_mask()

    @array.setter
    def array(self, value: xr.DataArray) -> None:
        # This line is directly from property inheritance example:
        # https://gist.github.com/Susensio/979259559e2bebcd0273f1a95d7c1e79
        super(MaskingMixIn, type(self)).array.fset(self, value)  # type: ignore

    def reset_mask(self, which: MaskType = "both") -> None:
        """
        Resets the current cube mask.

        Parameters
        ----------
        which: MaskType
            Which mask(s) to reset: "both", "xy" or "z".
        """

        if which == "both":
            self.mask = CubeMask.transparent(**self._builder)
        elif which == "xy":
            old_zmask = self.mask.zmask
            self.mask = CubeMask.transparent(**self._builder)
            self.mask.add_to_zmask(old_zmask)
        elif which == "z":
            old_xymask = self.mask.xymask
            self.mask = CubeMask.transparent(**self._builder)
            self.mask.add_to_xymask(old_xymask)

    def add_nodata_mask(self) -> None:
        """Adds a mask to the current cube mask based on the nodata value."""
        valid_array = array_is_set(self._array)
        nodata = valid_array[:, :, 0].drop_vars(self.zdim_name) == self.nodata
        self.mask.add_to_xymask(nodata)

    def _apply_mask(
        self,
        which: MaskType = "both",
        drop: bool = False,
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
        valid_array = array_is_set(self._array)  # Validation
        masks = {
            "both": ~self.mask.xymask & ~self.mask.zmask,
            "xy": ~self.mask.xymask,
            "z": ~self.mask.zmask,
        }
        return valid_array.where(masks[which], np.nan, drop=drop)

    def get_unmasked_array(self, ignore: MaskType = "both") -> xr.DataArray:
        """
        Get the unmasked version of the data cube.

        Parameters
        ----------
        ignore: MaskType
            Which mask(s) to ignore: "both", "xy" or "z".
        """
        valid_array = array_is_set(self._array)

        if ignore == "both":
            return valid_array
        elif ignore == "xy":
            return self._apply_mask("z")
        elif ignore == "z":
            return self._apply_mask("xy")
