from pathlib import Path

import xarray as xr

type ArrayEncoding = dict[str, dict[str, tuple[int, ...]]]


class ZarrWriter:
    def __init__(self, path: Path | str, append_dim: str | None = None) -> None:
        self.path = Path(path)
        self.append_dim = append_dim
        self._initialized = self.path.exists()

    def create(
        self, template: xr.Dataset, encoding: ArrayEncoding, overwrite: bool = False
    ):
        if self._initialized and (not overwrite):
            raise FileExistsError(f"{self.path}. To overwrite, set overwrite=True.")

        if (template.sizes.get(self.append_dim) is not None) and (
            template.sizes[self.append_dim] != 0
        ):
            raise ValueError("Template append dimension must be empty.")

        self._ensure_append_dim(template)

        template.to_zarr(self.path, mode="w", encoding=encoding, consolidated=False)
        self._initialized = True

    def append(self, batch: xr.Dataset) -> None:
        if not self._initialized:
            raise RuntimeError("Create the store before appending using .create()")

        self._ensure_append_dim(batch)

        if batch.sizes[self.append_dim] == 0:
            return

        batch.to_zarr(
            self.path, mode="a", append_dim=self.append_dim, consolidated=False
        )

    def overwrite(self, data: xr.Dataset, region: dict[str, slice]):
        if not self._initialized:
            raise RuntimeError("Create the store before appending using .create()")
        data.to_zarr(self.path, mode="r+", region=region, consolidated=False)

    def load(self) -> xr.Dataset:
        if not self._initialized:
            raise RuntimeError("The store has not been created.")

        return xr.open_zarr(self.path, consolidated=False)

    def _ensure_append_dim(self, test: xr.Dataset):
        if self.append_dim is None:
            return
        if self.append_dim not in test.dims:
            raise ValueError(f"Missing append dimension: {self.append_dim}")
