from typing import NamedTuple

import numpy as np
import psutil


class SpectralCollectionChunks(NamedTuple):
    rows: int
    cols: int


def available_memory() -> int:
    """Return currently available system memory in bytes."""
    return psutil.virtual_memory().available


def get_spectral_collection_chunks(
    nchannels: int, dtype: np.dtype, fill_percentage: float
) -> SpectralCollectionChunks:
    if (fill_percentage > 1) or (fill_percentage < 0):
        raise ValueError(f"Fill percentage must be between 0  and 1: {fill_percentage}")
    max_bytes = int(available_memory() * fill_percentage)
    itemsize = np.dtype(dtype).itemsize
    row_chunk_size = max_bytes // (itemsize * nchannels)
    return SpectralCollectionChunks(row_chunk_size, nchannels)
