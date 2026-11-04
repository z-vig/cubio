from astropy import units as u


def validate_array_shape(values: u.Quantity) -> None:
    if values.ndim != 1:
        raise ValueError("Array must be 1D.")


def validate_spectral_domain(values: u.Quantity) -> None:
    validate_array_shape(values)
