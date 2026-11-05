from typing import Annotated, TypedDict, TypeGuard

import astropy.units as u
import numpy as np
import numpy.typing as npt
from pydantic import AfterValidator, BeforeValidator, PlainSerializer


class QuantityDict(TypedDict):
    value: list[float]
    unit: str


def is_quantity_dict(data: object) -> TypeGuard[QuantityDict]:
    if not isinstance(data, dict):
        return False

    values = data.get("value")
    if not isinstance(values, list):
        return False

    if not values:
        return False

    if not all(type(i) in (int, float) for i in values):
        return False

    return isinstance(data.get("unit"), str)


def serialize_quantity(quantity: u.Quantity) -> QuantityDict:
    return {"value": quantity.value.tolist(), "unit": str(quantity.unit)}


def deserialize_quantity(data: QuantityDict) -> u.Quantity:
    return np.asarray(data["value"]) * u.Unit(data["unit"])


def validate_quantity_before(value: object) -> u.Quantity | None:
    if isinstance(value, u.Quantity):
        return value

    if is_quantity_dict(value):
        return np.asarray(value["value"]) * u.Unit(value["unit"])


def validate_array_shape(values: u.Quantity) -> None:
    if values.ndim != 1:
        raise ValueError("Array must be 1D.")


def validate_quantity_after(values: u.Quantity) -> u.Quantity:
    validate_array_shape(values)
    return values


QuantityType = Annotated[
    u.Quantity,
    BeforeValidator(validate_quantity_before),
    PlainSerializer(serialize_quantity, return_type=QuantityDict),
    AfterValidator(validate_quantity_after),
]


def validate_1d_mask(value: object) -> npt.NDArray[np.bool]:
    array = np.asarray(value)

    if not np.isdtype(array.dtype, kind=np.bool):
        raise TypeError("Array must be of boolean type.")

    if array.ndim != 1:
        raise ValueError("Array must be 1D.")

    return array


NumpyArrayMask1D = Annotated[
    npt.NDArray[np.bool],
    BeforeValidator(validate_1d_mask),
    PlainSerializer(lambda array: array.tolist(), return_type=list, when_used="json"),
]
