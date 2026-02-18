# Built-ins
from pathlib import Path
import re
import textwrap
from typing import Literal


def extract_hdr_band_names(hdr_fp: str | Path) -> list[str]:
    ptrn = re.compile(r"band\s*names\s*=\s*{([\s\S]*)}")
    with open(hdr_fp) as src:
        s = src.read()
    match = re.search(ptrn, s)
    if not match:
        raise ValueError("Invalid .HDR format: Cannot find band names.")
    result = match.groups()[0][1:]
    return [i.strip() for i in re.split(r"\s*,\s*\n?", result)]


def extract_hdr_wavelengths(
    hdr_fp: str | Path,
) -> list[float] | Literal["Wavelengths not found."]:
    ptrn = re.compile(r"(?<=\n)wavelength\s*=\s*{([\s\S]*?)}")
    with open(hdr_fp) as src:
        s = src.read()
    match = re.search(ptrn, s)
    if not match:
        return "Wavelengths not found."
    result = match.groups()[0]
    return [float(i.strip()) for i in re.split(r"\s*,\s*\n?", result)]


def extract_hdr_bbl(hdr_fp: str | Path) -> list[int] | Literal["No BBL Found"]:
    ptrn = re.compile(r"(?<=\n)bbl\s*=\s*{([\s\S]*?)}")
    with open(hdr_fp) as src:
        s = src.read()
    match = re.search(ptrn, s)
    if not match:
        return "No BBL Found"
    result = match.groups()[0]
    return [int(i.strip()) for i in re.split(r"\s*,\s*\n?", result)]


def extract_hdr_desc(hdr_fp: str | Path) -> str:
    ptrn = re.compile(r"description\s*=\s*{([\s\S]*?)}")
    with open(hdr_fp) as src:
        s = src.read()
    match = re.search(ptrn, s)
    if not match:
        raise ValueError("Invalid .HDR format: Cannot find description.")
    result = match.groups()[0]
    return result


def replace_hdr_band_names(hdr_fp: str | Path, new_band_names: list[str]):
    ptrn = re.compile(r"band\s*names\s*=\s*{([\s\S]*)}")
    with open(hdr_fp) as src:
        s = src.read()
    match = re.search(ptrn, s)
    if not match:
        raise ValueError("Invalid .HDR format: Cannot find band names.")
    result = match.groups()[0]
    new_hdr_str = s.replace(result, ", ".join(new_band_names))
    with open(hdr_fp, "w") as dst:
        dst.write(new_hdr_str)


def replace_hdr_description(hdr_fp: str | Path, new_desc: str):
    ptrn = re.compile(r"description\s*=\s*{([\s\S]*?)}")
    with open(hdr_fp) as src:
        s = src.read()
    match = re.search(ptrn, s)
    if not match:
        raise ValueError("Invalid .HDR format: Cannot find description.")
    result = match.groups()[0]
    new_hdr_str = s.replace(result, f"\n{textwrap.fill(new_desc, width=80)}")
    with open(hdr_fp, "w") as dst:
        dst.write(new_hdr_str)


def replace_integer_field(
    hdr_fp: str | Path, field_name: str, replace_value: int
) -> None:
    ptrn = re.compile(
        rf"({field_name}\s*=\s*\d+)"
    )  # \nlines\s*=\s*\d\nbands\s*=\s*\d")
    with open(hdr_fp) as src:
        s = src.read()
    match = re.search(ptrn, s)
    if not match:
        raise ValueError(
            f"Invalid .HDR format: Cannot find integer field: {field_name}"
        )
    result = match.groups()[0]
    new_hdr_str = s.replace(result, f"{field_name} = {replace_value}")
    with open(hdr_fp, "w") as dst:
        dst.write(new_hdr_str)


def replace_shape_fields(
    hdr_fp: Path | str, *, samples: int, lines: int, bands: int
):
    replace_integer_field(hdr_fp, "samples", samples)
    replace_integer_field(hdr_fp, "lines", lines)
    replace_integer_field(hdr_fp, "bands", bands)
