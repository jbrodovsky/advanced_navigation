"""Loaders and truth interpolation helpers for the drone project."""

from __future__ import annotations

import ast
import operator
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat

from .config import CAMERA_BODY_TRANSLATION_M, CAMERA_BODY_YAW_RAD, CAMERA_MATRIX, DIST_COEFFS, TAG_IDS
from .data_paths import DatasetNotFoundError, find_dataset_dir
from .types import DroneDataset, DronePacket, DroneParameters, TruthSample
from .utils import Rx, Rz, interpolate_rows, make_transform, wrap_angles

DATASET_NAME = "drone-data"


def drone_data_dir(root: str | Path | None = None) -> Path:
    """Return the folder holding ``parameters.txt`` and ``studentdata*.mat``."""
    return find_dataset_dir(DATASET_NAME, root, required=("parameters.txt",))


def default_parameters_path(root: str | Path | None = None) -> Path:
    """Return the path to ``parameters.txt`` in the drone dataset folder."""
    return drone_data_dir(root) / "parameters.txt"


def default_dataset_path(index: int, root: str | Path | None = None) -> Path:
    """Return the path to ``studentdata<index>.mat`` in the drone dataset folder."""
    path = drone_data_dir(root) / f"studentdata{index}.mat"
    if not path.exists():
        raise DatasetNotFoundError(f"{path.name} is not in {path.parent}")
    return path


# ---------------------------------------------------------------------------
# Dataset files
# ---------------------------------------------------------------------------


def _as_float64_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _packet_array(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        return array.astype(np.float64 if array.dtype.kind == "f" else np.int64, copy=False)
    return np.atleast_1d(array)


def _rate_signal(packet: dict) -> tuple[np.ndarray, str]:
    for field in ("omg", "drpy"):
        if field in packet:
            return _as_float64_array(packet[field]), field
    raise KeyError("packet has neither an 'omg' nor a 'drpy' angular-rate field")


def load_drone_dataset(path: str | Path | int) -> DroneDataset:
    """Load one quadrotor dataset and normalize packets into typed records.

    ``path`` may be a file path or an integer index, in which case
    ``studentdata<index>.mat`` is loaded from the drone dataset folder.
    Both on-disk schemas are supported: ``studentdata0`` (``drpy``, no ``p0``)
    and ``studentdata1``-``7`` (``omg``, ``p0``, ``is_ready``). The rate
    signal always lands in ``DronePacket.omg``; ``DronePacket.rate_field``
    records which raw field it came from.
    """
    path = default_dataset_path(path) if isinstance(path, int) else Path(path)
    raw = loadmat(path, simplify_cells=True)
    raw_packets = raw["data"]
    if isinstance(raw_packets, dict):
        raw_packets = [raw_packets]
    packets: list[DronePacket] = []
    for packet in raw_packets:
        rate, rate_field = _rate_signal(packet)
        packets.append(
            DronePacket(
                id=np.atleast_1d(np.asarray(packet["id"], dtype=np.int64)),
                p1=_packet_array(packet["p1"]),
                p2=_packet_array(packet["p2"]),
                p3=_packet_array(packet["p3"]),
                p4=_packet_array(packet["p4"]),
                t=float(packet["t"]),
                rpy=_as_float64_array(packet["rpy"]),
                omg=rate,
                acc=_as_float64_array(packet["acc"]),
                p0=_packet_array(packet["p0"]) if "p0" in packet else None,
                img=np.asarray(packet["img"]) if "img" in packet else None,
                is_ready=bool(packet["is_ready"]) if "is_ready" in packet else None,
                rate_field=rate_field,
            )
        )
    return DroneDataset(
        path=path,
        vicon=np.asarray(raw["vicon"], dtype=np.float64),
        time=np.asarray(raw["time"], dtype=np.float64).reshape(-1),
        packets=packets,
    )


# ---------------------------------------------------------------------------
# parameters.txt
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_scalar_expression(expression: str) -> float:
    """Evaluate a MATLAB-style scalar such as ``pi/4`` or ``-0.5`` safely."""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return float(np.pi)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression {expression!r}")

    return _eval(ast.parse(expression.strip().replace("^", "**"), mode="eval"))


def _parse_matrix(body: str) -> list[list[float]]:
    """Parse the inside of a MATLAB ``[...]`` literal into rows of floats."""
    body = body.replace("...", " ")
    rows = []
    for row_text in body.split(";"):
        tokens = [token for token in re.split(r"[\s,]+", row_text) if token]
        if tokens:
            rows.append([float(token) for token in tokens])
    return rows


def _bracket_after(text: str, header_pattern: str) -> str | None:
    """Return the body of the first ``[...]`` that follows a header comment."""
    header = re.search(header_pattern, text, flags=re.IGNORECASE)
    if header is None:
        return None
    match = re.search(r"\[([^\]]*)\]", text[header.end() :])
    return None if match is None else match.group(1)


class ParameterFileError(ValueError):
    """Raised when ``parameters.txt`` cannot be parsed."""


def parse_drone_parameters(text: str, source: str = "parameters.txt") -> DroneParameters:
    """Parse the text of ``parameters.txt`` into :class:`DroneParameters`.

    Reads the camera matrix, the distortion coefficients, ``XYZ``, ``Yaw``
    (an expression such as ``pi/4``) and the tag-ID grid. Raises
    :class:`ParameterFileError` naming the entry that is missing or malformed;
    the expected values are documented in :mod:`drone_pipeline.config`.
    """

    def fail(item: str, detail: str) -> ParameterFileError:
        return ParameterFileError(
            f"{source}: could not read '{item}' ({detail}). Compare the file with "
            "the documented defaults in drone_pipeline/config.py, or re-download it "
            "with `uv run python scripts/fetch_data.py drone-data`."
        )

    try:
        body = _bracket_after(text, r"camera\s+matrix")
        if body is None:
            raise fail("Camera Matrix", "header or [...] block not found")
        rows = _parse_matrix(body)
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            raise fail("Camera Matrix", f"expected 3 rows of 3 values, got row lengths {[len(row) for row in rows]}")
        camera_matrix = np.array(rows, dtype=np.float64)

        body = _bracket_after(text, r"distortion")
        if body is None:
            raise fail("Distortion parameters", "header or [...] block not found")
        dist_coeffs = np.array(_parse_matrix(body), dtype=np.float64).reshape(-1)
        if dist_coeffs.size != 5:
            raise fail("Distortion parameters", f"expected 5 values, got {dist_coeffs.size}")

        match = re.search(r"\bXYZ\s*=\s*\[([^\]]*)\]", text)
        if match is None:
            raise fail("XYZ", "no 'XYZ = [...]' line")
        translation = np.array(_parse_matrix(match.group(1)), dtype=np.float64).reshape(-1)
        if translation.size != 3:
            raise fail("XYZ", f"expected 3 values, got {translation.size}")

        match = re.search(r"\bYaw\s*=\s*([^;\n]+)", text)
        if match is None:
            raise fail("Yaw", "no 'Yaw = ...' line")
        try:
            yaw = float(_eval_scalar_expression(match.group(1)))
        except (SyntaxError, ValueError, TypeError, ArithmeticError) as error:
            raise fail("Yaw", f"{type(error).__name__}: {error}") from error
        if not np.isfinite(yaw):
            raise fail("Yaw", f"value {yaw} is not a finite real number")

        body = _bracket_after(text, r"tag\s+ids")
        if body is None:
            raise fail("Tag ids", "header or [...] block not found")
        rows = _parse_matrix(body)
        if not rows or len({len(row) for row in rows}) != 1:
            raise fail("Tag ids", "rows have different lengths")
        tag_ids = np.array(rows, dtype=np.int64)
    except ValueError as error:
        if isinstance(error, ParameterFileError):
            raise
        raise ParameterFileError(f"{source}: malformed number in parameter file ({error})") from error

    # Body z is up and the camera looks down: flip about x, then undo the yaw
    # offset. Verified on studentdata1-7 against Vicon (see the handout).
    rotation = Rz(-yaw) @ Rx(np.pi)
    return DroneParameters(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        translation_m=translation,
        yaw_offset_rad=float(yaw),
        tag_ids=tag_ids,
        T_camera_body=make_transform(rotation, translation),
    )


def load_drone_parameters(path: str | Path | None = None) -> DroneParameters:
    """Load and parse ``parameters.txt``.

    With no ``path``, the file is read from the drone dataset folder
    (``data/drone-data/parameters.txt``).
    """
    if path is None:
        path = default_parameters_path()
    path = Path(path)
    return parse_drone_parameters(path.read_text(), source=str(path))


def default_drone_parameters() -> DroneParameters:
    """Build :class:`DroneParameters` from the documented defaults in ``config.py``.

    Useful for tests and examples that must run without the dataset. Prefer
    :func:`load_drone_parameters` whenever the data is available.
    """
    rotation = Rz(-CAMERA_BODY_YAW_RAD) @ Rx(np.pi)
    return DroneParameters(
        camera_matrix=CAMERA_MATRIX.copy(),
        dist_coeffs=DIST_COEFFS.copy(),
        translation_m=CAMERA_BODY_TRANSLATION_M.copy(),
        yaw_offset_rad=float(CAMERA_BODY_YAW_RAD),
        tag_ids=TAG_IDS.copy(),
        T_camera_body=make_transform(rotation, CAMERA_BODY_TRANSLATION_M),
    )


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------


def interpolate_vicon_truth(dataset: DroneDataset, packet_times: np.ndarray | None = None) -> list[TruthSample]:
    """Interpolate Vicon truth onto packet or user-specified timestamps.

    Query times outside the Vicon time span get NaN fields and
    ``valid=False`` rather than a frozen copy of the first or last Vicon
    sample (``studentdata0`` has 75 such packets). Trim or skip them before
    computing errors. Attitude rows are unwrapped before interpolation and
    wrapped back into [-pi, pi) afterwards.
    """
    if packet_times is None:
        packet_times = dataset.packet_times
    query = np.asarray(packet_times, dtype=np.float64)
    vicon = np.array(dataset.vicon, dtype=np.float64, copy=True)
    vicon[3:6] = np.unwrap(vicon[3:6], axis=1)
    interpolated = interpolate_rows(dataset.time, vicon, query, fill_value=np.nan)
    interpolated[3:6] = wrap_angles(interpolated[3:6])
    valid = (query >= dataset.time[0]) & (query <= dataset.time[-1])
    truth_samples: list[TruthSample] = []
    for index, time_s in enumerate(query):
        truth_samples.append(
            TruthSample(
                t=float(time_s),
                position=interpolated[0:3, index],
                rpy=interpolated[3:6, index],
                velocity=interpolated[6:9, index],
                angular_velocity=interpolated[9:12, index],
                valid=bool(valid[index]),
            )
        )
    return truth_samples
