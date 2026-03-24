"""Parser entry points for declarative quantum model files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from qmodel.spec import (
    AssertionSpec,
    GateSpec,
    MeasurementSpec,
    OrganizationScheduleSpec,
    OrganizationStateSpec,
    OrganizationTransitionSpec,
    QuantumProgramSpec,
    UnitSpec,
)
from qmodel.validation import SpecValidationError, validate_program_spec


class QModelParseError(ValueError):
    """Raised when a `.qmodel` file cannot be parsed into `QuantumProgramSpec`."""


_SUPPORTED_FORMAT = "qmodel-v1"


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QModelParseError(f"{context} must be a mapping")
    return value


def _optional_mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    return _require_mapping(value, context)


def _require_string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise QModelParseError(f"{context} must be a list of strings")
    return value


def _parse_gate(entry: Any, index: int) -> GateSpec:
    gate = _require_mapping(entry, f"gates[{index}]")
    return GateSpec(
        name=str(gate.get("name", "")).strip(),
        targets=_require_string_list(gate.get("targets"), f"gates[{index}].targets"),
        controls=_require_string_list(gate.get("controls", []), f"gates[{index}].controls"),
        params=_optional_mapping(gate.get("params"), f"gates[{index}].params"),
        label=gate.get("label"),
    )


def _parse_measurement(entry: Any) -> MeasurementSpec:
    measurement = _require_mapping(entry, "measurement")
    return MeasurementSpec(
        qubits=_require_string_list(measurement.get("qubits"), "measurement.qubits"),
        basis=str(measurement.get("basis", "computational")),
        classical_bits=(
            _require_string_list(measurement.get("classical_bits"), "measurement.classical_bits")
            if measurement.get("classical_bits") is not None
            else None
        ),
    )


def _parse_unit(entry: Any, context: str) -> UnitSpec:
    unit = _require_mapping(entry, context)
    return UnitSpec(
        qubits=_require_string_list(unit.get("qubits"), f"{context}.qubits"),
        name=unit.get("name"),
        role=unit.get("role"),
    )


def _parse_organization_transition(
    entry: Any, context: str
) -> OrganizationTransitionSpec:
    transition = _require_mapping(entry, context)
    gate_index = transition.get("gate_index")
    if not isinstance(gate_index, int):
        raise QModelParseError(f"{context}.gate_index must be an integer")

    next_state = transition.get("next_state")
    if not isinstance(next_state, str) or not next_state.strip():
        raise QModelParseError(f"{context}.next_state must be a non-empty string")

    return OrganizationTransitionSpec(gate_index=gate_index, next_state=next_state.strip())


def _parse_organization_state(entry: Any, index: int) -> OrganizationStateSpec:
    state = _require_mapping(entry, f"organization_schedule.states[{index}]")
    name = state.get("name")
    if not isinstance(name, str) or not name.strip():
        raise QModelParseError(
            f"organization_schedule.states[{index}].name must be a non-empty string"
        )

    units_value = state.get("units")
    if not isinstance(units_value, list):
        raise QModelParseError(f"organization_schedule.states[{index}].units must be a list")

    transition_value = state.get("transition")
    transition = (
        _parse_organization_transition(
            transition_value,
            f"organization_schedule.states[{index}].transition",
        )
        if transition_value is not None
        else None
    )
    return OrganizationStateSpec(
        name=name.strip(),
        units=[
            _parse_unit(
                unit_entry,
                f"organization_schedule.states[{index}].units[{unit_index}]",
            )
            for unit_index, unit_entry in enumerate(units_value)
        ],
        transition=transition,
    )


def _parse_organization_schedule(entry: Any) -> OrganizationScheduleSpec:
    schedule = _require_mapping(entry, "organization_schedule")
    initial_state = schedule.get("initial_state")
    if not isinstance(initial_state, str) or not initial_state.strip():
        raise QModelParseError("organization_schedule.initial_state must be a non-empty string")

    states_value = schedule.get("states")
    if not isinstance(states_value, list):
        raise QModelParseError("organization_schedule.states must be a list")

    return OrganizationScheduleSpec(
        initial_state=initial_state.strip(),
        states=[
            _parse_organization_state(entry, index)
            for index, entry in enumerate(states_value)
        ],
    )


def _parse_assertion(entry: Any) -> AssertionSpec:
    assertion = _require_mapping(entry, "assertion")
    return AssertionSpec(
        kind=str(assertion.get("kind", "")).strip(),
        target=_require_mapping(assertion.get("target"), "assertion.target"),
        comparator=assertion.get("comparator"),
        threshold=assertion.get("threshold"),
        name=assertion.get("name"),
    )


def parse_qmodel_file(path: str) -> QuantumProgramSpec:
    model_path = Path(path)

    try:
        with model_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise QModelParseError(f"qmodel file not found: {model_path}") from exc
    except yaml.YAMLError as exc:
        raise QModelParseError(f"Invalid YAML in {model_path}: {exc}") from exc

    root = _require_mapping(data, "qmodel root")

    file_format = root.get("format")
    if file_format != _SUPPORTED_FORMAT:
        raise QModelParseError(
            f"format must be {_SUPPORTED_FORMAT!r}, got {file_format!r}"
        )

    gates_value = root.get("gates")
    if not isinstance(gates_value, list):
        raise QModelParseError("gates must be a list")

    units_value = root.get("units", [])
    if not isinstance(units_value, list):
        raise QModelParseError("units must be a list")

    spec = QuantumProgramSpec(
        program_name=str(root.get("program_name", "")).strip(),
        qubits=_require_string_list(root.get("qubits"), "qubits"),
        gates=[_parse_gate(entry, index) for index, entry in enumerate(gates_value)],
        initial_state=str(root.get("initial_state", "zero")),
        measurement=(
            _parse_measurement(root.get("measurement"))
            if root.get("measurement") is not None
            else None
        ),
        units=[
            _parse_unit(entry, f"units[{index}]")
            for index, entry in enumerate(units_value)
        ],
        organization_schedule=(
            _parse_organization_schedule(root.get("organization_schedule"))
            if root.get("organization_schedule") is not None
            else None
        ),
        assertions=[_parse_assertion(root.get("assertion"))],
        metadata=_optional_mapping(root.get("metadata"), "metadata"),
    )

    try:
        validate_program_spec(spec)
    except SpecValidationError as exc:
        raise QModelParseError(f"Invalid qmodel specification in {model_path}: {exc}") from exc

    return spec
