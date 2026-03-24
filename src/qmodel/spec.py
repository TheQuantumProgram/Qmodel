"""Canonical in-memory specification for quantum program experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GateSpec:
    """One gate occurrence in program order."""

    name: str
    targets: list[str]
    controls: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None


@dataclass(slots=True)
class MeasurementSpec:
    """Terminal measurement description."""

    qubits: list[str]
    basis: str = "computational"
    classical_bits: list[str] | None = None


@dataclass(slots=True)
class UnitSpec:
    """Human-specified abstract unit view."""

    qubits: list[str]
    name: str | None = None
    role: str | None = None


@dataclass(slots=True)
class OrganizationTransitionSpec:
    """One linear organization transition labeled by a gate index."""

    gate_index: int
    next_state: str


@dataclass(slots=True)
class OrganizationStateSpec:
    """One organization state in an explicit linear state chain."""

    name: str
    units: list[UnitSpec]
    transition: OrganizationTransitionSpec | None = None


@dataclass(slots=True)
class OrganizationScheduleSpec:
    """Explicit organization-state chain aligned with the gate sequence."""

    initial_state: str
    states: list[OrganizationStateSpec]


@dataclass(slots=True)
class AssertionSpec:
    """Verification property specification."""

    kind: str
    target: dict[str, Any]
    comparator: str | None = None
    threshold: float | None = None
    name: str | None = None


@dataclass(slots=True)
class QuantumProgramSpec:
    """Canonical specification shared by all backends."""

    program_name: str
    qubits: list[str]
    gates: list[GateSpec]
    initial_state: str = "zero"
    measurement: MeasurementSpec | None = None
    units: list[UnitSpec] = field(default_factory=list)
    organization_schedule: OrganizationScheduleSpec | None = None
    assertions: list[AssertionSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
