"""Abstract state representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit.quantum_info import DensityMatrix


@dataclass(slots=True)
class ReconstructionCertificate:
    """One internal certificate for reconstructing a workspace state."""

    certificate_id: str
    qubits: tuple[str, ...]
    witness_rho: DensityMatrix
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AbstractUnitState:
    """One abstract unit view with projector and executable witness state."""

    qubits: tuple[str, ...]
    projector: np.ndarray
    witness_rho: DensityMatrix
    name: str | None = None
    certificate_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AbstractState:
    """One abstract state along the gate-labeled execution trace."""

    units: tuple[AbstractUnitState, ...]
    position: int
    certificates: tuple[ReconstructionCertificate, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AbstractTransition:
    """One labeled abstract transition between two abstract states."""

    source_id: str
    target_id: str
    label: str
    affected_views: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def support_projector(rho: DensityMatrix, tol: float = 1e-9) -> np.ndarray:
    """Return the orthogonal projector onto the support of a density matrix."""

    if not isinstance(rho, DensityMatrix):
        raise TypeError("rho must be a qiskit.quantum_info.DensityMatrix")

    hermitian = (rho.data + rho.data.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    support_indices = [index for index, value in enumerate(eigenvalues) if value > tol]

    if not support_indices:
        return np.zeros_like(hermitian)

    support_vectors = eigenvectors[:, support_indices]
    projector = support_vectors @ support_vectors.conj().T
    return projector


def abstract_local_state(
    rho: DensityMatrix,
    qubits: list[str] | tuple[str, ...],
    name: str | None = None,
    certificate_ids: tuple[str, ...] = (),
) -> AbstractUnitState:
    """Construct one abstract unit from a local witness density matrix."""

    qubit_tuple = tuple(qubits)
    if not qubit_tuple:
        raise ValueError("qubits must be non-empty")

    return AbstractUnitState(
        qubits=qubit_tuple,
        projector=support_projector(rho),
        witness_rho=rho,
        name=name,
        certificate_ids=certificate_ids,
    )
