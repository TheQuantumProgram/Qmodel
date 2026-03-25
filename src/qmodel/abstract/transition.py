"""Abstract transition semantics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace

from qmodel.abstract.state import (
    AbstractState,
    ReconstructionCertificate,
    AbstractTransition,
    AbstractUnitState,
    abstract_local_state,
)
from qmodel.concrete.qiskit_backend import build_circuit
from qmodel.spec import GateSpec, QuantumProgramSpec, UnitSpec


ReconstructionMode = str


@dataclass(slots=True)
class AbstractExecutionTrace:
    """One linear abstract execution trace over a gate sequence."""

    states: tuple[AbstractState, ...]
    transitions: tuple[AbstractTransition, ...]


def gate_support(gate: GateSpec) -> tuple[str, ...]:
    """Return the ordered qubit support of one gate occurrence."""

    support: list[str] = []
    for name in [*gate.controls, *gate.targets]:
        if name not in support:
            support.append(name)
    return tuple(support)


def select_affected_views(
    units: tuple[AbstractUnitState, ...], gate: GateSpec
) -> tuple[AbstractUnitState, ...]:
    """Select all views that intersect the gate support."""

    support = set(gate_support(gate))
    return tuple(unit for unit in units if support & set(unit.qubits))


def select_reconstruction_support_units(
    *,
    all_units: Sequence[AbstractUnitState],
    seed_units: Sequence[AbstractUnitState],
    workspace_qubits: Sequence[str],
    global_qubits: list[str],
) -> tuple[AbstractUnitState, ...]:
    """Select the minimal pre-state support needed to cover one workspace.

    This is coverage-based, not transitive-overlap-based: start from the
    directly affected seed views and only add more pre-state views if some
    workspace qubits are still uncovered.
    """

    workspace_order = tuple(qubit for qubit in global_qubits if qubit in set(workspace_qubits))
    selected: list[AbstractUnitState] = []
    selected_ids: set[tuple[str | None, tuple[str, ...]]] = set()

    for unit in seed_units:
        identity = _unit_identity(unit.name, unit.qubits)
        if identity not in selected_ids:
            selected.append(unit)
            selected_ids.add(identity)

    covered = {qubit for unit in selected for qubit in unit.qubits}
    workspace_set = set(workspace_order)

    while not workspace_set.issubset(covered):
        uncovered = workspace_set - covered
        candidates = []
        for unit in all_units:
            identity = _unit_identity(unit.name, unit.qubits)
            if identity in selected_ids:
                continue
            gain = len(set(unit.qubits) & uncovered)
            if gain == 0:
                continue
            connected_overlap = len(set(unit.qubits) & covered)
            leftmost = min(global_qubits.index(qubit) for qubit in unit.qubits)
            candidates.append((-gain, -connected_overlap, len(unit.qubits), leftmost, unit))

        if not candidates:
            raise ValueError("Current pre-state units do not cover the requested workspace")

        _, _, _, _, chosen = min(candidates)
        selected.append(chosen)
        selected_ids.add(_unit_identity(chosen.name, chosen.qubits))
        covered.update(chosen.qubits)

    return tuple(selected)


def _gate_operator(gate: GateSpec) -> Operator:
    support = list(gate_support(gate))
    spec = QuantumProgramSpec(
        program_name=gate.label or gate.name,
        qubits=support,
        gates=[
            GateSpec(
                name=gate.name,
                targets=gate.targets[:],
                controls=gate.controls[:],
                params=dict(gate.params),
                label=gate.label,
            )
        ],
    )
    return Operator(build_circuit(spec))


def _ordered_union(global_qubits: list[str], units: Sequence[Sequence[str]]) -> tuple[str, ...]:
    names = {qubit for unit in units for qubit in unit}
    return tuple(qubit for qubit in global_qubits if qubit in names)


def _connected_overlap_components(units: Sequence[UnitSpec]) -> list[list[UnitSpec]]:
    remaining = list(range(len(units)))
    components: list[list[UnitSpec]] = []

    while remaining:
        seed = remaining.pop(0)
        component_indices = [seed]
        queue = [seed]

        while queue:
            current = queue.pop(0)
            current_qubits = set(units[current].qubits)
            connected = []
            for index in remaining:
                if current_qubits & set(units[index].qubits):
                    connected.append(index)
            for index in connected:
                remaining.remove(index)
                queue.append(index)
                component_indices.append(index)

        components.append([units[index] for index in sorted(component_indices)])

    return components


def _units_witness_bytes(units: Sequence[AbstractUnitState]) -> int:
    return sum(int(unit.witness_rho.data.nbytes) for unit in units)


def _unit_identity(name: str | None, qubits: Sequence[str]) -> tuple[str | None, tuple[str, ...]]:
    return name, tuple(qubits)


def _copy_unaffected_unit(pre_state: AbstractState, unit: UnitSpec) -> AbstractUnitState | None:
    identity = _unit_identity(unit.name, unit.qubits)
    for current in pre_state.units:
        if _unit_identity(current.name, current.qubits) == identity:
            return abstract_local_state(
                current.witness_rho,
                current.qubits,
                name=current.name,
                certificate_ids=current.certificate_ids,
            )
    return None


def _reduce_scope(
    witness: DensityMatrix, witness_qubits: Sequence[str], scope: Sequence[str]
) -> DensityMatrix:
    scope_list = list(scope)
    missing = [qubit for qubit in scope_list if qubit not in witness_qubits]
    if missing:
        raise ValueError(
            "Requested reduction scope is not contained in witness qubits: "
            + ", ".join(missing)
        )
    trace_out = [index for index, qubit in enumerate(witness_qubits) if qubit not in scope_list]
    return partial_trace(witness, trace_out) if trace_out else witness


def _certificate_map(
    state: AbstractState,
) -> dict[str, ReconstructionCertificate]:
    return {certificate.certificate_id: certificate for certificate in state.certificates}


def _overlap_consistent(
    left_rho: DensityMatrix,
    left_qubits: Sequence[str],
    right_rho: DensityMatrix,
    right_qubits: Sequence[str],
    tol: float = 1e-9,
) -> bool:
    overlap = [qubit for qubit in left_qubits if qubit in right_qubits]
    if not overlap:
        return True

    left_overlap = _reduce_scope(left_rho, left_qubits, overlap)
    right_overlap = _reduce_scope(right_rho, right_qubits, overlap)
    return np.allclose(left_overlap.data, right_overlap.data, atol=tol, rtol=tol)


def _reorder_density_matrix(
    witness: DensityMatrix,
    current_order: Sequence[str],
    target_order: Sequence[str],
) -> DensityMatrix:
    if tuple(current_order) == tuple(target_order):
        return witness

    if set(current_order) != set(target_order):
        raise ValueError("current_order and target_order must contain the same qubits")

    width = len(current_order)
    permutation = [list(current_order).index(qubit) for qubit in target_order]
    reshaped = witness.data.reshape([2] * (2 * width))
    axes = permutation + [index + width for index in permutation]
    reordered = reshaped.transpose(axes).reshape((2**width, 2**width))
    return DensityMatrix(reordered)


def _tensor_join(
    left_rho: DensityMatrix,
    left_qubits: Sequence[str],
    right_rho: DensityMatrix,
    right_qubits: Sequence[str],
    target_order: Sequence[str],
) -> DensityMatrix:
    combined = left_rho.expand(right_rho)
    return _reorder_density_matrix(
        combined,
        tuple(left_qubits) + tuple(right_qubits),
        tuple(target_order),
    )


def _is_diagonal_density_matrix(rho: DensityMatrix, tol: float = 1e-9) -> bool:
    diagonal = np.diag(np.diag(rho.data))
    return np.allclose(rho.data, diagonal, atol=tol, rtol=tol)


def _probability_map(rho: DensityMatrix) -> dict[str, float]:
    width = rho.num_qubits
    probabilities = {}
    for basis_index, basis_probability in enumerate(rho.probabilities()):
        probabilities[format(basis_index, f"0{width}b")[::-1]] = float(
            np.real_if_close(basis_probability)
        )
    return probabilities


def _classical_overlap_join(
    left_rho: DensityMatrix,
    left_qubits: Sequence[str],
    right_rho: DensityMatrix,
    right_qubits: Sequence[str],
    target_order: Sequence[str],
    tol: float = 1e-12,
) -> DensityMatrix:
    overlap = [qubit for qubit in left_qubits if qubit in right_qubits]
    if not overlap:
        return _tensor_join(left_rho, left_qubits, right_rho, right_qubits, target_order)

    left_probabilities = _probability_map(left_rho)
    right_probabilities = _probability_map(right_rho)
    overlap_probabilities = _probability_map(_reduce_scope(left_rho, left_qubits, overlap))
    target_positions = {qubit: index for index, qubit in enumerate(target_order)}
    diagonal = np.zeros(2 ** len(target_order), dtype=complex)

    for left_bits, left_probability in left_probabilities.items():
        if left_probability <= tol:
            continue
        left_overlap = "".join(left_bits[list(left_qubits).index(qubit)] for qubit in overlap)
        overlap_probability = overlap_probabilities.get(left_overlap, 0.0)
        if overlap_probability <= tol:
            continue

        for right_bits, right_probability in right_probabilities.items():
            if right_probability <= tol:
                continue
            right_overlap = "".join(right_bits[list(right_qubits).index(qubit)] for qubit in overlap)
            if left_overlap != right_overlap:
                continue

            union_bits = ["0"] * len(target_order)
            for qubit, bit in zip(left_qubits, left_bits, strict=True):
                union_bits[target_positions[qubit]] = bit
            for qubit, bit in zip(right_qubits, right_bits, strict=True):
                union_bits[target_positions[qubit]] = bit

            diagonal[int("".join(union_bits)[::-1], 2)] = (
                left_probability * right_probability / overlap_probability
            )

    total_probability = float(np.real_if_close(diagonal.sum()))
    if total_probability > tol:
        diagonal /= total_probability
    return DensityMatrix(np.diag(diagonal))


def _shared_certificate_cover(
    state: AbstractState,
    units: Sequence[AbstractUnitState],
    scope_qubits: Sequence[str],
) -> ReconstructionCertificate | None:
    if not units:
        return None

    shared_ids = set(units[0].certificate_ids)
    for unit in units[1:]:
        shared_ids &= set(unit.certificate_ids)

    certificates = _certificate_map(state)
    candidates = [
        certificates[certificate_id]
        for certificate_id in shared_ids
        if certificate_id in certificates and set(scope_qubits).issubset(set(certificates[certificate_id].qubits))
    ]
    if not candidates:
        return None

    return min(candidates, key=lambda certificate: len(certificate.qubits))


def _reconstruct_workspace_state(
    *,
    state: AbstractState,
    units: Sequence[AbstractUnitState],
    workspace_qubits: Sequence[str],
    global_qubits: list[str],
    mode: ReconstructionMode,
) -> DensityMatrix:
    direct_certificate = _shared_certificate_cover(state, units, workspace_qubits)
    if direct_certificate is not None:
        return _reduce_scope(direct_certificate.witness_rho, direct_certificate.qubits, workspace_qubits)

    ordered_units = sorted(
        units,
        key=lambda unit: min(global_qubits.index(qubit) for qubit in unit.qubits),
    )
    assembled_rho = ordered_units[0].witness_rho
    assembled_qubits = tuple(ordered_units[0].qubits)
    assembled_units = [ordered_units[0]]

    for unit in ordered_units[1:]:
        overlap = [qubit for qubit in assembled_qubits if qubit in unit.qubits]
        if not _overlap_consistent(assembled_rho, assembled_qubits, unit.witness_rho, unit.qubits):
            raise ValueError("Overlapping abstract units are inconsistent on their shared qubits")

        if overlap:
            pair_certificate = _shared_certificate_cover(state, [assembled_units[-1], unit], _ordered_union(global_qubits, [assembled_qubits, unit.qubits]))
            if pair_certificate is not None:
                assembled_qubits = _ordered_union(global_qubits, [assembled_qubits, unit.qubits])
                assembled_rho = _reduce_scope(
                    pair_certificate.witness_rho,
                    pair_certificate.qubits,
                    assembled_qubits,
                )
                assembled_units.append(unit)
                continue

            union_qubits = _ordered_union(global_qubits, [assembled_qubits, unit.qubits])
            if _is_diagonal_density_matrix(assembled_rho) and _is_diagonal_density_matrix(
                unit.witness_rho
            ):
                assembled_rho = _classical_overlap_join(
                    assembled_rho,
                    assembled_qubits,
                    unit.witness_rho,
                    unit.qubits,
                    union_qubits,
                )
                assembled_qubits = union_qubits
                assembled_units.append(unit)
                continue

            new_only = [qubit for qubit in unit.qubits if qubit not in assembled_qubits]
            if new_only:
                new_only_rho = _reduce_scope(unit.witness_rho, unit.qubits, new_only)
                union_qubits = _ordered_union(global_qubits, [assembled_qubits, new_only])
                assembled_rho = _tensor_join(
                    assembled_rho,
                    assembled_qubits,
                    new_only_rho,
                    new_only,
                    union_qubits,
                )
                assembled_qubits = union_qubits
            assembled_units.append(unit)
            continue

        union_qubits = _ordered_union(global_qubits, [assembled_qubits, unit.qubits])
        assembled_rho = _tensor_join(
            assembled_rho,
            assembled_qubits,
            unit.witness_rho,
            unit.qubits,
            union_qubits,
        )
        assembled_qubits = union_qubits
        assembled_units.append(unit)

    return _reduce_scope(assembled_rho, assembled_qubits, workspace_qubits)


def _filter_valid_certificate_ids(
    certificate_ids: Sequence[str],
    certificate_map: dict[str, ReconstructionCertificate],
    invalidated_qubits: Sequence[str],
) -> tuple[str, ...]:
    invalidated = set(invalidated_qubits)
    return tuple(
        certificate_id
        for certificate_id in certificate_ids
        if certificate_id in certificate_map
        and not (set(certificate_map[certificate_id].qubits) & invalidated)
    )


def _ensure_checked_scope_consistency(
    units: Sequence[AbstractUnitState],
) -> None:
    for index, left in enumerate(units):
        for right in units[index + 1 :]:
            if set(left.qubits) & set(right.qubits) and not _overlap_consistent(
                left.witness_rho,
                left.qubits,
                right.witness_rho,
                right.qubits,
            ):
                raise ValueError("Overlapping abstract units are inconsistent on their shared qubits")


def reconstruct_scope_state(
    state: AbstractState,
    global_qubits: list[str],
    scope: Sequence[str],
    mode: ReconstructionMode = "trusted",
) -> DensityMatrix:
    """Reconstruct one scope state from current unit witnesses and certificates."""

    scope_set = set(scope)
    involved_units = [unit for unit in state.units if scope_set & set(unit.qubits)]
    if not involved_units:
        raise ValueError("No abstract units intersect the requested scope")

    if mode == "checked":
        _ensure_checked_scope_consistency(involved_units)

    covering_units = [unit for unit in state.units if scope_set.issubset(set(unit.qubits))]
    if covering_units:
        covering_unit = min(covering_units, key=lambda unit: len(unit.qubits))
        return _reduce_scope(covering_unit.witness_rho, covering_unit.qubits, scope)

    return _reconstruct_workspace_state(
        state=state,
        units=involved_units,
        workspace_qubits=scope,
        global_qubits=global_qubits,
        mode=mode,
    )


def merge_update_rewrite(
    *,
    pre_state: AbstractState,
    gate: GateSpec,
    global_qubits: list[str],
    post_units: list[UnitSpec],
    reconstruction_mode: ReconstructionMode = "trusted",
) -> AbstractState:
    """Execute one merge-update-rewrite step from pre-state views to post-state views."""

    affected_pre_views = select_affected_views(pre_state.units, gate)
    if not affected_pre_views:
        raise ValueError("No affected pre-state views were found for the gate")

    pre_workspace_qubits = _ordered_union(global_qubits, [unit.qubits for unit in affected_pre_views])
    affected_post_specs = tuple(unit for unit in post_units if set(unit.qubits) & set(pre_workspace_qubits))
    if not affected_post_specs:
        raise ValueError("No affected post-state views were found for the gate")
    affected_post_identities = {
        _unit_identity(unit.name, unit.qubits) for unit in affected_post_specs
    }

    workspace_qubits = _ordered_union(
        global_qubits,
        [unit.qubits for unit in affected_pre_views] + [unit.qubits for unit in affected_post_specs],
    )
    reconstruction_units = select_reconstruction_support_units(
        all_units=pre_state.units,
        seed_units=affected_pre_views,
        workspace_qubits=workspace_qubits,
        global_qubits=global_qubits,
    )

    operator = _gate_operator(gate)
    if len(affected_pre_views) == 1 and set(workspace_qubits).issubset(set(affected_pre_views[0].qubits)):
        workspace_before = affected_pre_views[0].witness_rho
        workspace_order = affected_pre_views[0].qubits
    else:
        workspace_before = _reconstruct_workspace_state(
            state=pre_state,
            units=reconstruction_units,
            workspace_qubits=workspace_qubits,
            global_qubits=global_qubits,
            mode=reconstruction_mode,
        )
        workspace_order = workspace_qubits

    qargs = [list(workspace_order).index(qubit) for qubit in gate_support(gate)]
    workspace_after = workspace_before.evolve(operator, qargs=qargs)
    certificate_id = f"cert_s{pre_state.position + 1}_{'_'.join(workspace_qubits)}"
    new_certificate = ReconstructionCertificate(
        certificate_id=certificate_id,
        qubits=workspace_qubits,
        witness_rho=workspace_after,
        metadata={"source_gate": gate.label or gate.name, "position": pre_state.position + 1},
    )
    certificate_map = _certificate_map(pre_state)
    surviving_certificates = tuple(
        certificate
        for certificate in pre_state.certificates
        if not (set(certificate.qubits) & set(workspace_qubits))
    )

    next_units = []
    unaffected_copied_units: list[AbstractUnitState] = []
    for unit in post_units:
        if _unit_identity(unit.name, unit.qubits) in affected_post_identities:
            reduced = _reduce_scope(workspace_after, workspace_order, unit.qubits)
            next_units.append(
                abstract_local_state(
                    reduced,
                    unit.qubits,
                    name=unit.name,
                    certificate_ids=(certificate_id,),
                )
            )
            continue

        copied = _copy_unaffected_unit(pre_state, unit)
        if copied is not None:
            copied = abstract_local_state(
                copied.witness_rho,
                copied.qubits,
                name=copied.name,
                certificate_ids=_filter_valid_certificate_ids(
                    copied.certificate_ids,
                    certificate_map,
                    workspace_qubits,
                ),
            )
            next_units.append(copied)
            unaffected_copied_units.append(copied)
            continue

        raise ValueError(
            "Post-state units outside the updated workspace must be traceable to pre-state units"
        )

    return AbstractState(
        units=tuple(next_units),
        position=pre_state.position + 1,
        certificates=surviving_certificates + (new_certificate,),
        metadata={
            "affected_pre_views": tuple(
                unit.name if unit.name is not None else "|".join(unit.qubits)
                for unit in affected_pre_views
            ),
            "affected_post_views": tuple(
                unit.name if unit.name is not None else "|".join(unit.qubits)
                for unit in affected_post_specs
            ),
            "workspace_qubits": workspace_qubits,
            "transition_peak_bytes": int(workspace_after.data.nbytes)
            + _units_witness_bytes(unaffected_copied_units),
        },
    )


def _initial_state_from_zero(
    units: list[UnitSpec], position: int, global_qubits: list[str]
) -> AbstractState:
    abstract_units = []

    for unit in units:
        witness = DensityMatrix.from_label("0" * len(unit.qubits))
        abstract_units.append(
            abstract_local_state(
                witness,
                unit.qubits,
                name=unit.name,
                certificate_ids=(),
            )
        )
    return AbstractState(
        units=tuple(abstract_units),
        position=position,
        certificates=(),
    )


def _compile_organization_schedule(spec: QuantumProgramSpec) -> list[list[UnitSpec]]:
    if spec.organization_schedule is None:
        raise ValueError("spec.organization_schedule is not defined")

    states_by_name = {
        state.name: state
        for state in spec.organization_schedule.states
    }
    compiled_schedule: list[list[UnitSpec]] = []
    current_name = spec.organization_schedule.initial_state

    for gate_index in range(len(spec.gates)):
        current_state = states_by_name[current_name]
        compiled_schedule.append(current_state.units)
        transition = current_state.transition
        if transition is None or transition.gate_index != gate_index:
            raise ValueError("organization_schedule is not a valid sequential state chain")
        current_name = transition.next_state

    compiled_schedule.append(states_by_name[current_name].units)
    return compiled_schedule


def build_abstract_trace(
    spec: QuantumProgramSpec,
    organization_schedule: list[list[UnitSpec]] | None = None,
    reconstruction_mode: ReconstructionMode = "trusted",
) -> AbstractExecutionTrace:
    """Build the abstract execution trace of a linear quantum program."""

    if organization_schedule is None:
        if spec.organization_schedule is not None:
            organization_schedule = _compile_organization_schedule(spec)
        else:
            if not spec.units:
                raise ValueError("spec.units must be non-empty to build an abstract trace")
            organization_schedule = [spec.units for _ in range(len(spec.gates) + 1)]

    if len(organization_schedule) != len(spec.gates) + 1:
        raise ValueError("organization_schedule must have length len(spec.gates) + 1")

    states = [_initial_state_from_zero(organization_schedule[0], position=0, global_qubits=spec.qubits)]
    transitions: list[AbstractTransition] = []

    for index, gate in enumerate(spec.gates):
        pre_state = states[-1]
        post_state = merge_update_rewrite(
            pre_state=pre_state,
            gate=gate,
            global_qubits=spec.qubits,
            post_units=organization_schedule[index + 1],
            reconstruction_mode=reconstruction_mode,
        )

        states.append(post_state)
        transitions.append(
            AbstractTransition(
                source_id=f"s{index}",
                target_id=f"s{index + 1}",
                label=gate.label or gate.name,
                affected_views=post_state.metadata["affected_pre_views"],
                metadata={
                    "affected_pre_views": post_state.metadata["affected_pre_views"],
                    "affected_post_views": post_state.metadata["affected_post_views"],
                    "workspace_qubits": post_state.metadata["workspace_qubits"],
                    "transition_peak_bytes": post_state.metadata["transition_peak_bytes"],
                },
            )
        )

    return AbstractExecutionTrace(
        states=tuple(states),
        transitions=tuple(transitions),
    )
