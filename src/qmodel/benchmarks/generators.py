"""Benchmark generation entry points."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


GHZ_STANDARD_SIZES = (10, 20, 50, 100, 150, 200)
GHZ_BIASED_VARIANTS = (
    (20, 0.25),
    (50, 0.75),
)
BV_STANDARD_SIZES = (10, 20, 50, 100, 150, 200)
IQFT_FAMILY_VARIANTS = (
    (10, 5),
    (20, 5),
    (50, 5),
    (100, 5),
    (150, 5),
    (200, 5),
)
IQFT_SMALL_COMPARE_VARIANTS = tuple((n, 5) for n in range(10, 21))
ADDER_STANDARD_SIZES = (10, 20, 50, 100, 150, 200)
CUSTOM_MODEL_NAMES = (
    "custom_overlap_chain_prob_6",
    "custom_back_edge_prob_6",
    "custom_split_merge_prob_8",
    "custom_ccx_ladder_reach_9",
    "custom_uncompute_reach_8",
    "custom_disconnected_product_prob_10",
    "custom_overlap_chain_counter_6",
    "custom_split_merge_counter_8",
    "custom_disconnected_product_counter_10",
    "custom_ccx_ladder_counter_9",
    "custom_uncompute_counter_8",
    "custom_overlap_chain_prob_12",
    "custom_back_edge_prob_12",
    "custom_split_merge_prob_14",
    "custom_ccx_ladder_reach_15",
    "custom_disconnected_product_prob_20",
)


class _FlowList(list[str]):
    """Emit selected YAML sequences in flow style."""


class _QModelDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:  # type: ignore[override]
        return True


def _represent_flow_list(
    dumper: yaml.SafeDumper, data: _FlowList
) -> yaml.nodes.SequenceNode:
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq",
        list(data),
        flow_style=True,
    )


_QModelDumper.add_representer(_FlowList, _represent_flow_list)


def _probability_tag(probability: float) -> str:
    return f"p{int(round(probability * 100)):03d}"


def _ry_theta_for_probability(probability: float) -> float:
    if not (0.0 < probability < 1.0):
        raise ValueError("root_probability must lie strictly between 0 and 1")
    return 2.0 * math.asin(math.sqrt(probability))


def _unit_entry(name: str, qubits: list[str]) -> dict[str, Any]:
    return {"name": name, "qubits": qubits}


def _bits_to_int_lsb_first(bits: str) -> int:
    return sum(int(bit) << index for index, bit in enumerate(bits))


def _int_to_bits_lsb_first(value: int, width: int) -> str:
    return "".join("1" if (value >> index) & 1 else "0" for index in range(width))


def _distributed_positions(length: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [length // 2]
    if count >= length:
        return list(range(length))

    positions: list[int] = []
    used: set[int] = set()
    for index in range(count):
        candidate = round(index * (length - 1) / (count - 1))
        candidate = min(length - 1, max(0, candidate))
        if candidate in used:
            for offset in range(1, length):
                right = candidate + offset
                left = candidate - offset
                if right < length and right not in used:
                    candidate = right
                    break
                if left >= 0 and left not in used:
                    candidate = left
                    break
        used.add(candidate)
        positions.append(candidate)
    return sorted(positions)


def _organization_states(n: int) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for step in range(n + 1):
        units: list[dict[str, Any]] = []
        pair_count = max(0, step - 1)
        for pair_start in range(pair_count):
            units.append(
                _unit_entry(f"u{pair_start}{pair_start + 1}", [f"q{pair_start}", f"q{pair_start + 1}"])
            )
        singleton_start = 0 if pair_count == 0 else pair_count + 1
        for qubit_index in range(singleton_start, n):
            units.append(_unit_entry(f"u{qubit_index}", [f"q{qubit_index}"]))

        state: dict[str, Any] = {"name": f"s{step}", "units": units}
        if step < n:
            state["transition"] = {"gate_index": step, "next_state": f"s{step + 1}"}
        states.append(state)
    return states


def _single_qubit_states(qubit_names: list[str], gate_count: int) -> list[dict[str, Any]]:
    units = [_unit_entry(f"u{i}", [qubit]) for i, qubit in enumerate(qubit_names)]
    states: list[dict[str, Any]] = []
    for step in range(gate_count + 1):
        state: dict[str, Any] = {
            "name": f"s{step}",
            "units": [dict(name=unit["name"], qubits=list(unit["qubits"])) for unit in units],
        }
        if step < gate_count:
            state["transition"] = {"gate_index": step, "next_state": f"s{step + 1}"}
        states.append(state)
    return states


def _singleton_units(qubit_names: list[str]) -> list[dict[str, Any]]:
    return [_unit_entry(f"u{i}", [qubit]) for i, qubit in enumerate(qubit_names)]


def _clone_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_unit_entry(unit["name"], list(unit["qubits"])) for unit in units]


def _state_chain_from_snapshots(
    unit_snapshots: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for index, units in enumerate(unit_snapshots):
        state: dict[str, Any] = {"name": f"s{index}", "units": _clone_units(units)}
        if index < len(unit_snapshots) - 1:
            state["transition"] = {"gate_index": index, "next_state": f"s{index + 1}"}
        states.append(state)
    return states


def _window_units(
    qubit_names: list[str],
    active_qubits: list[str],
    *,
    name: str,
) -> list[dict[str, Any]]:
    active_set = set(active_qubits)
    units: list[dict[str, Any]] = []
    if active_qubits:
        units.append(_unit_entry(name, list(active_qubits)))
    for index, qubit in enumerate(qubit_names):
        if qubit not in active_set:
            units.append(_unit_entry(f"u{index}", [qubit]))
    return units


def _ordered_union(
    global_order: list[str],
    scopes: list[list[str]],
) -> list[str]:
    included = {qubit for scope in scopes for qubit in scope}
    return [qubit for qubit in global_order if qubit in included]


def _bv_hidden_string(n: int) -> str:
    input_qubits = n - 1
    weight = max(3, round(input_qubits / 4))
    positions = _distributed_positions(input_qubits, weight)
    bits = ["0"] * input_qubits
    for position in positions:
        bits[position] = "1"
    return "".join(bits)


def _iqft_target_string(n: int) -> str:
    weight = max(3, round(n / 6))
    positions = _distributed_positions(n, weight)
    bits = ["0"] * n
    for position in positions:
        bits[position] = "1"
    return "".join(bits)


def _iqft_threshold(n: int, window_size: int) -> float:
    if window_size == 5:
        return 0.997
    raise ValueError(
        f"No calibrated IQFT threshold for n={n}, window_size={window_size}"
    )


def _bitwise_singleton_states(qubit_names: list[str], gate_count: int) -> list[dict[str, Any]]:
    return _single_qubit_states(qubit_names, gate_count)


def _sliding_window_units(qubit_names: list[str], start_index: int, window_size: int) -> list[dict[str, Any]]:
    end_index = min(len(qubit_names) - 1, start_index + window_size - 1)
    active = set(qubit_names[start_index : end_index + 1])
    units = [_unit_entry(f"w_{start_index}_{end_index}", qubit_names[start_index : end_index + 1])]
    for index, qubit in enumerate(qubit_names):
        if qubit not in active:
            units.append(_unit_entry(f"u{index}", [qubit]))
    return units


def _adder_register_bitstrings(register_bits: int) -> tuple[str, str]:
    a_bits = "".join("1" if index % 2 == 0 else "0" for index in range(register_bits))
    b_bits = "".join("1" if index % 4 in {0, 1} else "0" for index in range(register_bits))
    return a_bits, b_bits


def _adder_output_bitstring(a_bits: str, b_bits: str) -> str:
    total = _bits_to_int_lsb_first(a_bits) + _bits_to_int_lsb_first(b_bits)
    register_bits = len(a_bits)
    b_sum_bits = _int_to_bits_lsb_first(total % (1 << register_bits), register_bits)
    cout_bit = "1" if (total >> register_bits) & 1 else "0"
    return "0" + a_bits + b_sum_bits + cout_bit


def _grover_mark_bits(search_bits: int) -> int:
    if search_bits <= 5:
        return 2
    if search_bits <= 25:
        return 3
    if search_bits <= 50:
        return 4
    if search_bits <= 75:
        return 5
    if search_bits <= 100:
        return 6
    return 7


def _grover_iterations(mark_bits: int) -> int:
    theta = math.asin(1.0 / math.sqrt(2**mark_bits))
    return max(1, round(math.pi / (4.0 * theta) - 0.5))


def _grover_threshold(mark_bits: int) -> float:
    if mark_bits == 2:
        return 0.999999
    if mark_bits == 3:
        return 0.94
    if mark_bits == 4:
        return 0.96
    if mark_bits in {5, 6, 7}:
        return 0.99
    raise ValueError(f"Unsupported Grover mark_bits={mark_bits}")


def _reduction_plan(
    controls: list[str],
    work_qubits: list[str],
) -> tuple[list[tuple[str, str, str]], tuple[str, str]]:
    if len(controls) < 2:
        raise ValueError("Reduction plan requires at least two controls")

    operations: list[tuple[str, str, str]] = []
    current = list(controls)
    work_index = 0
    while len(current) > 2:
        next_level: list[str] = []
        pair_index = 0
        while pair_index + 1 < len(current):
            if work_index >= len(work_qubits):
                raise ValueError("Insufficient work qubits for Grover reduction tree")
            left = current[pair_index]
            right = current[pair_index + 1]
            output = work_qubits[work_index]
            work_index += 1
            operations.append((left, right, output))
            next_level.append(output)
            pair_index += 2
        if pair_index < len(current):
            next_level.append(current[pair_index])
        current = next_level
    return operations, (current[0], current[1])


def _maj_gates(a_qubit: str, b_qubit: str, carry_qubit: str, stage: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "CX",
            "controls": [a_qubit],
            "targets": [b_qubit],
            "label": f"maj-cx-ab-{stage}",
        },
        {
            "name": "CX",
            "controls": [a_qubit],
            "targets": [carry_qubit],
            "label": f"maj-cx-ac-{stage}",
        },
        {
            "name": "CCX",
            "controls": [carry_qubit, b_qubit],
            "targets": [a_qubit],
            "label": f"maj-ccx-cba-{stage}",
        },
    ]


def _uma_gates(a_qubit: str, b_qubit: str, carry_qubit: str, stage: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "CCX",
            "controls": [carry_qubit, b_qubit],
            "targets": [a_qubit],
            "label": f"uma-ccx-cba-{stage}",
        },
        {
            "name": "CX",
            "controls": [a_qubit],
            "targets": [carry_qubit],
            "label": f"uma-cx-ac-{stage}",
        },
        {
            "name": "CX",
            "controls": [carry_qubit],
            "targets": [b_qubit],
            "label": f"uma-cx-cb-{stage}",
        },
    ]


def build_grover_payload(n: int) -> dict[str, Any]:
    if n < 10 or n % 2 != 0:
        raise ValueError("Grover models require an even qubit count of at least 10")

    qubits = [f"q{i}" for i in range(n)]
    search_bits = n // 2
    work_bits = search_bits - 1
    search_qubits = qubits[:search_bits]
    work_qubits = qubits[search_bits:-1]
    phase_qubit = qubits[-1]
    mark_bits = _grover_mark_bits(search_bits)
    marked_controls = search_qubits[:mark_bits]
    iterations = _grover_iterations(mark_bits)
    threshold = _grover_threshold(mark_bits)
    outcome = "1" * mark_bits

    core_work_qubits = list(work_qubits[: max(0, mark_bits - 2)])
    core_qubits = _ordered_union(qubits, [marked_controls, core_work_qubits, [phase_qubit]])

    def reduction_sequence(
        controls: list[str],
        prefix: str,
    ) -> list[dict[str, Any]]:
        if len(controls) == 2:
            return [
                {
                    "name": "CCX",
                    "controls": [controls[0], controls[1]],
                    "targets": [phase_qubit],
                    "label": f"{prefix}-phase-flip",
                }
            ]
        operations, final_controls = _reduction_plan(controls, core_work_qubits)
        sequence: list[dict[str, Any]] = []
        for index, (left, right, output) in enumerate(operations):
            sequence.append(
                {
                    "name": "CCX",
                    "controls": [left, right],
                    "targets": [output],
                    "label": f"{prefix}-compute-{index}",
                }
            )
        sequence.append(
            {
                "name": "CCX",
                "controls": list(final_controls),
                "targets": [phase_qubit],
                "label": f"{prefix}-phase-flip",
            }
        )
        for reverse_index, (left, right, output) in enumerate(reversed(operations)):
            sequence.append(
                {
                    "name": "CCX",
                    "controls": [left, right],
                    "targets": [output],
                    "label": f"{prefix}-uncompute-{reverse_index}",
                }
            )
        return sequence

    gates: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = [{"name": "s0", "units": _singleton_units(qubits)}]
    core_active = False

    def current_units() -> list[dict[str, Any]]:
        if not core_active:
            return _singleton_units(qubits)
        return _window_units(qubits, core_qubits, name="grover_core")

    def append_gate(gate: dict[str, Any]) -> None:
        gate_index = len(gates)
        states[-1]["transition"] = {
            "gate_index": gate_index,
            "next_state": f"s{gate_index + 1}",
        }
        gates.append(gate)
        states.append({"name": f"s{gate_index + 1}", "units": current_units()})

    def append_single_qubit_layer(
        gate_name: str,
        targets: list[str],
        label_prefix: str,
    ) -> None:
        for index, qubit in enumerate(targets):
            append_gate(
                {
                    "name": gate_name,
                    "targets": [qubit],
                    "label": f"{label_prefix}-{index}",
                }
            )

    append_single_qubit_layer("H", search_qubits, "prepare-search")
    append_gate({"name": "X", "targets": [phase_qubit], "label": "prepare-phase-one"})
    append_gate({"name": "H", "targets": [phase_qubit], "label": "prepare-phase-minus"})

    for iteration in range(iterations):
        core_active = True
        for gate in reduction_sequence(marked_controls, f"oracle-{iteration}"):
            append_gate(gate)

        append_single_qubit_layer("H", marked_controls, f"diffuser-{iteration}-h-pre")
        append_single_qubit_layer("X", marked_controls, f"diffuser-{iteration}-x-pre")
        for gate in reduction_sequence(marked_controls, f"diffuser-{iteration}"):
            append_gate(gate)
        append_single_qubit_layer("X", marked_controls, f"diffuser-{iteration}-x-post")
        append_single_qubit_layer("H", marked_controls, f"diffuser-{iteration}-h-post")
        core_active = False

    return {
        "format": "qmodel-v1",
        "program_name": f"grover_{n}",
        "metadata": {
            "family": "grover",
            "n": n,
            "search_bits": search_bits,
            "work_bits": work_bits,
            "mark_bits": mark_bits,
            "core_work_bits": len(core_work_qubits),
            "marked_fraction": 1.0 / (2**mark_bits),
            "iterations": iterations,
            "pattern": "local_core_multi_solution_joint_assertion",
        },
        "qubits": qubits,
        "initial_state": "zero",
        "gates": gates,
        "measurement": {"qubits": marked_controls, "basis": "computational"},
        "organization_schedule": {
            "initial_state": "s0",
            "states": states,
        },
        "assertion": {
            "name": "marked_subspace_probability",
            "kind": "probability",
            "target": {
                "type": "measurement_outcome",
                "scope": marked_controls,
                "outcomes": [outcome],
            },
            "comparator": ">=",
            "threshold": threshold,
        },
    }


def _chain_unit_name(indices: list[int]) -> str:
    return "u" + "".join(str(index) for index in indices)


def _overlap_chain_units(
    qubits: list[str],
    edge_index: int,
    *,
    include_back_edge: bool = False,
    index_labels: list[int] | None = None,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    covered: set[str] = set()
    labels = index_labels if index_labels is not None else list(range(len(qubits)))

    def add(indices: list[int]) -> None:
        unit_qubits = [qubits[index] for index in indices]
        units.append(_unit_entry(_chain_unit_name([labels[index] for index in indices]), unit_qubits))
        covered.update(unit_qubits)

    for start in range(0, max(edge_index - 1, 0), 2):
        add([start, start + 1, start + 2])

    if edge_index == 0:
        add([0, 1])
    elif edge_index % 2 == 1:
        add([edge_index - 1, edge_index, edge_index + 1])
    else:
        add([edge_index, edge_index + 1])

    if include_back_edge and len(qubits) > 2:
        add([1, len(qubits) - 1])

    for index, qubit in enumerate(qubits):
        if qubit not in covered:
            units.append(_unit_entry(f"u{labels[index]}", [qubit]))
    return units


def _custom_probability_assertion(
    scope: list[str],
    outcome: str,
    threshold: float,
    *,
    name: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "probability",
        "target": {
            "type": "measurement_outcome",
            "scope": scope,
            "outcomes": [outcome],
        },
        "comparator": "=",
        "threshold": threshold,
    }


def _custom_reachability_assertion(
    scope: list[str],
    state: str,
    *,
    name: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "reachability",
        "target": {
            "type": "basis_state",
            "scope": scope,
            "state": state,
        },
    }


def _custom_metadata(
    *,
    pattern: str,
    n: int,
    expected_judgment: str,
) -> dict[str, Any]:
    return {
        "family": "custom",
        "pattern": pattern,
        "n": n,
        "expected_judgment": expected_judgment,
    }


def _build_custom_chain_payload(
    n: int,
    *,
    program_name: str,
    pattern: str,
    expected_judgment: str,
    include_back_edge: bool = False,
    broken_tail: bool = False,
) -> dict[str, Any]:
    qubits = [f"q{i}" for i in range(n)]
    last_edge = n - 3 if broken_tail else n - 2
    gates: list[dict[str, Any]] = [
        {
            "name": "Ry",
            "targets": ["q0"],
            "params": {"theta": _ry_theta_for_probability(0.25)},
            "label": "prepare-root-biased",
        }
    ]
    gates.extend(
        {
            "name": "CX",
            "controls": [f"q{index}"],
            "targets": [f"q{index + 1}"],
            "label": f"chain-{index}{index + 1}",
        }
        for index in range(last_edge + 1)
    )
    snapshots = [_singleton_units(qubits), _singleton_units(qubits)]
    snapshots.extend(
        _overlap_chain_units(
            qubits,
            edge_index,
            include_back_edge=include_back_edge and edge_index == last_edge,
        )
        for edge_index in range(last_edge + 1)
    )
    return {
        "format": "qmodel-v1",
        "program_name": program_name,
        "metadata": _custom_metadata(
            pattern=pattern,
            n=n,
            expected_judgment=expected_judgment,
        ),
        "qubits": qubits,
        "initial_state": "zero",
        "gates": gates,
        "measurement": {"qubits": qubits, "basis": "computational"},
        "organization_schedule": {
            "initial_state": "s0",
            "states": _state_chain_from_snapshots(snapshots),
        },
        "assertion": _custom_probability_assertion(
            qubits,
            "1" * n,
            0.25,
            name="chain_basis_probability",
        ),
    }


def _build_custom_small_split_merge_payload(
    *,
    program_name: str,
    pattern: str,
    expected_judgment: str,
    include_merge_gate: bool,
) -> dict[str, Any]:
    qubits = [f"q{i}" for i in range(8)]
    gates = [
        {"name": "X", "targets": ["q0"], "label": "seed-left"},
        {"name": "CX", "controls": ["q0"], "targets": ["q1"], "label": "left-01"},
        {"name": "CX", "controls": ["q1"], "targets": ["q2"], "label": "left-12"},
        {"name": "X", "targets": ["q4"], "label": "seed-right"},
        {"name": "CX", "controls": ["q4"], "targets": ["q5"], "label": "right-45"},
        {"name": "CX", "controls": ["q5"], "targets": ["q6"], "label": "right-56"},
    ]
    snapshots = [
        _singleton_units(qubits),
        _singleton_units(qubits),
        [_unit_entry("u01", ["q0", "q1"]), _unit_entry("u2", ["q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u4", ["q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"])],
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u4", ["q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"])],
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u4", ["q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"])],
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u45", ["q4", "q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"])],
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u456", ["q4", "q5", "q6"]), _unit_entry("u7", ["q7"])],
    ]
    if include_merge_gate:
        gates.append(
            {"name": "CCX", "controls": ["q2", "q6"], "targets": ["q7"], "label": "merge-267"}
        )
        snapshots.append(
            [
                _unit_entry("u012", ["q0", "q1", "q2"]),
                _unit_entry("u3", ["q3"]),
                _unit_entry("u456", ["q4", "q5", "q6"]),
                _unit_entry("u267", ["q2", "q6", "q7"]),
            ]
        )

    return {
        "format": "qmodel-v1",
        "program_name": program_name,
        "metadata": _custom_metadata(
            pattern=pattern,
            n=8,
            expected_judgment=expected_judgment,
        ),
        "qubits": qubits,
        "initial_state": "zero",
        "gates": gates,
        "measurement": {"qubits": qubits, "basis": "computational"},
        "organization_schedule": {
            "initial_state": "s0",
            "states": _state_chain_from_snapshots(snapshots),
        },
        "assertion": _custom_probability_assertion(
            qubits,
            "11101111",
            1.0,
            name="split_merge_basis_probability",
        ),
    }


def _build_custom_small_ccx_ladder_payload(
    *,
    program_name: str,
    pattern: str,
    expected_judgment: str,
    seed_q1: bool,
) -> dict[str, Any]:
    qubits = [f"q{i}" for i in range(9)]
    gates = [{"name": "X", "targets": ["q0"], "label": "seed-0"}]
    if seed_q1:
        gates.append({"name": "X", "targets": ["q1"], "label": "seed-1"})
    gates.extend(
        [
            {"name": "CCX", "controls": ["q0", "q1"], "targets": ["q2"], "label": "ladder-012"},
            {"name": "CCX", "controls": ["q1", "q2"], "targets": ["q3"], "label": "ladder-123"},
            {"name": "CCX", "controls": ["q2", "q3"], "targets": ["q4"], "label": "ladder-234"},
            {"name": "X", "targets": ["q6"], "label": "seed-6"},
            {"name": "CX", "controls": ["q6"], "targets": ["q7"], "label": "branch-67"},
            {"name": "CCX", "controls": ["q4", "q7"], "targets": ["q8"], "label": "join-478"},
        ]
    )
    snapshots = [_singleton_units(qubits)]
    if seed_q1:
        snapshots.extend([_singleton_units(qubits), _singleton_units(qubits)])
    else:
        snapshots.append(_singleton_units(qubits))
    snapshots.extend(
        [
            [_unit_entry("w012", ["q0", "q1", "q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u4", ["q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"])],
            [_unit_entry("u0", ["q0"]), _unit_entry("w123", ["q1", "q2", "q3"]), _unit_entry("u4", ["q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"])],
            [_unit_entry("u0", ["q0"]), _unit_entry("u1", ["q1"]), _unit_entry("w234", ["q2", "q3", "q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"])],
            [_unit_entry("u0", ["q0"]), _unit_entry("u1", ["q1"]), _unit_entry("w234", ["q2", "q3", "q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"])],
            [_unit_entry("u0", ["q0"]), _unit_entry("u1", ["q1"]), _unit_entry("w234", ["q2", "q3", "q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u67", ["q6", "q7"]), _unit_entry("u8", ["q8"])],
            [_unit_entry("u0", ["q0"]), _unit_entry("u1", ["q1"]), _unit_entry("u2", ["q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("w478", ["q4", "q7", "q8"])],
        ]
    )
    return {
        "format": "qmodel-v1",
        "program_name": program_name,
        "metadata": _custom_metadata(
            pattern=pattern,
            n=9,
            expected_judgment=expected_judgment,
        ),
        "qubits": qubits,
        "initial_state": "zero",
        "gates": gates,
        "organization_schedule": {
            "initial_state": "s0",
            "states": _state_chain_from_snapshots(snapshots),
        },
        "assertion": _custom_reachability_assertion(
            ["q8"],
            "1",
            name="ladder_flag_reachable",
        ),
    }


def _build_custom_small_uncompute_payload(
    *,
    program_name: str,
    pattern: str,
    expected_judgment: str,
    prepare_superposition: bool,
) -> dict[str, Any]:
    qubits = [f"q{i}" for i in range(8)]
    gates: list[dict[str, Any]] = []
    snapshots = [_singleton_units(qubits)]
    if prepare_superposition:
        gates.append({"name": "H", "targets": ["q0"], "label": "prepare-superposition"})
        snapshots.append(_singleton_units(qubits))
    gates.append({"name": "CX", "controls": ["q0"], "targets": ["q1"], "label": "entangle-01"})
    snapshots.append(
        [_unit_entry("u01", ["q0", "q1"]), _unit_entry("u2", ["q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u4", ["q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"])]
    )
    gates.append({"name": "CCX", "controls": ["q0", "q1"], "targets": ["q2"], "label": "mark-012"})
    snapshots.append(
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u4", ["q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"])]
    )
    gates.append({"name": "CCX", "controls": ["q0", "q1"], "targets": ["q2"], "label": "unmark-012"})
    snapshots.append(
        [_unit_entry("u01", ["q0", "q1"]), _unit_entry("u2", ["q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u4", ["q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"])]
    )
    gates.append({"name": "CX", "controls": ["q0"], "targets": ["q1"], "label": "unentangle-01"})
    snapshots.append(_singleton_units(qubits))
    if prepare_superposition:
        gates.append({"name": "H", "targets": ["q0"], "label": "return-zero"})
        snapshots.append(_singleton_units(qubits))
    return {
        "format": "qmodel-v1",
        "program_name": program_name,
        "metadata": _custom_metadata(
            pattern=pattern,
            n=8,
            expected_judgment=expected_judgment,
        ),
        "qubits": qubits,
        "initial_state": "zero",
        "gates": gates,
        "organization_schedule": {
            "initial_state": "s0",
            "states": _state_chain_from_snapshots(snapshots),
        },
        "assertion": _custom_reachability_assertion(
            ["q2"],
            "1",
            name="midway_mark_reachable",
        ),
    }


def _build_custom_disconnected_product_payload(
    *,
    n: int,
    program_name: str,
    pattern: str,
    expected_judgment: str,
    left_segment: list[int],
    right_segment: list[int],
    mid_one: list[int],
    tail_one: list[int],
    broken_right: bool = False,
) -> dict[str, Any]:
    qubits = [f"q{i}" for i in range(n)]
    left_qubits = [qubits[index] for index in left_segment]
    right_qubits = [qubits[index] for index in right_segment]
    left_last_edge = len(left_segment) - 2
    right_last_edge = len(right_segment) - 3 if broken_right else len(right_segment) - 2

    gates: list[dict[str, Any]] = [
        {
            "name": "Ry",
            "targets": [left_qubits[0]],
            "params": {"theta": _ry_theta_for_probability(0.25)},
            "label": "prepare-left-root",
        }
    ]
    gates.extend(
        {
            "name": "CX",
            "controls": [left_qubits[index]],
            "targets": [left_qubits[index + 1]],
            "label": f"left-{left_segment[index]}{left_segment[index + 1]}",
        }
        for index in range(left_last_edge + 1)
    )
    gates.extend(
        {
            "name": "X",
            "targets": [qubits[index]],
            "label": f"set-mid-{index}",
        }
        for index in mid_one
    )
    gates.append({"name": "H", "targets": [right_qubits[0]], "label": "prepare-right-root"})
    gates.extend(
        {
            "name": "CX",
            "controls": [right_qubits[index]],
            "targets": [right_qubits[index + 1]],
            "label": f"right-{right_segment[index]}{right_segment[index + 1]}",
        }
        for index in range(right_last_edge + 1)
    )
    gates.extend(
        {
            "name": "X",
            "targets": [qubits[index]],
            "label": f"set-tail-{index}",
        }
        for index in tail_one
    )

    snapshots = [_singleton_units(qubits), _singleton_units(qubits)]
    left_final_units = _overlap_chain_units(left_qubits, left_last_edge, index_labels=left_segment)
    snapshots.extend(
        _clone_units(
            _merge_component_units(
                qubits,
                _overlap_chain_units(left_qubits, edge_index, index_labels=left_segment),
                [],
            )
        )
        for edge_index in range(left_last_edge + 1)
    )
    for _ in mid_one:
        snapshots.append(_clone_units(_merge_component_units(qubits, left_final_units, [])))

    right_singletons = _merge_component_units(qubits, left_final_units, [])
    snapshots.append(_clone_units(right_singletons))
    right_final_units = _overlap_chain_units(right_qubits, right_last_edge, index_labels=right_segment)
    snapshots.extend(
        _clone_units(
            _merge_component_units(
                qubits,
                left_final_units,
                _overlap_chain_units(right_qubits, edge_index, index_labels=right_segment),
            )
        )
        for edge_index in range(right_last_edge + 1)
    )
    final_units = _merge_component_units(qubits, left_final_units, right_final_units)
    for _ in tail_one:
        snapshots.append(_clone_units(final_units))

    bits = ["0"] * n
    for index in left_segment + right_segment + mid_one + tail_one:
        bits[index] = "1"
    outcome = "".join(bits)
    return {
        "format": "qmodel-v1",
        "program_name": program_name,
        "metadata": _custom_metadata(
            pattern=pattern,
            n=n,
            expected_judgment=expected_judgment,
        ),
        "qubits": qubits,
        "initial_state": "zero",
        "gates": gates,
        "measurement": {"qubits": qubits, "basis": "computational"},
        "organization_schedule": {
            "initial_state": "s0",
            "states": _state_chain_from_snapshots(snapshots),
        },
        "assertion": _custom_probability_assertion(
            qubits,
            outcome,
            0.125,
            name="disconnected_product_probability",
        ),
    }


def _merge_component_units(
    qubits: list[str],
    left_units: list[dict[str, Any]],
    right_units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    left_set = {qubit for unit in left_units for qubit in unit["qubits"]}
    right_set = {qubit for unit in right_units for qubit in unit["qubits"]}
    covered = left_set | right_set
    merged = _clone_units(left_units) + _clone_units(right_units)
    for index, qubit in enumerate(qubits):
        if qubit not in covered:
            merged.append(_unit_entry(f"u{index}", [qubit]))
    return merged


def build_custom_overlap_chain_prob_payload() -> dict[str, Any]:
    return _build_custom_chain_payload(
        6,
        program_name="custom_overlap_chain_prob_6",
        pattern="overlap_chain_probability",
        expected_judgment="satisfied",
    )


def build_custom_back_edge_prob_payload() -> dict[str, Any]:
    return _build_custom_chain_payload(
        6,
        program_name="custom_back_edge_prob_6",
        pattern="back_edge_probability",
        expected_judgment="satisfied",
        include_back_edge=True,
    )


def build_custom_split_merge_prob_payload() -> dict[str, Any]:
    return _build_custom_small_split_merge_payload(
        program_name="custom_split_merge_prob_8",
        pattern="split_merge_probability",
        expected_judgment="satisfied",
        include_merge_gate=True,
    )


def build_custom_ccx_ladder_reach_payload() -> dict[str, Any]:
    return _build_custom_small_ccx_ladder_payload(
        program_name="custom_ccx_ladder_reach_9",
        pattern="ccx_ladder_reachability",
        expected_judgment="satisfied",
        seed_q1=True,
    )


def build_custom_uncompute_reach_payload() -> dict[str, Any]:
    return _build_custom_small_uncompute_payload(
        program_name="custom_uncompute_reach_8",
        pattern="uncompute_reachability",
        expected_judgment="satisfied",
        prepare_superposition=True,
    )


def build_custom_disconnected_product_prob_payload() -> dict[str, Any]:
    return _build_custom_disconnected_product_payload(
        n=10,
        program_name="custom_disconnected_product_prob_10",
        pattern="disconnected_product_probability",
        expected_judgment="satisfied",
        left_segment=[0, 1, 2],
        right_segment=[5, 6, 7],
        mid_one=[3],
        tail_one=[8],
    )


def build_custom_overlap_chain_counter_payload() -> dict[str, Any]:
    return _build_custom_chain_payload(
        6,
        program_name="custom_overlap_chain_counter_6",
        pattern="overlap_chain_counterexample",
        expected_judgment="violated",
        broken_tail=True,
    )


def build_custom_split_merge_counter_payload() -> dict[str, Any]:
    return _build_custom_small_split_merge_payload(
        program_name="custom_split_merge_counter_8",
        pattern="split_merge_counterexample",
        expected_judgment="violated",
        include_merge_gate=False,
    )


def build_custom_disconnected_product_counter_payload() -> dict[str, Any]:
    return _build_custom_disconnected_product_payload(
        n=10,
        program_name="custom_disconnected_product_counter_10",
        pattern="disconnected_product_counterexample",
        expected_judgment="violated",
        left_segment=[0, 1, 2],
        right_segment=[5, 6, 7],
        mid_one=[3],
        tail_one=[8],
        broken_right=True,
    )


def build_custom_ccx_ladder_counter_payload() -> dict[str, Any]:
    return _build_custom_small_ccx_ladder_payload(
        program_name="custom_ccx_ladder_counter_9",
        pattern="ccx_ladder_counterexample",
        expected_judgment="violated",
        seed_q1=False,
    )


def build_custom_uncompute_counter_payload() -> dict[str, Any]:
    return _build_custom_small_uncompute_payload(
        program_name="custom_uncompute_counter_8",
        pattern="uncompute_counterexample",
        expected_judgment="violated",
        prepare_superposition=False,
    )


def build_custom_overlap_chain_prob_12_payload() -> dict[str, Any]:
    return _build_custom_chain_payload(
        12,
        program_name="custom_overlap_chain_prob_12",
        pattern="overlap_chain_probability",
        expected_judgment="satisfied",
    )


def build_custom_back_edge_prob_12_payload() -> dict[str, Any]:
    return _build_custom_chain_payload(
        12,
        program_name="custom_back_edge_prob_12",
        pattern="back_edge_probability",
        expected_judgment="satisfied",
        include_back_edge=True,
    )


def build_custom_split_merge_prob_14_payload() -> dict[str, Any]:
    qubits = [f"q{i}" for i in range(14)]
    gates = [
        {"name": "X", "targets": ["q0"], "label": "seed-left"},
        {"name": "CX", "controls": ["q0"], "targets": ["q1"], "label": "left-01"},
        {"name": "CX", "controls": ["q1"], "targets": ["q2"], "label": "left-12"},
        {"name": "CX", "controls": ["q2"], "targets": ["q3"], "label": "left-23"},
        {"name": "CX", "controls": ["q3"], "targets": ["q4"], "label": "left-34"},
        {"name": "X", "targets": ["q7"], "label": "seed-right"},
        {"name": "CX", "controls": ["q7"], "targets": ["q8"], "label": "right-78"},
        {"name": "CX", "controls": ["q8"], "targets": ["q9"], "label": "right-89"},
        {"name": "CX", "controls": ["q9"], "targets": ["q10"], "label": "right-910"},
        {"name": "CX", "controls": ["q10"], "targets": ["q11"], "label": "right-1011"},
        {"name": "CCX", "controls": ["q4", "q11"], "targets": ["q12"], "label": "merge-41112"},
        {"name": "CX", "controls": ["q12"], "targets": ["q13"], "label": "copy-1213"},
    ]
    snapshots = [
        _singleton_units(qubits),
        _singleton_units(qubits),
        [_unit_entry("u01", ["q0", "q1"]), _unit_entry("u2", ["q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u4", ["q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u10", ["q10"]), _unit_entry("u11", ["q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"])],
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u4", ["q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u10", ["q10"]), _unit_entry("u11", ["q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"])],
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u23", ["q2", "q3"]), _unit_entry("u4", ["q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u10", ["q10"]), _unit_entry("u11", ["q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"])],
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u234", ["q2", "q3", "q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u10", ["q10"]), _unit_entry("u11", ["q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"])],
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u234", ["q2", "q3", "q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u10", ["q10"]), _unit_entry("u11", ["q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"])],
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u234", ["q2", "q3", "q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u78", ["q7", "q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u10", ["q10"]), _unit_entry("u11", ["q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"])],
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u234", ["q2", "q3", "q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u789", ["q7", "q8", "q9"]), _unit_entry("u10", ["q10"]), _unit_entry("u11", ["q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"])],
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u234", ["q2", "q3", "q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u789", ["q7", "q8", "q9"]), _unit_entry("u910", ["q9", "q10"]), _unit_entry("u11", ["q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"])],
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u234", ["q2", "q3", "q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u789", ["q7", "q8", "q9"]), _unit_entry("u91011", ["q9", "q10", "q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"])],
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u234", ["q2", "q3", "q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u789", ["q7", "q8", "q9"]), _unit_entry("u91011", ["q9", "q10", "q11"]), _unit_entry("u41112", ["q4", "q11", "q12"]), _unit_entry("u13", ["q13"])],
        [_unit_entry("u012", ["q0", "q1", "q2"]), _unit_entry("u234", ["q2", "q3", "q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u789", ["q7", "q8", "q9"]), _unit_entry("u91011", ["q9", "q10", "q11"]), _unit_entry("u1213", ["q12", "q13"])],
    ]
    return {
        "format": "qmodel-v1",
        "program_name": "custom_split_merge_prob_14",
        "metadata": _custom_metadata(
            pattern="split_merge_probability",
            n=14,
            expected_judgment="satisfied",
        ),
        "qubits": qubits,
        "initial_state": "zero",
        "gates": gates,
        "measurement": {"qubits": qubits, "basis": "computational"},
        "organization_schedule": {
            "initial_state": "s0",
            "states": _state_chain_from_snapshots(snapshots),
        },
        "assertion": _custom_probability_assertion(
            qubits,
            "11111001111111",
            1.0,
            name="split_merge_basis_probability",
        ),
    }


def build_custom_ccx_ladder_reach_15_payload() -> dict[str, Any]:
    qubits = [f"q{i}" for i in range(15)]
    gates = [
        {"name": "X", "targets": ["q0"], "label": "seed-0"},
        {"name": "X", "targets": ["q1"], "label": "seed-1"},
        {"name": "CCX", "controls": ["q0", "q1"], "targets": ["q2"], "label": "ladder-012"},
        {"name": "CCX", "controls": ["q1", "q2"], "targets": ["q3"], "label": "ladder-123"},
        {"name": "CCX", "controls": ["q2", "q3"], "targets": ["q4"], "label": "ladder-234"},
        {"name": "CCX", "controls": ["q3", "q4"], "targets": ["q5"], "label": "ladder-345"},
        {"name": "X", "targets": ["q10"], "label": "seed-10"},
        {"name": "CX", "controls": ["q10"], "targets": ["q11"], "label": "branch-1011"},
        {"name": "CX", "controls": ["q11"], "targets": ["q12"], "label": "branch-1112"},
        {"name": "CCX", "controls": ["q5", "q12"], "targets": ["q13"], "label": "join-51213"},
        {"name": "CCX", "controls": ["q4", "q13"], "targets": ["q14"], "label": "join-41314"},
    ]
    snapshots = [
        _singleton_units(qubits),
        _singleton_units(qubits),
        _singleton_units(qubits),
        [_unit_entry("w012", ["q0", "q1", "q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u4", ["q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u10", ["q10"]), _unit_entry("u11", ["q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"]), _unit_entry("u14", ["q14"])],
        [_unit_entry("u0", ["q0"]), _unit_entry("w123", ["q1", "q2", "q3"]), _unit_entry("u4", ["q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u10", ["q10"]), _unit_entry("u11", ["q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"]), _unit_entry("u14", ["q14"])],
        [_unit_entry("u0", ["q0"]), _unit_entry("u1", ["q1"]), _unit_entry("w234", ["q2", "q3", "q4"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u10", ["q10"]), _unit_entry("u11", ["q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"]), _unit_entry("u14", ["q14"])],
        [_unit_entry("u0", ["q0"]), _unit_entry("u1", ["q1"]), _unit_entry("u2", ["q2"]), _unit_entry("w345", ["q3", "q4", "q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u10", ["q10"]), _unit_entry("u11", ["q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"]), _unit_entry("u14", ["q14"])],
        [_unit_entry("u0", ["q0"]), _unit_entry("u1", ["q1"]), _unit_entry("u2", ["q2"]), _unit_entry("w345", ["q3", "q4", "q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u10", ["q10"]), _unit_entry("u11", ["q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"]), _unit_entry("u14", ["q14"])],
        [_unit_entry("u0", ["q0"]), _unit_entry("u1", ["q1"]), _unit_entry("u2", ["q2"]), _unit_entry("w345", ["q3", "q4", "q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u1011", ["q10", "q11"]), _unit_entry("u12", ["q12"]), _unit_entry("u13", ["q13"]), _unit_entry("u14", ["q14"])],
        [_unit_entry("u0", ["q0"]), _unit_entry("u1", ["q1"]), _unit_entry("u2", ["q2"]), _unit_entry("w345", ["q3", "q4", "q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u101112", ["q10", "q11", "q12"]), _unit_entry("u13", ["q13"]), _unit_entry("u14", ["q14"])],
        [_unit_entry("u0", ["q0"]), _unit_entry("u1", ["q1"]), _unit_entry("u2", ["q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u4", ["q4"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u101112", ["q10", "q11", "q12"]), _unit_entry("w51213", ["q5", "q12", "q13"]), _unit_entry("u14", ["q14"])],
        [_unit_entry("u0", ["q0"]), _unit_entry("u1", ["q1"]), _unit_entry("u2", ["q2"]), _unit_entry("u3", ["q3"]), _unit_entry("u5", ["q5"]), _unit_entry("u6", ["q6"]), _unit_entry("u7", ["q7"]), _unit_entry("u8", ["q8"]), _unit_entry("u9", ["q9"]), _unit_entry("u101112", ["q10", "q11", "q12"]), _unit_entry("w41314", ["q4", "q13", "q14"])],
    ]
    return {
        "format": "qmodel-v1",
        "program_name": "custom_ccx_ladder_reach_15",
        "metadata": _custom_metadata(
            pattern="ccx_ladder_reachability",
            n=15,
            expected_judgment="satisfied",
        ),
        "qubits": qubits,
        "initial_state": "zero",
        "gates": gates,
        "organization_schedule": {
            "initial_state": "s0",
            "states": _state_chain_from_snapshots(snapshots),
        },
        "assertion": _custom_reachability_assertion(
            ["q14"],
            "1",
            name="ladder_flag_reachable",
        ),
    }


def build_custom_disconnected_product_prob_20_payload() -> dict[str, Any]:
    return _build_custom_disconnected_product_payload(
        n=20,
        program_name="custom_disconnected_product_prob_20",
        pattern="disconnected_product_probability",
        expected_judgment="satisfied",
        left_segment=[0, 1, 2, 3, 4, 5],
        right_segment=[10, 11, 12, 13, 14, 15],
        mid_one=[6, 7],
        tail_one=[16, 17],
    )


def _build_iqft_payload(
    n: int,
    window_size: int,
    *,
    program_name: str,
    metadata_extra: dict[str, Any] | None = None,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    if n < 2:
        raise ValueError("IQFT models require at least 2 qubits")
    if window_size < 2:
        raise ValueError("IQFT window_size must be at least 2")

    qubits = [f"q{i}" for i in range(n)]
    target = _iqft_target_string(n)
    radius = window_size - 1
    threshold = (
        threshold_override if threshold_override is not None else _iqft_threshold(n, window_size)
    )

    gates: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    step = 0
    states.append(
        {
            "name": f"s{step}",
            "units": _bitwise_singleton_states(qubits, 0)[0]["units"],
        }
    )

    def append_transition(next_units: list[dict[str, Any]]) -> None:
        nonlocal step
        states[-1]["transition"] = {"gate_index": step, "next_state": f"s{step + 1}"}
        step += 1
        states.append({"name": f"s{step}", "units": next_units})

    for index, qubit in enumerate(qubits):
        gates.append({"name": "H", "targets": [qubit], "label": f"prep-h-{index}"})
        append_transition([_unit_entry(f"u{i}", [name]) for i, name in enumerate(qubits)])

        theta = 2.0 * math.pi * sum(
            int(bit) / (2 ** offset)
            for offset, bit in enumerate(target[index:], start=1)
        )
        if abs(theta) > 1e-12:
            gates.append(
                {
                    "name": "P",
                    "targets": [qubit],
                    "params": {"theta": theta},
                    "label": f"prep-p-{index}",
                }
            )
            append_transition([_unit_entry(f"u{i}", [name]) for i, name in enumerate(qubits)])

    for k in range(n - 1, -1, -1):
        decode_units = _sliding_window_units(qubits, k, window_size)
        upper = min(n - 1, k + radius)
        for j in range(upper, k, -1):
            gates.append(
                {
                    "name": "CP",
                    "controls": [qubits[j]],
                    "targets": [qubits[k]],
                    "params": {"theta": -math.pi / (2 ** (j - k))},
                    "label": f"iqft-cp-{j}-{k}",
                }
            )
            append_transition(decode_units)
        gates.append({"name": "H", "targets": [qubits[k]], "label": f"iqft-h-{k}"})
        append_transition(decode_units)

    metadata = {
        "family": "iqft",
        "n": n,
        "window_size": window_size,
        "phase_radius": radius,
        "target_string": target,
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    return {
        "format": "qmodel-v1",
        "program_name": program_name,
        "metadata": metadata,
        "qubits": qubits,
        "initial_state": "zero",
        "gates": gates,
        "measurement": {"qubits": qubits, "basis": "computational"},
        "organization_schedule": {
            "initial_state": "s0",
            "states": states,
        },
        "assertion": {
            "name": "recover_target_bits",
            "kind": "probability",
            "target": {
                "type": "bitwise_measurement_outcome",
                "scope": qubits,
                "outcome": target,
            },
            "comparator": ">=",
            "threshold": threshold,
        },
    }


def build_iqft_payload(n: int, window_size: int) -> dict[str, Any]:
    return _build_iqft_payload(
        n,
        window_size,
        program_name=f"iqft_{n}_w{window_size}",
    )


def build_iqft_compare_payload(n: int, window_size: int = 5) -> dict[str, Any]:
    return _build_iqft_payload(
        n,
        window_size,
        program_name=f"iqft_compare_{n}_w{window_size}",
        metadata_extra={
            "experiment_group": "small_full_execution_compare",
            "compare_range": "10_to_20",
        },
        threshold_override=0.996,
    )


def build_adder_payload(n: int) -> dict[str, Any]:
    if n < 6 or n % 2 != 0:
        raise ValueError("Adder models require an even qubit count of at least 6")

    register_bits = (n - 2) // 2
    qubits = [f"q{i}" for i in range(n)]
    cin = qubits[0]
    a_qubits = qubits[1 : 1 + register_bits]
    b_qubits = qubits[1 + register_bits : 1 + 2 * register_bits]
    cout = qubits[-1]
    a_bits, b_bits = _adder_register_bitstrings(register_bits)
    expected_output = _adder_output_bitstring(a_bits, b_bits)

    gates: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = [{"name": "s0", "units": _singleton_units(qubits)}]

    def append_gate(gate: dict[str, Any], next_units: list[dict[str, Any]]) -> None:
        gate_index = len(gates)
        states[-1]["transition"] = {"gate_index": gate_index, "next_state": f"s{gate_index + 1}"}
        gates.append(gate)
        states.append({"name": f"s{gate_index + 1}", "units": next_units})

    for index, bit in enumerate(a_bits):
        if bit == "1":
            append_gate(
                {
                    "name": "X",
                    "targets": [a_qubits[index]],
                    "label": f"init-a-{index}",
                },
                _singleton_units(qubits),
            )
    for index, bit in enumerate(b_bits):
        if bit == "1":
            append_gate(
                {
                    "name": "X",
                    "targets": [b_qubits[index]],
                    "label": f"init-b-{index}",
                },
                _singleton_units(qubits),
            )

    for index in range(register_bits):
        carry_qubit = cin if index == 0 else a_qubits[index - 1]
        pair_units = _window_units(
            qubits,
            [a_qubits[index], b_qubits[index]],
            name=f"pair_{index}",
        )
        triple_units = _window_units(
            qubits,
            [carry_qubit, a_qubits[index], b_qubits[index]],
            name=f"carry_{index}",
        )
        maj_gates = _maj_gates(a_qubits[index], b_qubits[index], carry_qubit, index)
        append_gate(maj_gates[0], pair_units)
        append_gate(maj_gates[1], triple_units)
        append_gate(maj_gates[2], _singleton_units(qubits))

    append_gate(
        {
            "name": "CX",
            "controls": [a_qubits[-1]],
            "targets": [cout],
            "label": "write-cout",
        },
        _singleton_units(qubits),
    )

    for index in range(register_bits - 1, -1, -1):
        carry_qubit = cin if index == 0 else a_qubits[index - 1]
        triple_units = _window_units(
            qubits,
            [carry_qubit, a_qubits[index], b_qubits[index]],
            name=f"carry_{index}",
        )
        uma_gates = _uma_gates(a_qubits[index], b_qubits[index], carry_qubit, index)
        append_gate(uma_gates[0], triple_units)
        append_gate(uma_gates[1], triple_units)
        append_gate(uma_gates[2], _singleton_units(qubits))

    return {
        "format": "qmodel-v1",
        "program_name": f"adder_{n}",
        "metadata": {
            "family": "adder",
            "n": n,
            "register_bits": register_bits,
            "pattern": "cuccaro_carry_window_schedule",
            "a_input": a_bits,
            "b_input": b_bits,
            "expected_output": expected_output,
        },
        "qubits": qubits,
        "initial_state": "zero",
        "gates": gates,
        "measurement": {"qubits": qubits, "basis": "computational"},
        "organization_schedule": {
            "initial_state": "s0",
            "states": states,
        },
        "assertion": {
            "name": "adder_output_probability",
            "kind": "probability",
            "target": {
                "type": "bitwise_measurement_outcome",
                "scope": qubits,
                "outcome": expected_output,
            },
            "comparator": ">=",
            "threshold": 0.999999,
        },
    }


def build_iqft_family_payloads() -> list[dict[str, Any]]:
    return [build_iqft_payload(n, window_size) for n, window_size in IQFT_FAMILY_VARIANTS]


def build_iqft_small_compare_payloads() -> list[dict[str, Any]]:
    return [
        build_iqft_compare_payload(n, window_size)
        for n, window_size in IQFT_SMALL_COMPARE_VARIANTS
    ]


def build_adder_family_payloads() -> list[dict[str, Any]]:
    return [build_adder_payload(n) for n in ADDER_STANDARD_SIZES]


def build_custom_family_payloads() -> list[dict[str, Any]]:
    return [
        build_custom_overlap_chain_prob_payload(),
        build_custom_back_edge_prob_payload(),
        build_custom_split_merge_prob_payload(),
        build_custom_ccx_ladder_reach_payload(),
        build_custom_uncompute_reach_payload(),
        build_custom_disconnected_product_prob_payload(),
        build_custom_overlap_chain_counter_payload(),
        build_custom_split_merge_counter_payload(),
        build_custom_disconnected_product_counter_payload(),
        build_custom_ccx_ladder_counter_payload(),
        build_custom_uncompute_counter_payload(),
        build_custom_overlap_chain_prob_12_payload(),
        build_custom_back_edge_prob_12_payload(),
        build_custom_split_merge_prob_14_payload(),
        build_custom_ccx_ladder_reach_15_payload(),
        build_custom_disconnected_product_prob_20_payload(),
    ]


def build_bv_payload(n: int) -> dict[str, Any]:
    if n < 3:
        raise ValueError("BV models require at least 3 qubits")

    qubits = [f"q{i}" for i in range(n)]
    ancilla = f"q{n - 1}"
    input_qubits = qubits[:-1]
    measurement_qubits = list(input_qubits)
    assertion_scope = list(input_qubits)
    hidden_string = _bv_hidden_string(n)
    hidden_positions = [index for index, bit in enumerate(hidden_string) if bit == "1"]

    gates: list[dict[str, Any]] = [
        {"name": "X", "targets": [ancilla], "label": "prepare-ancilla-one"},
        {"name": "H", "targets": [ancilla], "label": "prepare-ancilla-plus"},
    ]
    gates.extend(
        {
            "name": "H",
            "targets": [qubit],
            "label": f"prepare-input-{qubit}",
        }
        for qubit in input_qubits
    )
    gates.extend(
        {
            "name": "CX",
            "controls": [f"q{position}"],
            "targets": [ancilla],
            "label": f"oracle-{position}",
        }
        for position in hidden_positions
    )
    gates.extend(
        {
            "name": "H",
            "targets": [qubit],
            "label": f"decode-input-{qubit}",
        }
        for qubit in input_qubits
    )

    gate_count = len(gates)
    return {
        "format": "qmodel-v1",
        "program_name": f"bv_{n}_sparse",
        "metadata": {
            "family": "bv",
            "n": n,
            "pattern": "sparse_oracle",
            "hidden_string": hidden_string,
            "hidden_weight": len(hidden_positions),
        },
        "qubits": qubits,
        "initial_state": "zero",
        "gates": gates,
        "measurement": {"qubits": measurement_qubits, "basis": "computational"},
        "organization_schedule": {
            "initial_state": "s0",
            "states": _single_qubit_states(qubits, gate_count),
        },
        "assertion": {
            "name": "bv_hidden_string_probability",
            "kind": "probability",
            "target": {
                "type": "bitwise_measurement_outcome",
                "scope": assertion_scope,
                "outcome": hidden_string,
            },
            "comparator": ">=",
            "threshold": 0.999999,
        },
    }


def build_bv_family_payloads() -> list[dict[str, Any]]:
    return [build_bv_payload(n) for n in BV_STANDARD_SIZES]


def build_ghz_staircase_payload(
    n: int,
    *,
    root_probability: float | None = None,
) -> dict[str, Any]:
    if n < 2:
        raise ValueError("GHZ staircase models require at least 2 qubits")

    qubits = [f"q{i}" for i in range(n)]
    if root_probability is None:
        program_name = f"ghz_{n}_staircase"
        metadata: dict[str, Any] = {"family": "ghz", "n": n, "pattern": "staircase_cx"}
        first_gate = {
            "name": "H",
            "targets": ["q0"],
            "label": "prepare-root",
        }
        threshold = 0.5
    else:
        program_name = f"ghz_{n}_root_{_probability_tag(root_probability)}"
        metadata = {
            "family": "ghz",
            "n": n,
            "pattern": "staircase_cx",
            "variant": "biased_root",
            "root_probability": root_probability,
        }
        first_gate = {
            "name": "Ry",
            "targets": ["q0"],
            "params": {"theta": _ry_theta_for_probability(root_probability)},
            "label": "prepare-root-biased",
        }
        threshold = root_probability

    gates: list[dict[str, Any]] = [first_gate]
    for index in range(n - 1):
        gates.append(
            {
                "name": "CX",
                "controls": [f"q{index}"],
                "targets": [f"q{index + 1}"],
                "label": f"entangle-q{index}-q{index + 1}",
            }
        )

    return {
        "format": "qmodel-v1",
        "program_name": program_name,
        "metadata": metadata,
        "qubits": qubits,
        "initial_state": "zero",
        "gates": gates,
        "measurement": {"qubits": [f"q{n - 1}"], "basis": "computational"},
        "organization_schedule": {
            "initial_state": "s0",
            "states": _organization_states(n),
        },
        "assertion": {
            "name": "tail_one_probability",
            "kind": "probability",
            "target": {
                "type": "measurement_outcome",
                "scope": [f"q{n - 1}"],
                "outcomes": ["1"],
            },
            "comparator": "=",
            "threshold": threshold,
        },
    }


def build_ghz_family_payloads() -> list[dict[str, Any]]:
    payloads = [build_ghz_staircase_payload(n) for n in GHZ_STANDARD_SIZES]
    payloads.extend(
        build_ghz_staircase_payload(n, root_probability=probability)
        for n, probability in GHZ_BIASED_VARIANTS
    )
    return payloads


def write_qmodel_payload(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_payload = dict(payload)
    serializable_payload["qubits"] = _FlowList(payload["qubits"])
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.dump(
            serializable_payload,
            handle,
            Dumper=_QModelDumper,
            sort_keys=False,
            allow_unicode=False,
            default_flow_style=False,
        )
    return output_path


def emit_ghz_family_models(output_dir: str | Path) -> list[Path]:
    target_dir = Path(output_dir)
    written_paths: list[Path] = []
    for payload in build_ghz_family_payloads():
        written_paths.append(
            write_qmodel_payload(payload, target_dir / f"{payload['program_name']}.qmodel")
        )
    return written_paths


def emit_bv_family_models(output_dir: str | Path) -> list[Path]:
    target_dir = Path(output_dir)
    written_paths: list[Path] = []
    for payload in build_bv_family_payloads():
        written_paths.append(
            write_qmodel_payload(payload, target_dir / f"{payload['program_name']}.qmodel")
        )
    return written_paths


def emit_adder_family_models(output_dir: str | Path) -> list[Path]:
    target_dir = Path(output_dir)
    written_paths: list[Path] = []
    for payload in build_adder_family_payloads():
        written_paths.append(
            write_qmodel_payload(payload, target_dir / f"{payload['program_name']}.qmodel")
        )
    return written_paths


def emit_iqft_family_models(output_dir: str | Path) -> list[Path]:
    target_dir = Path(output_dir)
    written_paths: list[Path] = []
    for payload in build_iqft_family_payloads():
        written_paths.append(
            write_qmodel_payload(payload, target_dir / f"{payload['program_name']}.qmodel")
        )
    return written_paths


def emit_iqft_small_compare_models(output_dir: str | Path) -> list[Path]:
    target_dir = Path(output_dir)
    written_paths: list[Path] = []
    for payload in build_iqft_small_compare_payloads():
        written_paths.append(
            write_qmodel_payload(payload, target_dir / f"{payload['program_name']}.qmodel")
        )
    return written_paths


def emit_custom_family_models(output_dir: str | Path) -> list[Path]:
    target_dir = Path(output_dir)
    written_paths: list[Path] = []
    for payload in build_custom_family_payloads():
        written_paths.append(
            write_qmodel_payload(payload, target_dir / f"{payload['program_name']}.qmodel")
        )
    return written_paths
