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
AIQFT_FAMILY_VARIANTS = (
    (10, 5),
    (20, 5),
    (50, 5),
    (100, 5),
    (150, 5),
    (200, 5),
)
ADDER_STANDARD_SIZES = (10, 20, 50, 100, 150, 200)


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


def _bv_hidden_string(n: int) -> str:
    input_qubits = n - 1
    weight = max(3, round(input_qubits / 4))
    positions = _distributed_positions(input_qubits, weight)
    bits = ["0"] * input_qubits
    for position in positions:
        bits[position] = "1"
    return "".join(bits)


def _aiqft_target_string(n: int) -> str:
    weight = max(3, round(n / 6))
    positions = _distributed_positions(n, weight)
    bits = ["0"] * n
    for position in positions:
        bits[position] = "1"
    return "".join(bits)


def _aiqft_threshold(n: int, window_size: int) -> float:
    if window_size == 5:
        return 0.997
    raise ValueError(
        f"No calibrated AIQFT threshold for n={n}, window_size={window_size}"
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


def build_aiqft_payload(n: int, window_size: int) -> dict[str, Any]:
    if n < 2:
        raise ValueError("AIQFT models require at least 2 qubits")
    if window_size < 2:
        raise ValueError("AIQFT window_size must be at least 2")

    qubits = [f"q{i}" for i in range(n)]
    target = _aiqft_target_string(n)
    radius = window_size - 1
    threshold = _aiqft_threshold(n, window_size)

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

    return {
        "format": "qmodel-v1",
        "program_name": f"aiqft_{n}_w{window_size}",
        "metadata": {
            "family": "aiqft",
            "n": n,
            "window_size": window_size,
            "phase_radius": radius,
            "target_string": target,
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


def build_aiqft_family_payloads() -> list[dict[str, Any]]:
    return [build_aiqft_payload(n, window_size) for n, window_size in AIQFT_FAMILY_VARIANTS]


def build_adder_family_payloads() -> list[dict[str, Any]]:
    return [build_adder_payload(n) for n in ADDER_STANDARD_SIZES]


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


def emit_aiqft_family_models(output_dir: str | Path) -> list[Path]:
    target_dir = Path(output_dir)
    written_paths: list[Path] = []
    for payload in build_aiqft_family_payloads():
        written_paths.append(
            write_qmodel_payload(payload, target_dir / f"{payload['program_name']}.qmodel")
        )
    return written_paths
