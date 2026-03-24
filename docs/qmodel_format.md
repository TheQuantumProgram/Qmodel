# `.qmodel` Format Notes

This document records the current Version 1 rules for handwritten model files used in experiments.

The small illustrative `.qmodel` examples currently live under `project_code/tests/models/`. The `project_code/experiment_data/` tree is reserved for formal experiment inputs and generated results.

Formal experiment models are organized by algorithm family under:

- `project_code/experiment_data/models/GHZ/`
- `project_code/experiment_data/models/BV/`
- `project_code/experiment_data/models/Grover/`
- `project_code/experiment_data/models/AIQFT/`
- `project_code/experiment_data/models/Adder/`
- `project_code/experiment_data/models/Custom/`

## Purpose

One `.qmodel` file describes one executable experiment instance:

- one quantum program,
- one human-specified unit organization, either static or state-by-state,
- one terminal measurement setting,
- one terminal assertion.

The file is declarative. It does not contain Python code or backend instructions.

## Top-Level Fields

Version 1 uses the following top-level fields:

- `format`
- `program_name`
- `metadata`
- `qubits`
- `initial_state`
- `gates`
- `measurement`
- `units`
- `organization_schedule`
- `assertion`

Rules:

- `format` is fixed to `qmodel-v1`.
- `qubits` is the global ordered qubit list.
- `initial_state` is currently restricted to `zero`.
- `assertion` appears exactly once in each file.
- `measurement` is terminal-only in Version 1.
- `units` may overlap.
- `organization_schedule`, when present, overrides static `units` for abstract-trace construction.

## Organization Schedule Rules

Version 1 optionally supports an explicit linear organization-state chain:

- `organization_schedule.initial_state`
- `organization_schedule.states`

Each state entry contains:

- `name`
- `units`
- optional `transition`

Each transition entry contains:

- `gate_index`
- `next_state`

Rules:

- the schedule is a linear state chain, not a branching automaton
- `initial_state` must reference one defined state
- every non-terminal state must define exactly one transition
- transitions must cover gate indices in sequential order from `0` to `len(gates)-1`
- the final state must not define a transition
- every state must be reachable from `initial_state`

If `organization_schedule` is absent, the abstract backend falls back to the top-level static `units` field and reuses that same organization at every step.

## Gate Entry Rules

Each gate occurrence is written as one list item in `gates` and keeps program order.

Supported fields:

- `name`: gate name
- `targets`: ordered target qubits
- `controls`: ordered control qubits, optional
- `params`: gate parameters, optional
- `label`: optional human-readable label

General rules:

- Every qubit name used in a gate must already appear in `qubits`.
- `targets` must not be empty.
- `controls` and `targets` must not overlap inside one gate entry.
- Gate order is semantic and must be preserved exactly.
- `CCX` and general `MCX` remain first-class input constructs and do not need to be decomposed in `.qmodel`.

## Supported Gate Vocabulary

### Clifford gate set

The Version 1 gate vocabulary explicitly includes the following Clifford gates:

- `I`
- `X`
- `Y`
- `Z`
- `H`
- `S`
- `Sdg`
- `CX`
- `CZ`
- `SWAP`

These gates are recorded as first-class input symbols. This is useful for Clifford-only benchmarks, Clifford-dominant subcircuits, and future exact backends that do not need density-matrix simulation for such structures.

### Other supported gates

Version 1 also supports:

- `Ry`
- `T`
- `Tdg`
- `P`
- `CCX`
- `MCX`

`Ry` and `P` should carry their angle or symbolic parameter in `params`, for example:

```yaml
- name: Ry
  targets: [q0]
  params:
    theta: 1.0471975511965976
```

```yaml
- name: P
  targets: [q3]
  params:
    theta: "pi/3"
```

For `MCX`, the preferred representation is:

```yaml
- name: MCX
  controls: [q0, q1, q2]
  targets: [q3]
```

## Assertion Rules

Version 1 uses exactly one terminal assertion per file.

Supported assertion kinds:

- `reachability`
- `probability`

Version 1 uses the following concrete assertion shapes:

- `reachability`
  - `target.type` must be `basis_state`
  - `target.scope` is a non-empty ordered qubit list
  - `target.state` is a bitstring whose length matches `scope`
  - semantic meaning: along the execution trace, some reachable step has non-zero overlap with the basis-state projector on the given scope

- `probability`
  - `target.type` must be `measurement_outcome`
  - `target.scope` is a non-empty ordered qubit list
  - `target.outcomes` is a non-empty string list of computational outcomes
  - `comparator` and `threshold` are required
  - semantic meaning: after the declared gate sequence finishes, the terminal probability mass of the listed outcomes satisfies the numeric bound

## Example 1: Clifford Bell-State Model

```yaml
format: qmodel-v1
program_name: clifford_bell
metadata:
  family: clifford
  n: 2
qubits: [q0, q1]
initial_state: zero
gates:
  - name: H
    targets: [q0]
  - name: CX
    controls: [q0]
    targets: [q1]
measurement:
  qubits: [q0, q1]
  basis: computational
units:
  - name: bell_pair
    qubits: [q0, q1]
assertion:
  name: bell_measurement_mass
  kind: probability
  target:
    type: measurement_outcome
    scope: [q0, q1]
    outcomes: ["00", "11"]
  comparator: ">="
  threshold: 1.0
```

## Example 1b: Reachability Assertion

```yaml
format: qmodel-v1
program_name: single_h_reach
qubits: [q0]
initial_state: zero
gates:
  - name: H
    targets: [q0]
units:
  - name: local
    qubits: [q0]
assertion:
  name: reach_one
  kind: reachability
  target:
    type: basis_state
    scope: [q0]
    state: "1"
```

## Example 2: Clifford Gate Showcase

```yaml
format: qmodel-v1
program_name: clifford_gate_showcase
metadata:
  family: clifford
  n: 3
qubits: [q0, q1, q2]
initial_state: zero
gates:
  - name: H
    targets: [q0]
  - name: S
    targets: [q0]
  - name: X
    targets: [q1]
  - name: Z
    targets: [q1]
  - name: CZ
    controls: [q0]
    targets: [q1]
  - name: SWAP
    targets: [q1, q2]
measurement:
  qubits: [q0, q1, q2]
  basis: computational
units:
  - name: left_pair
    qubits: [q0, q1]
  - name: right_pair
    qubits: [q1, q2]
assertion:
  name: terminal_distribution_check
  kind: probability
  target:
    type: measurement_outcome
    scope: [q0, q1, q2]
    outcomes: ["001", "101"]
  comparator: ">="
  threshold: 0.0
```

## Example 3: Overlapping-Unit CCX Model

```yaml
format: qmodel-v1
program_name: ccx_overlap_demo
metadata:
  family: overlap_demo
  n: 5
qubits: [q0, q1, q2, q3, q4]
initial_state: zero
gates:
  - name: H
    targets: [q0]
  - name: H
    targets: [q1]
  - name: P
    targets: [q3]
    params:
      theta: "theta"
  - name: H
    targets: [q4]
  - name: CCX
    controls: [q0, q1]
    targets: [q2]
  - name: CX
    controls: [q2]
    targets: [q3]
  - name: CX
    controls: [q3]
    targets: [q4]
measurement:
  qubits: [q4]
  basis: computational
units:
  - name: u1
    qubits: [q0, q1, q2]
  - name: u2
    qubits: [q2, q3]
  - name: u3
    qubits: [q3, q4]
assertion:
  name: q4_one_probability
  kind: probability
  target:
    type: measurement_outcome
    scope: [q4]
    outcomes: ["1"]
  comparator: ">="
  threshold: 0.5
```

## Example 4: Explicit Organization-State Chain

```yaml
format: qmodel-v1
program_name: organization_schedule_chain
qubits: [q0, q1, q2]
initial_state: zero
gates:
  - name: H
    targets: [q0]
    label: prepare-q0
  - name: CX
    controls: [q0]
    targets: [q1]
    label: couple-q0-q1
  - name: CX
    controls: [q1]
    targets: [q2]
    label: couple-q1-q2
organization_schedule:
  initial_state: s0
  states:
    - name: s0
      units:
        - name: uq0
          qubits: [q0]
        - name: uq1
          qubits: [q1]
        - name: uq2
          qubits: [q2]
      transition:
        gate_index: 0
        next_state: s1
    - name: s1
      units:
        - name: uq0
          qubits: [q0]
        - name: uq1
          qubits: [q1]
        - name: uq2
          qubits: [q2]
      transition:
        gate_index: 1
        next_state: s2
    - name: s2
      units:
        - name: uq01
          qubits: [q0, q1]
        - name: uq2
          qubits: [q2]
      transition:
        gate_index: 2
        next_state: s3
    - name: s3
      units:
        - name: uq01
          qubits: [q0, q1]
        - name: uq12
          qubits: [q1, q2]
assertion:
  name: q2_one_probability
  kind: probability
  target:
    type: measurement_outcome
    scope: [q2]
    outcomes: ["1"]
  comparator: ">="
  threshold: 0.0
```
