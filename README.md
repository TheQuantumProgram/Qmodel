# Quantum Program Modeling Project

This repository contains the executable package for abstraction-based modeling and verification of circuit-oriented quantum programs.

## Quick Start

1. Clone the repository and enter the project directory:
   ```bash
   git clone https://github.com/TheQuantumProgram/Qmodel.git
   cd Qmodel
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the package and its dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the included example model:
   ```bash
   qmodel run examples/clifford_bell.qmodel
   ```

5. Run the same model with the concrete reference backend enabled:
   ```bash
   qmodel run examples/clifford_bell.qmodel --run-concrete
   ```

6. Run every `.qmodel` file under a directory:
   ```bash
   qmodel run-all examples
   ```

The CLI prints a JSON payload containing the assertion result, abstract execution statistics, and comparison metadata.

## Layout

- `src/qmodel/`: Python package for specifications, parsing, concrete simulation, abstract execution, property checking, and the command-line interface.
- `examples/`: Small runnable `.qmodel` examples.
- `docs/`: Model-format notes and result-schema documentation.
- `requirements.txt`: Environment snapshot for reproducing the package setup.
- `pyproject.toml`: Package metadata and CLI entry-point definition.

## Modeling Format

A `.qmodel` file describes one verification instance:

- a fixed ordered qubit register
- a sequence of gate occurrences
- an optional terminal measurement
- either static units or an explicit `organization_schedule`
- one assertion

The current assertion kinds are:

- `probability`: terminal measurement or bitwise measurement probability checking
- `reachability`: path-style basis-state reachability along the execution trace
- `terminal_reachability`: basis-state reachability at the final program state

The model-format rules and examples are documented in `docs/qmodel_format.md`.

## CLI

Run one model:

```bash
qmodel run path/to/model.qmodel
```

Run one model with exact concrete comparison:

```bash
qmodel run path/to/model.qmodel --run-concrete
```

Run a directory of models:

```bash
qmodel run-all path/to/models
```

Filter a model directory by family subdirectory:

```bash
qmodel run-all path/to/models --family GHZ
```

Select the abstract reconstruction mode:

```bash
qmodel run path/to/model.qmodel --mode checked
```

## Capabilities

Specification and parsing:

- `QuantumProgramSpec` dataclasses for declarative quantum-program instances
- validation for qubits, gates, measurements, organization schedules, and assertions
- `.qmodel` parsing into the shared in-memory specification
- support for static unit layouts and explicit organization-state chains

Concrete execution:

- translation from `QuantumProgramSpec` to Qiskit circuits
- support for `I`, `X`, `Y`, `Z`, `H`, `S`, `Sdg`, `T`, `Tdg`, `Ry`, `P`, `CX`, `CP`, `CZ`, `SWAP`, `CCX`, and `MCX`
- exact statevector simulation for small reference runs
- terminal probability, path reachability, and terminal reachability evaluation

Abstract execution:

- unit-local witness states and support projectors
- gate-labeled abstract transitions over static or scheduled unit organizations
- single-view updates for gates contained in one affected unit
- cross-unit workspace reconstruction for coupled updates
- trusted and checked reconstruction modes
- explicit-state storage statistics for abstract states and transition workspaces

## Output

`qmodel run` returns a JSON object with:

- `abstract`: assertion result and abstract execution statistics
- `concrete`: exact reference result when `--run-concrete` is enabled
- `comparison`: statevector and abstract-storage comparison metadata
- `assertion_kind`, `assertion_name`, `program_name`, and timing fields

The result schema is documented in `docs/run_single_result_schema.json`.
