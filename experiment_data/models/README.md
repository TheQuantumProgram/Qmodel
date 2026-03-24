# Formal Experiment Model Families

This directory stores formal `.qmodel` instances used for actual experiments. Small illustrative examples for parser or unit tests stay under `project_code/tests/models/`.

## Family Layout

- `GHZ/`
  - GHZ-state preparation and staircase entanglement chains
  - useful for tracking how overlapping two-qubit units evolve along long linear circuits
- `BV/`
  - Bernstein-Vazirani circuits with structured oracle implementations
  - useful for circuits dominated by Clifford gates and oracle-controlled interactions
- `Grover/`
  - Grover-search circuits, including oracle and diffusion structures
  - useful for repeated controlled operations and growing interaction ranges
- `AIQFT/`
  - approximate inverse-QFT recovery families
  - useful for structured phase-recovery circuits under bounded sliding-window organizations
- `Adder/`
  - ripple-carry adders and related arithmetic subcircuits
  - useful for structured `CX`/`CCX`-heavy local interactions
- `Custom/`
  - original circuit suites designed for this project
  - intended for overlap-heavy chains, clustered-then-cross-cluster couplings, and `CCX`/`MCX`-centric cases

## Current Formal Models

- `GHZ/ghz_10_staircase.qmodel`
  - 10-qubit GHZ preparation with a staircase `CX` chain
  - organization schedule starts from fully separated units and grows into overlapping two-qubit units
- `GHZ/ghz_20_staircase.qmodel`
- `GHZ/ghz_50_staircase.qmodel`
- `GHZ/ghz_100_staircase.qmodel`
- `GHZ/ghz_150_staircase.qmodel`
- `GHZ/ghz_200_staircase.qmodel`
  - standard GHZ family at the six planned qubit scales
- `GHZ/ghz_20_root_p025.qmodel`
- `GHZ/ghz_50_root_p075.qmodel`
  - biased-root GHZ variants using `Ry(theta)` on `q0` before the staircase `CX` chain
