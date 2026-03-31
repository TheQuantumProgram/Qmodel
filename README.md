# Quantum Program Modeling Project

This directory contains the executable code for the abstraction-based modeling and verification project.


## Quick Start

1. Create and activate a Python environment, for example:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the runnable environment snapshot:
   ```bash
   pip install -r requirements.txt
   ```
3. Run one model:
   ```bash
   python scripts/qmodel_cli.py run tests/models/clifford_bell.qmodel
   ```
4. Run all formal experiment models:
   ```bash
   python scripts/qmodel_cli.py run-all
   ```
5. Run one model family only:
   ```bash
   python scripts/qmodel_cli.py run-all --family GHZ
   ```
6. Enable the concrete backend only for small models when an exact reference is needed:
   ```bash
   python scripts/qmodel_cli.py run tests/models/clifford_bell.qmodel --run-concrete
   ```

## Layout

- `src/qmodel/`: Python package for program specification, parsing, concrete simulation, abstract modeling, assertions, and benchmarks.
- `docs/`: Format notes and local project conventions for handwritten models.
- `tests/models/`: Handwritten example `.qmodel` files used by the test suite.
- `experiment_data/models/`: Formal experiment models for actual evaluation runs, organized by algorithm family.
- `experiment_data/raw_results/`: Raw experiment outputs such as JSON and CSV.
- `experiment_data/summaries/`: Aggregated tables and processed summaries.
- `scripts/`: Command-line entry points and benchmark-running helpers.
- `tests/`: Unit tests for core components.

## Capabilities

Specification and parsing:
- core dataclasses for `QuantumProgramSpec`
- basic validation utilities
- package skeleton for later parser, concrete backend, abstract backend, assertions, and benchmarks
- `.qmodel` field rules and examples are documented
- supported gate vocabulary is recorded explicitly, including the Clifford gate set
- `.qmodel` files can now be parsed into `QuantumProgramSpec`
- `.qmodel` can now describe either a static unit layout or an explicit organization-state chain through `organization_schedule`

Concrete execution:
- `QuantumProgramSpec` can now be translated into a Qiskit `QuantumCircuit`
- supported gate mapping includes Clifford gates, `Ry`, `P`, `CCX`, `MCX`, and controlled `X`
- terminal measurement is appended from the declarative specification
- exact pre-measurement terminal states can be simulated through Qiskit statevectors
- terminal probability assertions over declared measurement outcomes can now be evaluated exactly
- basis-state reachability assertions can now be evaluated exactly along the concrete state trajectory

Abstract execution:
- abstract states store per-unit witness states and support projectors
- abstract states also carry internal reconstruction certificates produced by merge-update-rewrite steps
- abstract transitions update unit witnesses directly when a gate stays within one affected view
- multi-view or reorganized transitions reconstruct workspace inputs from existing unit states and certificates instead of storing a persistent full global witness inside the abstract trace
- trusted reconstruction mode allows tolerant stitching of overlapping units after consistency checks
- checked reconstruction mode treats missing shared certificates on overlapping joins as modeling errors
- reachability and terminal probability checks reconstruct queried scopes from unit-local witnesses and reconstruction certificates

Documentation:
- `docs/qmodel_format.md`: current `.qmodel` field rules, supported gate vocabulary, and examples
- `tests/models/clifford_bell.qmodel`: minimal Clifford example using an explicit organization-state chain
- `tests/models/clifford_gate_showcase.qmodel`: Clifford example using a repeated organization-state chain
- `tests/models/ccx_overlap_demo.qmodel`: overlapping-view example with stepwise organization changes aligned with the manuscript
- `experiment_data/models/README.md`: family-level catalog for formal experiment models

Formal Experiment-Model Families:
- `experiment_data/models/GHZ/`: GHZ-state preparation and related staircase-entanglement instances
- `experiment_data/models/BV/`: Bernstein-Vazirani instances with structured oracle layouts
- `experiment_data/models/Grover/`: Grover-search instances and oracle/diffusion variants
- `experiment_data/models/IQFT/`: approximate inverse-QFT recovery families with sliding-window phase interactions
- `experiment_data/models/IQFTCompare/`: small-scale IQFT comparison instances used to compare abstract model checking against full execution on `n = 10..20`
- `experiment_data/models/Adder/`: ripple-carry adders and related arithmetic-circuit families
- `experiment_data/models/Custom/`: original circuit suites for overlap-heavy and structured-control stress tests

GHZ Models:
- standard staircase GHZ: `n = 10, 20, 50, 100, 150, 200`
- biased-root staircase GHZ: `n = 20` with `p(root=1)=0.25`, and `n = 50` with `p(root=1)=0.75`

BV Models:
- sparse-oracle BV: `n = 10, 20, 50, 100, 150, 200`

IQFT Models:
- sliding-window IQFT recovery: `n = 10, 20, 50, 100, 150, 200`, all with window size `5`
- small comparison suite for abstract/full-execution cost ratios: `n = 10, 11, ..., 20`, all with window size `5`

Adder Models:
- ripple-carry adders with dynamic carry-window schedules: `n = 10, 20, 50, 100, 150, 200`

Grover Models:
- bounded-local-core multi-solution Grover instances: `n = 10, 20, 50, 100, 150, 200`

Custom Models:
- small positive baselines:
  - `custom_overlap_chain_prob_6`
  - `custom_back_edge_prob_6`
  - `custom_split_merge_prob_8`
  - `custom_ccx_ladder_reach_9`
  - `custom_uncompute_reach_8`
  - `custom_disconnected_product_prob_10`
- small counterexamples:
  - `custom_overlap_chain_counter_6`
  - `custom_split_merge_counter_8`
  - `custom_disconnected_product_counter_10`
  - `custom_ccx_ladder_counter_9`
  - `custom_uncompute_counter_8`
- medium positive models:
  - `custom_overlap_chain_prob_12`
  - `custom_back_edge_prob_12`
  - `custom_split_merge_prob_14`
  - `custom_ccx_ladder_reach_15`
  - `custom_disconnected_product_prob_20`

Limitations:
- the batch runner is intentionally lightweight and currently prints aggregated JSON payloads to stdout instead of writing raw-result files automatically
- tolerant reconstruction across overlapping units is still heuristic when no direct shared certificate covers the full requested workspace
- the exact scope-state provider in the concrete backend remains only as a reference/debug utility

Runner behavior:
- `scripts/qmodel_cli.py` provides the Quick Start entry point for single-model and batch execution
- `scripts/run_single.py` now runs the abstract backend by default
- the concrete backend is skipped by default so large-`n` models are not forced into exact global simulation
- pass `--run-concrete` only for small instances when an exact reference comparison is actually needed
- when `organization_schedule` is present, the runner and abstract trace builder use it as the per-step unit organization source instead of repeating static top-level `units`
- `scripts/run_single.py` records abstract verification time, explicit-state memory peaks, transition-time memory peaks, and the overall execution memory peak
- `scripts/run_single.py` also emits a `comparison` block with:
  - an abstract ideal pure-state lower-bound estimate
  - exact full-execution statevector/density-matrix space formulas
  - optional full-execution timing when the qubit count is below the automatic cutoff

Timing Fields:
- `abstract.elapsed_seconds`
  - abstract backend only
  - includes abstract trace construction plus abstract assertion checking
- `comparison.full_execution.time_benchmark.statevector_elapsed_seconds`
  - concrete full-execution baseline for exact terminal statevector simulation only
- `comparison.full_execution.time_benchmark.assertion_evaluation_seconds`
  - concrete assertion-checking time after the statevector is already available
- `comparison.full_execution.time_benchmark.concrete_backend_elapsed_seconds`
  - concrete full-execution baseline time for `statevector + assertion evaluation`
- `total_elapsed_seconds`
  - whole `run_single.py` wall-clock time
  - includes abstract execution, optional full-execution timing, JSON assembly, and other runner overhead
