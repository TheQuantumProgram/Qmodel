"""Abstract-state modeling backend."""

from .state import (
    AbstractState,
    AbstractTransition,
    AbstractUnitState,
    ReconstructionCertificate,
    abstract_local_state,
    support_projector,
)
from .property_checking import (
    AbstractPropertyCheckingError,
    evaluate_assertion,
    evaluate_reachability_assertion,
    evaluate_terminal_probability_assertion,
    evaluate_terminal_probability_assertion_on_state,
    final_scope_witness,
    final_scope_witness_from_state,
    state_scope_witness,
)
from .transition import (
    AbstractExecutionFinalState,
    AbstractExecutionStats,
    AbstractExecutionTrace,
    build_abstract_trace,
    execute_abstract_to_final_state,
    gate_support,
    merge_update_rewrite,
    reconstruct_scope_state,
    select_affected_views,
)

__all__ = [
    "AbstractState",
    "AbstractTransition",
    "AbstractUnitState",
    "AbstractExecutionFinalState",
    "AbstractExecutionStats",
    "AbstractExecutionTrace",
    "AbstractPropertyCheckingError",
    "ReconstructionCertificate",
    "abstract_local_state",
    "build_abstract_trace",
    "evaluate_assertion",
    "evaluate_reachability_assertion",
    "evaluate_terminal_probability_assertion",
    "evaluate_terminal_probability_assertion_on_state",
    "final_scope_witness",
    "final_scope_witness_from_state",
    "gate_support",
    "execute_abstract_to_final_state",
    "merge_update_rewrite",
    "reconstruct_scope_state",
    "select_affected_views",
    "state_scope_witness",
    "support_projector",
]
