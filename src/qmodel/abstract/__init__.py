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
    final_scope_witness,
    state_scope_witness,
)
from .transition import (
    AbstractExecutionTrace,
    build_abstract_trace,
    gate_support,
    merge_update_rewrite,
    reconstruct_scope_state,
    select_affected_views,
)

__all__ = [
    "AbstractState",
    "AbstractTransition",
    "AbstractUnitState",
    "AbstractExecutionTrace",
    "AbstractPropertyCheckingError",
    "ReconstructionCertificate",
    "abstract_local_state",
    "build_abstract_trace",
    "evaluate_assertion",
    "evaluate_reachability_assertion",
    "evaluate_terminal_probability_assertion",
    "final_scope_witness",
    "gate_support",
    "merge_update_rewrite",
    "reconstruct_scope_state",
    "select_affected_views",
    "state_scope_witness",
    "support_projector",
]
