"""Preserve the reference encoder's online state around independent evaluations."""
from parallel_evaluation import evaluate_parallel as original_evaluate


def evaluate_parallel(backend, cfg, episodes, deterministic):
    extra = backend.extra_runtime_state()
    try:
        return original_evaluate(backend, cfg, episodes, deterministic)
    finally:
        backend.restore_extra_runtime_state(extra)
