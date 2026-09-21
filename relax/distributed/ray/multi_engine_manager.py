# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compatibility imports for the shared inference lifecycle implementation."""

from relax.distributed.ray.inference_manager import (
    InferenceManager as MultiEngineManager,
)
from relax.distributed.ray.inference_manager import (
    _InferenceObservation as _InferenceObservation,
)
from relax.distributed.ray.inference_manager import (
    _is_engine_dead as _is_engine_dead,
)
from relax.distributed.ray.inference_manager import (
    _model_discovery_state as _model_discovery_state,
)
from relax.distributed.ray.inference_manager import (
    ray as ray,
)


__all__ = ["MultiEngineManager"]
