"""XCS (Accelerated Compound Systems) for Ember.

Provides a computational graph-based system for building, optimizing, and
executing complex operator pipelines with automatic parallelization and
optimization.
"""

# === API Types ===
from ember.xcs.api.types import ExecutionResult as APIExecutionResult
from ember.xcs.api.types import (
    JITOptions,
    TransformOptions,
    XCSExecutionOptions,
)
from ember.xcs.common.plans import ExecutionResult, XCSPlan, XCSTask
from ember.xcs.engine.execution_options import (
    ExecutionOptions,
    execution_options,
)

# === Execution Engine ===
from ember.xcs.engine.unified_engine import (
    ExecutionMetrics,
    GraphExecutor,
    execute_graph,
)
from ember.xcs.graph.dependency_analyzer import DependencyAnalyzer
from ember.xcs.graph.graph_builder import EnhancedTraceGraphBuilder, GraphBuilder

# === Graph Representation ===
from ember.xcs.graph.xcs_graph import XCSGraph, XCSNode

# === Core JIT System ===
from ember.xcs.jit import JITCache, JITMode, explain_jit_selection, get_jit_stats, jit

# === Scheduler System ===
from ember.xcs.schedulers.base_scheduler import BaseScheduler
from ember.xcs.schedulers.factory import create_scheduler
from ember.xcs.schedulers.unified_scheduler import (
    NoOpScheduler,
    ParallelScheduler,
    SequentialScheduler,
    TopologicalScheduler,
    WaveScheduler,
)
from ember.xcs.tracer._context_types import TraceContextData
from ember.xcs.tracer.autograph import AutoGraphBuilder, autograph

# === Tracing Infrastructure ===
from ember.xcs.tracer.xcs_tracing import TracerContext, TraceRecord
from ember.xcs.transforms.mesh import DeviceMesh, PartitionSpec, mesh_sharded
from ember.xcs.transforms.pmap import pjit, pmap

# === Transformations ===
from ember.xcs.transforms.transform_base import (
    BaseTransformation,
    BatchingOptions,
    ParallelOptions,
    TransformError,
    compose,
)
from ember.xcs.transforms.vmap import vmap

__all__ = [
    # Core JIT system
    "jit",
    "JITMode",
    "get_jit_stats",
    "JITCache",
    "explain_jit_selection",
    # API Types
    "JITOptions",
    "XCSExecutionOptions",
    "APIExecutionResult",
    "TransformOptions",
    # Tracing infrastructure
    "TracerContext",
    "TraceRecord",
    "TraceContextData",
    "AutoGraphBuilder",
    "autograph",
    # Graph representation
    "XCSGraph",
    "XCSNode",
    "DependencyAnalyzer",
    "GraphBuilder",
    "EnhancedTraceGraphBuilder",
    # Execution engine
    "execute_graph",
    "ExecutionOptions",
    "execution_options",
    "GraphExecutor",
    "ExecutionMetrics",
    "XCSPlan",
    "XCSTask",
    "ExecutionResult",
    # Scheduler system
    "BaseScheduler",
    "NoOpScheduler",
    "ParallelScheduler",
    "SequentialScheduler",
    "TopologicalScheduler",
    "WaveScheduler",
    "create_scheduler",
    # Transformations
    "vmap",
    "pmap",
    "pjit",
    "DeviceMesh",
    "PartitionSpec",
    "mesh_sharded",
    "compose",
    "TransformError",
    "BaseTransformation",
    "BatchingOptions",
    "ParallelOptions",
]
