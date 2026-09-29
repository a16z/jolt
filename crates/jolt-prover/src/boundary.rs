//! Stage-boundary hooks shared by the Dory and Akita provers.

#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use jolt_kernels::{MaybeAllocative, ProofSession};

/// Write a profile-only heap snapshot of the stage output and the proof
/// session.
#[cfg(feature = "allocative")]
pub(crate) fn stage_flamegraph(stage: &str, session: &ProofSession, output: &impl MaybeAllocative) {
    let Some(prefix) = jolt_profiling::flamegraph_prefix() else {
        return;
    };
    let mut flamegraph = FlameGraphBuilder::default();
    flamegraph.visit_root(output);
    flamegraph.visit_root(session);
    jolt_profiling::write_flamegraph_folded(flamegraph, format!("{prefix}{stage}.folded"));
}

#[cfg(not(feature = "allocative"))]
pub(crate) fn stage_flamegraph(
    _stage: &str,
    _session: &ProofSession,
    _output: &impl MaybeAllocative,
) {
}

/// Purge allocator-retained pages after a stage drops its temporaries.
pub(crate) fn stage_boundary(stage: &str, log_t: usize) {
    let _span = tracing::info_span!("release_retained_memory", stage).entered();
    jolt_kernels::mem::purge_retained_memory(log_t);
}
