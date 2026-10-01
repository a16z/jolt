//! Stage-boundary hooks shared by the Dory and Akita provers.

use jolt_kernels::{MaybeAllocative, ProofSession};

/// Snapshot the surviving stage state, then purge allocator-retained temporaries.
#[cfg_attr(not(feature = "allocative"), expect(unused_variables))]
pub(crate) fn finish_stage(
    stage: &str,
    log_t: usize,
    session: &ProofSession,
    output: &impl MaybeAllocative,
) {
    #[cfg(feature = "allocative")]
    jolt_profiling::capture_heap_snapshot(stage, |snapshot| {
        snapshot.visit_root(output);
        snapshot.visit_root(session);
    });
    let _span = tracing::info_span!("release_retained_memory", stage).entered();
    jolt_kernels::mem::purge_retained_memory(log_t);
}
