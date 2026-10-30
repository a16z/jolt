#![cfg(all(feature = "profiling", feature = "cuda", not(feature = "akita")))]
#![expect(clippy::expect_used)]

use jolt_profiling::taxonomy::{ROOT_SPAN, STAGE_SPANS};
use jolt_prover::profile::{BackendKind, OutputFormat, ProfileArgs, Workload};

#[test]
fn cuda_profile_proves_verifies_and_records_the_backend() {
    let artifacts = jolt_prover::profile::run(&ProfileArgs {
        name: Workload::Fibonacci,
        scale: Some(13),
        format: OutputFormat::Chrome,
        backend: BackendKind::Cuda,
        gpus: None,
    });
    let summary = artifacts.summary.expect("profile summary");
    assert_eq!(summary.run.backend, "cuda");
    assert_eq!(summary.run.scale_log2, 13);
    assert_eq!(summary.root.expect("root span").label, ROOT_SPAN);
    for stage in STAGE_SPANS {
        assert!(summary.stages.iter().any(|entry| entry.label == stage));
    }
}
