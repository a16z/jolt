use super::{BackendKind, OutputFormat, ProfileArgs, Workload, PROTOCOL_SUFFIX};
use jolt_akita::AkitaChunkProfile;

#[test]
fn chunk_profiles_have_distinct_artifact_and_csv_names() {
    for (profile, workload_name, csv_name, trace_name) in [
        (
            AkitaChunkProfile::Single,
            "fibonacci",
            format!("fibonacci{PROTOCOL_SUFFIX}"),
            format!("modular_fibonacci{PROTOCOL_SUFFIX}_20_optimized"),
        ),
        (
            AkitaChunkProfile::Two,
            "fibonacci_w2r2",
            format!("fibonacci{PROTOCOL_SUFFIX}_w2r2"),
            format!("modular_fibonacci{PROTOCOL_SUFFIX}_20_optimized_w2r2"),
        ),
        (
            AkitaChunkProfile::Four,
            "fibonacci_w4r2",
            format!("fibonacci{PROTOCOL_SUFFIX}_w4r2"),
            format!("modular_fibonacci{PROTOCOL_SUFFIX}_20_optimized_w4r2"),
        ),
        (
            AkitaChunkProfile::Eight,
            "fibonacci_w8r2",
            format!("fibonacci{PROTOCOL_SUFFIX}_w8r2"),
            format!("modular_fibonacci{PROTOCOL_SUFFIX}_20_optimized_w8r2"),
        ),
    ] {
        let args = ProfileArgs {
            name: Workload::Fibonacci,
            scale: Some(20),
            format: OutputFormat::None,
            backend: BackendKind::Optimized,
            akita_chunk_profile: profile,
        };
        assert_eq!(args.workload_name(), workload_name);
        assert_eq!(args.benchmark_name(), csv_name);
        assert_eq!(args.trace_name(20), trace_name);
    }
}
