use clap::Parser;
#[cfg(feature = "akita")]
use jolt_akita::AkitaChunkProfile;

#[cfg(feature = "akita")]
use super::{parse_akita_chunk_profile, BackendKind, OutputFormat, Workload, PROTOCOL_SUFFIX};
use super::{MemoryBudget, ProfileArgs};

#[test]
fn profile_owners_reserve_capacities_and_the_public_io_request_once() {
    use common::jolt_device::JoltDevice;
    use jolt_program::execution::{JoltProgram, MemoryImage, TraceData, TraceOutput};
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_riscv::JoltTraceRow;

    use super::{reserve_profile_owners, ProofField};

    let program = JoltProgram::from_parts(
        Vec::with_capacity(23),
        Vec::new(),
        Vec::with_capacity(31),
        0,
        0,
    );
    let mut device = JoltDevice::default();
    device.inputs.reserve_exact(13);
    device.inputs.extend([1, 2, 3]);
    device.trusted_advice.extend([4, 5]);
    device.untrusted_advice.push(6);
    device.outputs.extend([7, 8, 9, 10]);
    let preprocessing = JoltProgramPreprocessing::new(
        Vec::new(),
        Vec::new(),
        device.memory_layout.clone(),
        0,
        1 << 13,
        program.profile,
    )
    .unwrap();
    let trace = TraceOutput::new(
        TraceData::new(vec![JoltTraceRow::default(); 4]),
        device,
        Some(MemoryImage {
            bytes: Vec::with_capacity(17),
        }),
        Some(Vec::with_capacity(19)),
    );
    let budget = MemoryBudget::from_bytes(usize::MAX);
    let reserved =
        reserve_profile_owners::<ProofField>(budget, &trace, &program, &preprocessing).unwrap();
    assert_eq!(
        budget.remaining() - reserved.remaining(),
        trace.heap_bytes() - trace.trace.heap_bytes()
            + program.heap_bytes()
            + preprocessing.heap_bytes()
            + 10
    );
}

/// `--memory-budget` takes bytes with an optional binary suffix through the
/// CLI and rejects anything else.
#[test]
fn memory_budget_parses_bytes_and_binary_suffixes() {
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: ProfileArgs,
    }
    let parse = |budget: &[&str]| {
        Cli::try_parse_from(["jolt-prover", "--name", "fibonacci"].iter().chain(budget))
            .map(|cli| cli.args.memory_budget)
    };
    assert_eq!(parse(&[]).unwrap(), None);
    for (value, bytes) in [
        ("1536", 1536),
        ("3MiB", 3_145_728),
        ("32GiB", 34_359_738_368),
    ] {
        assert_eq!(
            parse(&["--memory-budget", value]).unwrap(),
            Some(MemoryBudget::from_bytes(bytes)),
            "{value}"
        );
    }
    for value in ["", "1.5GiB", "32GB", "-1", "18446744073709551615KiB"] {
        assert!(parse(&["--memory-budget", value]).is_err(), "{value}");
    }
}

#[cfg(feature = "akita")]
#[test]
fn chunk_profiles_have_distinct_artifact_and_csv_names() {
    for (profile, workload_name, csv_name, trace_name) in [
        (
            AkitaChunkProfile::new(1, 0),
            "fibonacci",
            format!("fibonacci{PROTOCOL_SUFFIX}"),
            format!("modular_fibonacci{PROTOCOL_SUFFIX}_20_optimized"),
        ),
        (
            AkitaChunkProfile::new(2, 2),
            "fibonacci_w2r2",
            format!("fibonacci{PROTOCOL_SUFFIX}_w2r2"),
            format!("modular_fibonacci{PROTOCOL_SUFFIX}_20_optimized_w2r2"),
        ),
        (
            AkitaChunkProfile::new(4, 2),
            "fibonacci_w4r2",
            format!("fibonacci{PROTOCOL_SUFFIX}_w4r2"),
            format!("modular_fibonacci{PROTOCOL_SUFFIX}_20_optimized_w4r2"),
        ),
        (
            AkitaChunkProfile::new(8, 2),
            "fibonacci_w8r2",
            format!("fibonacci{PROTOCOL_SUFFIX}_w8r2"),
            format!("modular_fibonacci{PROTOCOL_SUFFIX}_20_optimized_w8r2"),
        ),
        (
            AkitaChunkProfile::new(16, 4),
            "fibonacci_w16r4",
            format!("fibonacci{PROTOCOL_SUFFIX}_w16r4"),
            format!("modular_fibonacci{PROTOCOL_SUFFIX}_20_optimized_w16r4"),
        ),
    ] {
        assert_eq!(
            parse_akita_chunk_profile(&format!("w{}r{}", profile.chunks, profile.levels)),
            Ok(profile)
        );
        let args = ProfileArgs {
            name: Workload::Fibonacci,
            scale: Some(20),
            format: OutputFormat::None,
            backend: BackendKind::Optimized,
            guest_elf: None,
            input_file: None,
            result_json: None,
            akita_chunk_profile: profile,
            memory_budget: None,
        };
        assert_eq!(args.workload_name(), workload_name);
        assert_eq!(args.benchmark_name(), csv_name);
        assert_eq!(args.trace_name(20), trace_name);
    }
}

#[cfg(feature = "akita")]
#[test]
fn chunk_profile_parser_rejects_invalid_geometry() {
    for value in ["w0r2", "w3r2", "w1r2", "w2r0", "w2", "w128r2"] {
        assert!(parse_akita_chunk_profile(value).is_err(), "{value}");
    }
}
