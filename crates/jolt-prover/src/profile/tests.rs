use clap::Parser;
#[cfg(feature = "akita")]
use jolt_akita::AkitaChunkProfile;

#[cfg(feature = "akita")]
use super::{parse_akita_chunk_profile, OutputFormat, Workload};
use super::{BackendKind, MemoryBudget, ProfileArgs};

#[test]
fn streaming_option_selects_the_requested_families() {
    use jolt_kernels::optimized::StreamingFamilies;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: ProfileArgs,
    }
    for (selection, expected) in [
        (
            "instruction-input",
            StreamingFamilies {
                instruction_input: true,
                spartan: false,
            },
        ),
        (
            "spartan",
            StreamingFamilies {
                instruction_input: false,
                spartan: true,
            },
        ),
        (
            "instruction-input,spartan",
            StreamingFamilies {
                instruction_input: true,
                spartan: true,
            },
        ),
    ] {
        let cli = Cli::try_parse_from([
            "jolt-prover",
            "--name",
            "fibonacci",
            "--backend",
            "optimized",
            "--streaming",
            selection,
        ])
        .unwrap();
        assert_eq!(cli.args.streaming, Some(expected));
    }
}

#[test]
fn streaming_rejects_a_memory_budget_or_reference_backend() {
    use super::run;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: ProfileArgs,
    }
    let options = [
        "jolt-prover",
        "--name",
        "fibonacci",
        "--backend",
        "optimized",
        "--streaming",
        "instruction-input",
    ];
    assert!(Cli::try_parse_from(options.into_iter().chain(["--memory-budget", "32GiB"])).is_err());
    let mut args = Cli::try_parse_from(options).unwrap().args;
    args.memory_budget = Some(MemoryBudget::from_bytes(usize::MAX));
    assert!(run(&args)
        .unwrap_err()
        .to_string()
        .contains("cannot be combined with --memory-budget"));
    args.memory_budget = None;
    args.backend = BackendKind::Reference;
    assert!(run(&args)
        .unwrap_err()
        .to_string()
        .contains("requires --backend optimized"));
}

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

#[test]
fn streaming_profiles_have_distinct_artifact_and_csv_names() {
    use super::PROTOCOL_SUFFIX;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: ProfileArgs,
    }
    for (selection, suffix) in [
        (None, ""),
        (Some("instruction-input"), "_streaming_instruction_input"),
        (Some("spartan"), "_streaming_spartan"),
        (
            Some("instruction-input,spartan"),
            "_streaming_instruction_input_spartan",
        ),
        (
            Some("spartan,instruction-input"),
            "_streaming_instruction_input_spartan",
        ),
    ] {
        let mut options = vec![
            "jolt-prover",
            "--name",
            "fibonacci",
            "--backend",
            "optimized",
        ];
        if let Some(selection) = selection {
            options.extend(["--streaming", selection]);
        }
        let args = Cli::try_parse_from(options).unwrap().args;
        assert_eq!(args.workload_name(), format!("fibonacci{suffix}"));
        assert_eq!(
            args.benchmark_name(),
            format!("fibonacci{PROTOCOL_SUFFIX}{suffix}")
        );
        assert_eq!(
            args.trace_name(20),
            format!("modular_fibonacci{PROTOCOL_SUFFIX}_20_optimized{suffix}")
        );
    }
}

#[cfg(feature = "akita")]
#[test]
fn chunk_profiles_have_distinct_artifact_and_csv_names() {
    for (profile, workload_name, csv_name, trace_name) in [
        (
            AkitaChunkProfile::new(1, 0),
            "fibonacci",
            "fibonacci_akita",
            "modular_fibonacci_akita_20_optimized",
        ),
        (
            AkitaChunkProfile::new(2, 2),
            "fibonacci_w2r2",
            "fibonacci_akita_w2r2",
            "modular_fibonacci_akita_20_optimized_w2r2",
        ),
        (
            AkitaChunkProfile::new(4, 2),
            "fibonacci_w4r2",
            "fibonacci_akita_w4r2",
            "modular_fibonacci_akita_20_optimized_w4r2",
        ),
        (
            AkitaChunkProfile::new(8, 2),
            "fibonacci_w8r2",
            "fibonacci_akita_w8r2",
            "modular_fibonacci_akita_20_optimized_w8r2",
        ),
        (
            AkitaChunkProfile::new(16, 4),
            "fibonacci_w16r4",
            "fibonacci_akita_w16r4",
            "modular_fibonacci_akita_20_optimized_w16r4",
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
            streaming: None,
            guest_elf: None,
            input_file: None,
            result_json: None,
            akita_chunk_profile: profile,
            akita_prover_pool: false,
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
