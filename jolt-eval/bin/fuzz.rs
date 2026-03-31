use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use jolt_eval::invariant::completeness_prover::ProverCompletenessInvariant;
use jolt_eval::invariant::completeness_verifier::VerifierCompletenessInvariant;
use jolt_eval::invariant::determinism::DeterminismInvariant;
use jolt_eval::invariant::serialization_roundtrip::SerializationRoundtripInvariant;
use jolt_eval::invariant::soundness::SoundnessInvariant;
use jolt_eval::invariant::synthesis::SynthesisRegistry;
use jolt_eval::invariant::zk_consistency::ZkConsistencyInvariant;
use jolt_eval::invariant::{DynInvariant, InvariantReport, SynthesisTarget};
use jolt_eval::TestCase;

#[derive(Parser)]
#[command(name = "fuzz")]
#[command(about = "Fuzz-test Jolt invariants with random inputs")]
struct Cli {
    /// Only fuzz the named invariant (default: all fuzzable)
    #[arg(long)]
    invariant: Option<String>,

    /// Total number of fuzz iterations (across all invariants)
    #[arg(long, default_value = "1000")]
    iterations: usize,

    /// Maximum wall-clock duration (e.g. "60s", "5m", "1h")
    #[arg(long)]
    duration: Option<String>,

    /// Size of random byte buffer fed to Arbitrary (bytes)
    #[arg(long, default_value = "4096")]
    input_size: usize,

    /// Path to a pre-compiled guest ELF
    #[arg(long)]
    elf: Option<String>,

    /// Max trace length for the test program
    #[arg(long, default_value = "65536")]
    max_trace_length: usize,

    /// List available fuzzable invariants and exit
    #[arg(long)]
    list: bool,
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let test_case = if let Some(elf_path) = &cli.elf {
        let elf_bytes = std::fs::read(elf_path)?;
        let memory_config = common::jolt_device::MemoryConfig {
            max_input_size: 4096,
            max_output_size: 4096,
            max_untrusted_advice_size: 0,
            max_trusted_advice_size: 0,
            stack_size: 65536,
            heap_size: 32768,
            program_size: None,
        };
        Arc::new(TestCase {
            elf_contents: elf_bytes,
            memory_config,
            max_trace_length: cli.max_trace_length,
        })
    } else if !cli.list {
        eprintln!("Error: --elf <path> is required. Provide a pre-compiled guest ELF.");
        std::process::exit(1);
    } else {
        // --list doesn't need an ELF; use a dummy to populate names
        print_available_invariants();
        return Ok(());
    };

    let default_inputs = vec![];
    let mut registry = SynthesisRegistry::new();
    register_invariants(&mut registry, &test_case, &default_inputs);

    if cli.list {
        for inv in registry.for_target(SynthesisTarget::Fuzz) {
            println!("  {}", inv.name());
        }
        return Ok(());
    }

    let fuzzable: Vec<&dyn DynInvariant> = if let Some(name) = &cli.invariant {
        let matches: Vec<_> = registry
            .for_target(SynthesisTarget::Fuzz)
            .into_iter()
            .filter(|inv| inv.name() == name.as_str())
            .collect();
        if matches.is_empty() {
            eprintln!("Invariant '{name}' not found or not fuzzable.");
            eprintln!("Run with --list to see available invariants.");
            std::process::exit(1);
        }
        matches
    } else {
        registry.for_target(SynthesisTarget::Fuzz)
    };

    if fuzzable.is_empty() {
        eprintln!("No fuzzable invariants registered.");
        std::process::exit(1);
    }

    let deadline = cli.duration.as_deref().map(|s| {
        let dur = parse_duration(s).unwrap_or_else(|| {
            eprintln!("Invalid duration '{s}'. Use e.g. 60s, 5m, 1h.");
            std::process::exit(1);
        });
        Instant::now() + dur
    });

    println!(
        "Fuzzing {} invariant(s), {} iterations, input size {} bytes",
        fuzzable.len(),
        cli.iterations,
        cli.input_size,
    );
    if let Some(d) = &cli.duration {
        println!("Time limit: {d}");
    }
    println!();

    let mut total_checks = 0usize;
    let mut total_violations = 0usize;
    let start = Instant::now();

    for inv in &fuzzable {
        println!("  {} — setting up...", inv.name());

        // DynInvariant::run_checks handles setup internally, but for a fuzz
        // loop we want to amortize setup across many iterations. Use run_checks
        // in batches.
        let per_invariant = cli.iterations / fuzzable.len();
        let mut checks = 0usize;
        let mut violations = Vec::new();

        // Run in batches so we can check the deadline between batches
        let batch_size = per_invariant.min(100);
        let mut remaining = per_invariant;

        while remaining > 0 {
            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    println!("  (time limit reached)");
                    break;
                }
            }

            let n = remaining.min(batch_size);
            let results = inv.run_checks(n);
            for r in &results {
                checks += 1;
                if let Err(e) = r {
                    violations.push(e.to_string());
                }
            }
            remaining = remaining.saturating_sub(n);
        }

        let report = InvariantReport {
            name: inv.name().to_string(),
            total: checks,
            passed: checks - violations.len(),
            failed: violations.len(),
            violations: violations.clone(),
        };
        print_report(&report);

        total_checks += checks;
        total_violations += violations.len();
    }

    let elapsed = start.elapsed();
    println!();
    println!(
        "Done: {} checks in {:.1}s, {} violations",
        total_checks,
        elapsed.as_secs_f64(),
        total_violations,
    );

    if total_violations > 0 {
        std::process::exit(1);
    }

    Ok(())
}

fn register_invariants(
    registry: &mut SynthesisRegistry,
    test_case: &Arc<TestCase>,
    default_inputs: &[u8],
) {
    registry.register(Box::new(SoundnessInvariant::new(
        Arc::clone(test_case),
        default_inputs.to_vec(),
    )));
    registry.register(Box::new(VerifierCompletenessInvariant::new(Arc::clone(
        test_case,
    ))));
    registry.register(Box::new(ProverCompletenessInvariant::new(Arc::clone(
        test_case,
    ))));
    registry.register(Box::new(DeterminismInvariant::new(Arc::clone(test_case))));
    registry.register(Box::new(SerializationRoundtripInvariant::new(
        Arc::clone(test_case),
        default_inputs.to_vec(),
    )));
    registry.register(Box::new(ZkConsistencyInvariant::new(Arc::clone(
        test_case,
    ))));
}

fn print_available_invariants() {
    println!("Fuzzable invariants:");
    println!("  soundness");
    println!("  verifier_completeness");
    println!("  prover_completeness");
    println!("  determinism");
    println!("  serialization_roundtrip");
    println!("  zk_consistency");
}

fn print_report(report: &InvariantReport) {
    if report.failed == 0 {
        println!(
            "  {} — {}/{} passed",
            report.name, report.passed, report.total
        );
    } else {
        println!(
            "  {} — FAILED {}/{} checks",
            report.name, report.failed, report.total
        );
        for (i, v) in report.violations.iter().enumerate().take(5) {
            println!("    [{i}] {v}");
        }
        if report.violations.len() > 5 {
            println!("    ... and {} more", report.violations.len() - 5);
        }
    }
}

fn parse_duration(s: &str) -> Option<Duration> {
    let s = s.trim();
    if let Some(n) = s.strip_suffix('s') {
        n.parse::<u64>().ok().map(Duration::from_secs)
    } else if let Some(n) = s.strip_suffix('m') {
        n.parse::<u64>().ok().map(|m| Duration::from_secs(m * 60))
    } else if let Some(n) = s.strip_suffix('h') {
        n.parse::<u64>()
            .ok()
            .map(|h| Duration::from_secs(h * 3600))
    } else {
        s.parse::<u64>().ok().map(Duration::from_secs)
    }
}
