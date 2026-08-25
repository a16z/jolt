#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "measurement harness: argument and fixture errors fail loudly, results go to stdout"
)]

mod arms;
mod fixture;
mod probe;

use std::fmt::Write as _;
use std::time::Duration;

use jolt_field::Fr;
use jolt_prover::profile::Workload;
use jolt_witness::JoltWitnessPlane;

use arms::VerticalTiming;
use fixture::Fixture;
use jolt_prover::profile::BackendKind;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Plane {
    Base,
    Advice,
}

struct Arm {
    name: &'static str,
    plane: Plane,
    run: fn(&Fixture, &dyn JoltWitnessPlane<Fr>, BackendKind) -> VerticalTiming,
}

const ARMS: &[Arm] = &[
    Arm {
        name: "witness-generation",
        plane: Plane::Base,
        run: arms::measure_witness_generation,
    },
    Arm {
        name: "advice-opening",
        plane: Plane::Advice,
        run: arms::advice_opening,
    },
    Arm {
        name: "booleanity-address",
        plane: Plane::Base,
        run: arms::measure_booleanity_address,
    },
    Arm {
        name: "booleanity-cycle",
        plane: Plane::Base,
        run: arms::measure_booleanity_cycle,
    },
    Arm {
        name: "bytecode-read-raf-address",
        plane: Plane::Base,
        run: arms::measure_bytecode_read_raf_address,
    },
    Arm {
        name: "bytecode-read-raf-cycle",
        plane: Plane::Base,
        run: arms::measure_bytecode_read_raf_cycle,
    },
    Arm {
        name: "bytecode-reduction-address",
        plane: Plane::Advice,
        run: arms::measure_bytecode_reduction_address,
    },
    Arm {
        name: "bytecode-reduction-cycle",
        plane: Plane::Advice,
        run: arms::measure_bytecode_reduction_cycle,
    },
    Arm {
        name: "commit",
        plane: Plane::Base,
        run: arms::measure_commit,
    },
    Arm {
        name: "hamming-weight-claim-reduction",
        plane: Plane::Base,
        run: arms::measure_hamming_weight_claim_reduction,
    },
    Arm {
        name: "inc-claim-reduction",
        plane: Plane::Base,
        run: arms::measure_inc_claim_reduction,
    },
    Arm {
        name: "instruction-claim-reduction",
        plane: Plane::Base,
        run: arms::measure_instruction_claim_reduction,
    },
    Arm {
        name: "instruction-input",
        plane: Plane::Base,
        run: arms::measure_instruction_input,
    },
    Arm {
        name: "instruction-ra-virtualization",
        plane: Plane::Base,
        run: arms::measure_instruction_ra_virtualization,
    },
    Arm {
        name: "instruction-read-raf",
        plane: Plane::Base,
        run: arms::measure_instruction_read_raf,
    },
    Arm {
        name: "joint-opening",
        plane: Plane::Base,
        run: arms::measure_joint_opening,
    },
    Arm {
        name: "program-image-reduction-address",
        plane: Plane::Advice,
        run: arms::measure_program_image_reduction_address,
    },
    Arm {
        name: "program-image-reduction-cycle",
        plane: Plane::Advice,
        run: arms::measure_program_image_reduction_cycle,
    },
    Arm {
        name: "ram-hamming-booleanity",
        plane: Plane::Base,
        run: arms::measure_ram_hamming_booleanity,
    },
    Arm {
        name: "ram-output-check",
        plane: Plane::Base,
        run: arms::measure_ram_output_check,
    },
    Arm {
        name: "ram-ra-claim-reduction",
        plane: Plane::Base,
        run: arms::measure_ram_ra_claim_reduction,
    },
    Arm {
        name: "ram-raf-evaluation",
        plane: Plane::Base,
        run: arms::measure_ram_raf_evaluation,
    },
    Arm {
        name: "ram-ra-virtualization",
        plane: Plane::Base,
        run: arms::measure_ram_ra_virtualization,
    },
    Arm {
        name: "ram-read-write",
        plane: Plane::Base,
        run: arms::measure_ram_read_write,
    },
    Arm {
        name: "ram-val-check",
        plane: Plane::Base,
        run: arms::measure_ram_val_check,
    },
    Arm {
        name: "registers-claim-reduction",
        plane: Plane::Base,
        run: arms::measure_registers_claim_reduction,
    },
    Arm {
        name: "registers-read-write",
        plane: Plane::Base,
        run: arms::measure_registers_read_write,
    },
    Arm {
        name: "registers-val-evaluation",
        plane: Plane::Base,
        run: arms::measure_registers_val_evaluation,
    },
    Arm {
        name: "spartan-outer",
        plane: Plane::Base,
        run: arms::measure_spartan_outer,
    },
    Arm {
        name: "spartan-product",
        plane: Plane::Base,
        run: arms::measure_spartan_product,
    },
    Arm {
        name: "spartan-shift",
        plane: Plane::Base,
        run: arms::measure_spartan_shift,
    },
    Arm {
        name: "trusted-advice-address",
        plane: Plane::Advice,
        run: arms::trusted_advice_address,
    },
    Arm {
        name: "trusted-advice-cycle",
        plane: Plane::Advice,
        run: arms::trusted_advice_cycle,
    },
    Arm {
        name: "untrusted-advice-address",
        plane: Plane::Advice,
        run: arms::untrusted_advice_address,
    },
    Arm {
        name: "untrusted-advice-cycle",
        plane: Plane::Advice,
        run: arms::untrusted_advice_cycle,
    },
];

struct Args {
    workload: Workload,
    scales: Vec<u32>,
    repeats: usize,
    bytecode_chunks: usize,
    gpus: usize,
    filter: Option<String>,
    csv: Option<String>,
    hold: Duration,
    smi_poll: Duration,
}

fn usage() -> ! {
    println!(
        "usage: cuda_relation_bench [--name <workload>] [--scales 16,22] [--repeats 3]\n\
         \x20                          [--bytecode-chunks 2] [--gpus 1] [--arm <substring>]\n\
         \x20                          [--csv <path>] [--hold-ms 1000] [--smi-poll-ms 500]\n\
         \n\
         workloads: fibonacci | sha2-chain | sha3-chain | btreemap\n\
         Default scales are 16 and 22; 25 is opt-in (it flips the one-hot chunk\n\
         geometry, and its fixture costs 60-120 s per scale).\n\
         Tiers are always optimized and cuda: reference is ~230x slower and is a\n\
         correctness oracle, not a performance baseline.\n\
         --hold-ms is the cuda-only device probe: the arm is looped for that long\n\
         while per-device memory is polled in process and nvidia-smi is polled for\n\
         utilization and power. 0 disables the probe. The timing table is measured\n\
         with no sampler running, so it stays comparable across runs. Each nvidia-smi\n\
         query costs ~45 ms of driver time on this box, so compare the probe's\n\
         iter ms against the timing table's total before trusting util%."
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut args = Args {
        workload: Workload::Sha2Chain,
        scales: vec![16, 22],
        repeats: 3,
        bytecode_chunks: 2,
        gpus: 1,
        filter: None,
        csv: None,
        hold: Duration::from_secs(1),
        smi_poll: Duration::from_millis(500),
    };
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = || argv.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "--name" => {
                let raw = value();
                args.workload = match raw.as_str() {
                    "fibonacci" => Workload::Fibonacci,
                    "sha2-chain" => Workload::Sha2Chain,
                    "sha3-chain" => Workload::Sha3Chain,
                    "btreemap" => Workload::BTreeMap,
                    _ => usage(),
                };
            }
            "--scales" => {
                args.scales = value()
                    .split(',')
                    .map(|s| s.trim().parse().unwrap_or_else(|_| usage()))
                    .collect();
            }
            "--repeats" => args.repeats = value().parse().unwrap_or_else(|_| usage()),
            "--bytecode-chunks" => {
                args.bytecode_chunks = value().parse().unwrap_or_else(|_| usage());
            }
            "--gpus" => args.gpus = value().parse().unwrap_or_else(|_| usage()),
            "--arm" => args.filter = Some(value()),
            "--csv" => args.csv = Some(value()),
            "--hold-ms" => {
                args.hold = Duration::from_millis(value().parse().unwrap_or_else(|_| usage()));
            }
            "--smi-poll-ms" => {
                args.smi_poll = Duration::from_millis(value().parse().unwrap_or_else(|_| usage()));
            }
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }
    if args.repeats == 0 || args.scales.is_empty() || args.gpus == 0 {
        usage();
    }
    jolt_kernels::cuda::request_devices(args.gpus);
    args
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn median_timing(runs: Vec<VerticalTiming>) -> VerticalTiming {
    VerticalTiming {
        log_t: runs[0].log_t,
        prepare: median(runs.iter().map(|r| r.prepare).collect()),
        address: median(runs.iter().map(|r| r.address).collect()),
        handoff: median(runs.iter().map(|r| r.handoff).collect()),
        cycle: median(runs.iter().map(|r| r.cycle).collect()),
        claims: median(runs.iter().map(|r| r.claims).collect()),
    }
}

const TIERS: [(&str, BackendKind); 2] = [
    ("optimized", BackendKind::Optimized),
    ("cuda", BackendKind::Cuda),
];

const MIB: f64 = 1024.0 * 1024.0;

fn main() {
    let args = parse_args();
    let mut csv = String::from(
        "arm,tier,log_t,prepare_ms,address_ms,handoff_ms,cycle_ms,claims_ms,total_ms,\
         d2h_calls,d2h_bytes,d2h_blocked_ms,h2d_calls,h2d_bytes,h2d_ms\n",
    );
    let mut gpu_csv = String::from(
        "arm,log_t,gpu,baseline_mib,peak_mib,own_mib,util_pct,mem_util_pct,watts,iteration_ms,polled_iteration_ms\n",
    );
    let devices = jolt_kernels::cuda::device_count();
    let probing = !args.hold.is_zero() && devices > 0;

    for &scale in &args.scales {
        println!(
            "\n=== {} at 2^{scale}, median of {} ===",
            args.workload.as_str(),
            args.repeats
        );
        let fixture = Fixture::build(args.workload, scale, args.bytecode_chunks);
        let base = fixture.base_witness();
        let advice = fixture.advice_witness();
        println!(
            "{:>34}  {:>9}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}",
            "arm", "tier", "prepare", "address", "handoff", "cycle", "claims", "total",
        );

        let mut probes: Vec<(&str, probe::GpuProbe)> = Vec::new();
        for arm in ARMS {
            if let Some(filter) = &args.filter {
                if !arm.name.contains(filter.as_str()) {
                    continue;
                }
            }
            let witness: &dyn JoltWitnessPlane<Fr> = match arm.plane {
                Plane::Base => &base,
                Plane::Advice => &advice,
            };
            for (tier, backend) in TIERS {
                jolt_kernels::cuda::xfer_stats::reset();
                let runs: Vec<VerticalTiming> = (0..args.repeats)
                    .map(|_| (arm.run)(&fixture, witness, backend))
                    .collect();
                let transfers = jolt_kernels::cuda::xfer_stats::snapshot();
                let timing = median_timing(runs);
                if timing.total().is_zero() {
                    println!(
                        "{:>34}  {:>9}  {:>75}",
                        arm.name, tier, "skipped: zero rounds in this geometry",
                    );
                    continue;
                }
                println!(
                    "{:>34}  {:>9}  {:>11.3?}  {:>11.3?}  {:>11.3?}  {:>11.3?}  {:>11.3?}  {:>11.3?}",
                    arm.name,
                    tier,
                    timing.prepare,
                    timing.address,
                    timing.handoff,
                    timing.cycle,
                    timing.claims,
                    timing.total(),
                );
                let ms = |d: Duration| d.as_secs_f64() * 1e3;
                let per = |value: u64| value / args.repeats.max(1) as u64;
                let _ = writeln!(
                    csv,
                    "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{:.4},{},{},{:.4}",
                    arm.name,
                    tier,
                    timing.log_t,
                    ms(timing.prepare),
                    ms(timing.address),
                    ms(timing.handoff),
                    ms(timing.cycle),
                    ms(timing.claims),
                    ms(timing.total()),
                    per(transfers.d2h.calls),
                    per(transfers.d2h.bytes),
                    transfers.d2h.nanos as f64 / 1.0e6 / args.repeats.max(1) as f64,
                    per(transfers.h2d.calls),
                    per(transfers.h2d.bytes),
                    transfers.h2d.nanos as f64 / 1.0e6 / args.repeats.max(1) as f64,
                );
                if transfers.d2h.calls > 0 {
                    println!(
                        "{:>34}  {:>9}  D2H {} calls, {:.1} MB, {:.1} ms blocked | H2D {} calls, {:.1} MB",
                        "",
                        tier,
                        per(transfers.d2h.calls),
                        per(transfers.d2h.bytes) as f64 / MIB,
                        transfers.d2h.nanos as f64 / 1.0e6 / args.repeats.max(1) as f64,
                        per(transfers.h2d.calls),
                        per(transfers.h2d.bytes) as f64 / MIB,
                    );
                }
                if probing && backend == BackendKind::Cuda {
                    probes.push((
                        arm.name,
                        probe::probe(devices, args.hold, args.smi_poll, || {
                            (arm.run)(&fixture, witness, backend)
                        }),
                    ));
                }
            }
        }

        if probes.is_empty() {
            continue;
        }
        println!(
            "\n{:>34}  {:>3}  {:>11}  {:>11}  {:>11}  {:>7}  {:>7}  {:>7}  {:>8}  {:>4}",
            "arm", "gpu", "baseline", "peak", "own", "util%", "mem%", "watts", "iter ms", "smi",
        );
        for (name, gpu) in &probes {
            for (ordinal, device) in gpu.devices.iter().enumerate() {
                let show = |value: Option<f64>| {
                    value.map_or_else(|| "--".to_owned(), |value| format!("{value:.1}"))
                };
                println!(
                    "{:>34}  {ordinal:>3}  {:>8.0} MiB  {:>8.0} MiB  {:>8.0} MiB  {:>7}  {:>7}  {:>7}  {:>8.1}  {:>4}",
                    name,
                    device.baseline as f64 / MIB,
                    device.peak as f64 / MIB,
                    device.own() as f64 / MIB,
                    show(device.util),
                    show(device.mem_util),
                    show(device.watts),
                    gpu.iteration.as_secs_f64() * 1e3,
                    gpu.smi_samples,
                );
                let field = |value: Option<f64>| {
                    value.map_or_else(String::new, |value| format!("{value:.2}"))
                };
                let _ = writeln!(
                    gpu_csv,
                    "{},{},{},{:.1},{:.1},{:.1},{},{},{},{:.3},{:.3}",
                    name,
                    scale,
                    ordinal,
                    device.baseline as f64 / MIB,
                    device.peak as f64 / MIB,
                    device.own() as f64 / MIB,
                    field(device.util),
                    field(device.mem_util),
                    field(device.watts),
                    gpu.iteration.as_secs_f64() * 1e3,
                    gpu.polled_iteration.as_secs_f64() * 1e3,
                );
            }
        }
    }

    if let Some(path) = &args.csv {
        std::fs::write(path, &csv).expect("write the CSV");
        println!("\nwrote {path}");
        if probing {
            let gpu_path = format!("{path}.gpu.csv");
            std::fs::write(&gpu_path, &gpu_csv).expect("write the GPU CSV");
            println!("wrote {gpu_path}");
        }
    }
}
