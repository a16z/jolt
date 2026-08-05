//! Isolated stage-0 contention objective: times the Metal witness commitment
//! and the hoisted trace-record walk separately and co-running, without any
//! full prove.
//!
//! ```text
//! /usr/bin/lockf -k /tmp/jolt-metal-wave2-cargo.lock cargo run --release \
//!     -p jolt-eval --bin st0-contention --features metal -- \
//!     --scale 22 --iters 5 --legs commit,walk,corun
//! ```
//!
//! Legs (per `--iters` iteration, one JSON line each):
//!
//! - `commit`  — the Metal commit slot alone (production stage-0 grid/ids);
//!   `commit-N` overrides the G1 segment cap for same-binary A/B.
//! - `g1-N`    — one production-shaped `jk_g1_seg_sum` superchunk at segment
//!   cap `N`, reporting device time, useful GB/s, and Fq Montgomery Gmul/s.
//! - `g1s-N`   — the same dispatch through the retained serial A/B oracle.
//! - `walk`    — the production background record walk alone
//!   (`spawn_shared_record_collect` + the stage-1 join), on its capped pool.
//! - `corun`   — the production shape: spawn the walk, run the commit, join.
//! - `soak`    — the commit co-running with `--soak-threads` memory workers:
//!   `--soak-mode stream` (default) re-fills resident 256 MiB buffers — the
//!   bandwidth-only control; `--soak-mode fault` maps, first-touches, and
//!   unmaps fresh 64 MiB anonymous regions — the page-fault/VM-pressure
//!   control at negligible bandwidth. Together they separate fabric traffic
//!   from VM-subsystem serialization.
//!
//! Mechanism knobs (all read by the production spawn, same binary):
//! `JOLT_RECORD_BACKGROUND_THREADS`, `JOLT_RECORD_HOIST_DELAY_MS`,
//! `JOLT_RECORD_QOS=background|utility`. Guest compile, tracing, and PCS
//! setup happen once, untimed; every iteration re-runs the timed leg on the
//! same witness (fresh `ProofSession`, fresh lane allocations).

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    harness::run();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("st0-contention requires --features metal on macOS");
    std::process::exit(1);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod harness {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

    use clap::Parser;
    use common::jolt_device::MemoryConfig;
    use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_inlines_sha2 as _;
    use jolt_kernels::metal::testing::gpu_lock;
    use jolt_kernels::metal::{G1SegBenchCase, G1SegBenchFixture};
    use jolt_kernels::optimized::trace_record::{
        join_shared_record_for_bench, spawn_shared_record_collect,
    };
    use jolt_kernels::{CommitmentGrid, JoltBackend};
    use jolt_program::execution::{
        ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
    };
    use jolt_prover::{JoltProverPreprocessing, ProverConfig};
    use jolt_prover_legacy::curve::Bn254Curve;
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme;
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
    use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_witness::{
        JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessOracle, JoltWitnessPlane, RowSource,
        TraceBackend,
    };
    use tracer::execution_backend::TracerBackend;

    /// Same guest-input mapping as `modular_benchmark` for sha2-chain.
    const CYCLES_PER_SHA256: f64 = 3396.0;
    const SAFETY_MARGIN: f64 = 0.9;

    #[derive(Parser, Debug)]
    struct Cli {
        /// Padded trace scale (log2 cycles).
        #[clap(long, default_value_t = 22)]
        scale: usize,

        /// Timed iterations per leg.
        #[clap(long, default_value_t = 5)]
        iters: usize,

        /// Comma-separated legs: commit, walk, corun, soak.
        #[clap(long, default_value = "commit,walk,corun")]
        legs: String,

        /// Plain memory-streamer threads for the soak leg.
        #[clap(long, default_value_t = 8)]
        soak_threads: usize,

        /// Soak worker shape: `stream` (bandwidth) or `fault` (VM pressure).
        #[clap(long, default_value = "stream")]
        soak_mode: String,

        /// Dirty (touch, then free) this many GiB before every iteration —
        /// the page-residency probe emulating a just-finished prove's
        /// depleted free list.
        #[clap(long, default_value_t = 0)]
        dirty_gb: usize,
    }

    /// Closed `TraceRecord::collect` spans: `(start, end)` per walk, drained
    /// once per iteration. A `tracing` layer is the only way to see the
    /// walk's own duration inside a co-run — the join point only observes
    /// its completion.
    static WALK_SPANS: Mutex<Vec<(Instant, Instant)>> = Mutex::new(Vec::new());

    struct WalkSpanLayer;

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for WalkSpanLayer {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            id: &tracing::span::Id,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if attrs.metadata().name() == "TraceRecord::collect" {
                let mut open = OPEN_SPANS.lock().expect("span lock");
                open.push((id.into_u64(), Instant::now()));
            }
        }

        fn on_close(&self, id: tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
            let mut open = OPEN_SPANS.lock().expect("span lock");
            if let Some(index) = open
                .iter()
                .position(|(open_id, _)| *open_id == id.into_u64())
            {
                let (_, start) = open.swap_remove(index);
                WALK_SPANS
                    .lock()
                    .expect("span lock")
                    .push((start, Instant::now()));
            }
        }
    }

    static OPEN_SPANS: Mutex<Vec<(u64, Instant)>> = Mutex::new(Vec::new());

    /// The last drained walk span relative to `leg_start`.
    fn drain_walk_span(leg_start: Instant) -> (Option<f64>, Option<f64>) {
        let mut spans = WALK_SPANS.lock().expect("span lock");
        let span = spans.pop();
        spans.clear();
        span.map_or((None, None), |(start, end)| {
            (
                Some(start.duration_since(leg_start).as_secs_f64()),
                Some(end.duration_since(start).as_secs_f64()),
            )
        })
    }

    /// Touch-and-free ballast: depletes the free list the way a finished
    /// prove's transient allocations do.
    fn dirty_pages(gib: usize) {
        if gib == 0 {
            return;
        }
        let mut ballast = vec![0u8; gib << 30];
        ballast
            .chunks_mut(1 << 14)
            .for_each(|page| page[0] = page[0].wrapping_add(1));
        std::hint::black_box(&ballast);
    }

    fn rusage() -> (i64, i64, f64, f64) {
        let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        assert_eq!(rc, 0, "getrusage failed");
        let seconds = |tv: libc::timeval| tv.tv_sec as f64 + f64::from(tv.tv_usec) / 1_000_000.0;
        (
            usage.ru_minflt as i64,
            usage.ru_majflt as i64,
            seconds(usage.ru_utime),
            seconds(usage.ru_stime),
        )
    }

    struct LegSample {
        total_s: f64,
        commit_s: Option<f64>,
        join_s: Option<f64>,
        walk_start_s: Option<f64>,
        walk_dur_s: Option<f64>,
        minflt: i64,
        majflt: i64,
        utime_s: f64,
        stime_s: f64,
    }

    fn timed_leg(body: impl FnOnce() -> (Option<f64>, Option<f64>)) -> LegSample {
        let (minflt0, majflt0, utime0, stime0) = rusage();
        let start = Instant::now();
        let (commit_s, join_s) = body();
        let total_s = start.elapsed().as_secs_f64();
        let (walk_start_s, walk_dur_s) = drain_walk_span(start);
        let (minflt1, majflt1, utime1, stime1) = rusage();
        LegSample {
            total_s,
            commit_s,
            join_s,
            walk_start_s,
            walk_dur_s,
            minflt: minflt1 - minflt0,
            majflt: majflt1 - majflt0,
            utime_s: utime1 - utime0,
            stime_s: stime1 - stime0,
        }
    }

    fn emit(leg: &str, iter: usize, sample: &LegSample) {
        let optional =
            |value: Option<f64>| value.map_or("null".to_owned(), |seconds| format!("{seconds:.4}"));
        println!(
            "{{\"leg\":\"{leg}\",\"iter\":{iter},\"total_s\":{:.4},\"commit_s\":{},\"join_s\":{},\
             \"walk_start_s\":{},\"walk_dur_s\":{},\
             \"minflt\":{},\"majflt\":{},\"utime_s\":{:.3},\"stime_s\":{:.3}}}",
            sample.total_s,
            optional(sample.commit_s),
            optional(sample.join_s),
            optional(sample.walk_start_s),
            optional(sample.walk_dur_s),
            sample.minflt,
            sample.majflt,
            sample.utime_s,
            sample.stime_s,
        );
    }

    fn summarize(leg: &str, totals: &mut [f64]) {
        totals.sort_by(f64::total_cmp);
        let median = totals[totals.len() / 2];
        println!(
            "{{\"summary\":\"{leg}\",\"n\":{},\"min_s\":{:.4},\"median_s\":{median:.4},\"max_s\":{:.4}}}",
            totals.len(),
            totals[0],
            totals[totals.len() - 1],
        );
    }

    pub fn run() {
        use tracing_subscriber::layer::SubscriberExt as _;
        tracing::subscriber::set_global_default(tracing_subscriber::registry().with(WalkSpanLayer))
            .expect("set tracing subscriber");
        let cli = Cli::parse();
        let scale = cli.scale;
        let max_trace_length = 1usize << scale;
        let target = (max_trace_length as f64 * SAFETY_MARGIN) as usize;
        let input = [
            postcard::to_stdvec(&[5u8; 32]).expect("serialize input"),
            postcard::to_stdvec(&std::cmp::max(
                1,
                (target as f64 / CYCLES_PER_SHA256) as u32,
            ))
            .expect("serialize input"),
        ]
        .concat();

        // --- Untimed setup: guest compile/decode/trace + preprocessing,
        // exactly the modular_benchmark pipeline.
        let mut program = host::Program::new("sha2-chain-guest");
        let (bytecode, init_memory_state, _, entry_address) = program.decode();
        let (_, legacy_trace, _, io_device) = program.trace(&input, &[], &[]);
        assert!(
            legacy_trace.len().next_power_of_two() <= max_trace_length,
            "trace longer than 2^{scale}"
        );
        drop(legacy_trace);
        let elf_contents = program.get_elf_contents().expect("elf contents");
        let memory_layout = io_device.memory_layout.clone();

        let program_data =
            LegacyProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
                .expect("legacy preprocess");
        let shared_preprocessing =
            JoltSharedPreprocessing::new(program_data, memory_layout.clone(), max_trace_length);
        let legacy_preprocessing = LegacyProverPreprocessing::<
            jolt_prover_legacy::ark_bn254::Fr,
            Bn254Curve,
            DoryCommitmentScheme,
        >::new(shared_preprocessing);
        let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);
        let program_preprocessing = verifier_preprocessing
            .program
            .as_full()
            .expect("full program preprocessing")
            .clone();
        let jolt_program = JoltProgram::from_elf_bytes(elf_contents);

        let memory_config = MemoryConfig {
            max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
            max_trusted_advice_size: memory_layout.max_trusted_advice_size,
            max_input_size: memory_layout.max_input_size,
            max_output_size: memory_layout.max_output_size,
            stack_size: memory_layout.stack_size,
            heap_size: memory_layout.heap_size,
            program_size: Some(memory_layout.program_size),
        };
        let trace_output = TracerBackend::new()
            .trace(
                &jolt_program,
                TraceInputs {
                    inputs: input.clone(),
                    untrusted_advice: Vec::new(),
                    trusted_advice: Vec::new(),
                    memory_config,
                },
            )
            .expect("modular trace");
        let config = ProverConfig::derive::<Fr>(
            trace_output.trace.rows(),
            &memory_layout,
            verifier_preprocessing.program.min_bytecode_address(),
            verifier_preprocessing.program.program_image_len_words(),
            max_trace_length,
        )
        .expect("derive config");
        let log_t = config.trace_length.ilog2() as usize;
        let mut rows = trace_output.trace.rows().to_vec();
        rows.resize(config.trace_length, TraceRow::default());
        let padded_output = TraceOutput::new(
            OwnedTrace::new(rows),
            trace_output.device.clone(),
            trace_output.final_memory,
        );
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
        );

        let advice_vars = |max_advice_size_bytes: u64| -> usize {
            ((max_advice_size_bytes / 8) as usize)
                .next_power_of_two()
                .max(1)
                .ilog2() as usize
        };
        let total_vars = (config.one_hot_config.committed_chunk_bits() + log_t)
            .max(advice_vars(memory_layout.max_trusted_advice_size))
            .max(advice_vars(memory_layout.max_untrusted_advice_size));
        let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
            verifier: verifier_preprocessing,
            pcs_setup: DoryScheme::setup_prover(total_vars),
            committed_program: None,
        };
        let backend = JoltBackend::<Fr, DoryScheme>::metal().expect("metal backend");

        // Production stage-0 grid and id set (no advice, no committed
        // program for this guest).
        let ids: Vec<JoltCommittedPolynomial> = JoltWitnessOracle::<Fr>::committed_order(&witness)
            .expect("committed order")
            .into_iter()
            .filter(|id| {
                !matches!(
                    id,
                    JoltCommittedPolynomial::TrustedAdvice
                        | JoltCommittedPolynomial::UntrustedAdvice
                )
            })
            .collect();
        let grid = CommitmentGrid {
            total_vars: config.commitment_total_vars(&memory_layout, false, false, None),
            log_t,
            log_k_chunk: config.one_hot_config.committed_chunk_bits(),
            order: config.trace_polynomial_order,
        };

        let knob = |name: &str| std::env::var(name).unwrap_or_else(|_| "-".to_owned());
        println!(
            "{{\"config\":{{\"scale\":{scale},\"log_t\":{log_t},\"columns\":{},\"total_vars\":{},\
             \"record_threads\":\"{}\",\"record_delay_ms\":\"{}\",\"record_qos\":\"{}\",\
             \"soak_threads\":{}}}}}",
            ids.len(),
            grid.total_vars,
            knob("JOLT_RECORD_BACKGROUND_THREADS"),
            knob("JOLT_RECORD_HOIST_DELAY_MS"),
            knob("JOLT_RECORD_QOS"),
            cli.soak_threads,
        );

        let commit_once = || {
            let mut session = backend.begin_proof();
            let committed = backend
                .commit
                .commit_witness(
                    &mut session,
                    &witness as &dyn RowSource,
                    &ids,
                    grid,
                    &prover_preprocessing.pcs_setup,
                )
                .expect("commit");
            assert_eq!(committed.len(), ids.len());
        };

        let legs: Vec<&str> = cli.legs.split(',').collect();
        let parse_g1_leg = |leg: &str| {
            leg.strip_prefix("g1-")
                .map(|cap| (cap, false))
                .or_else(|| leg.strip_prefix("g1s-").map(|cap| (cap, true)))
                .and_then(|(cap, serial)| cap.parse().ok().map(|cap| (cap, serial)))
        };
        let mut g1_caps: Vec<usize> = legs
            .iter()
            .filter_map(|leg| parse_g1_leg(leg).map(|(cap, _)| cap))
            .collect();
        g1_caps.sort_unstable();
        g1_caps.dedup();
        let g1_fixture = (!g1_caps.is_empty()).then(|| {
            G1SegBenchFixture::new(
                &witness as &dyn RowSource,
                &ids,
                grid,
                &prover_preprocessing.pcs_setup,
            )
            .expect("G1 segment fixture")
        });
        let g1_cases: Vec<(usize, G1SegBenchCase)> =
            g1_fixture.as_ref().map_or_else(Vec::new, |f| {
                g1_caps
                    .iter()
                    .map(|&cap| (cap, f.build_case(cap).expect("G1 segment case")))
                    .collect()
            });
        let _gpu_lock = gpu_lock();
        if let Some(fixture) = &g1_fixture {
            let cases: Vec<&G1SegBenchCase> = g1_cases.iter().map(|(_, case)| case).collect();
            fixture
                .assert_equivalent(&cases)
                .expect("segment-cap tier-1 oracle");
        }

        for leg in legs {
            let mut totals = Vec::with_capacity(cli.iters);
            for iter in 0..cli.iters {
                dirty_pages(cli.dirty_gb);
                if let Some((cap, serial)) = parse_g1_leg(leg) {
                    let fixture = g1_fixture.as_ref().expect("G1 fixture");
                    let case = g1_cases
                        .iter()
                        .find(|(candidate, _)| *candidate == cap)
                        .map(|(_, case)| case)
                        .expect("requested G1 case");
                    if serial {
                        std::env::set_var("JOLT_METAL_G1_SEG_SERIAL", "1");
                    }
                    let sample = case.sample(fixture).expect("G1 segment dispatch");
                    std::env::remove_var("JOLT_METAL_G1_SEG_SERIAL");
                    println!(
                        "{{\"leg\":\"{leg}\",\"iter\":{iter},\"gpu_s\":{:.6},\
                         \"wall_s\":{:.6},\"segments\":{},\"additions\":{},\
                         \"useful_gbps\":{:.3},\"gmul_s\":{:.3}}}",
                        sample.gpu_s,
                        sample.wall_s,
                        sample.segments,
                        sample.additions,
                        sample.useful_gbps,
                        sample.gmul_s,
                    );
                    totals.push(sample.gpu_s);
                    continue;
                }
                let sample = match leg {
                    "commit" => timed_leg(|| {
                        commit_once();
                        (None, None)
                    }),
                    _ if leg
                        .strip_prefix("commit-s")
                        .and_then(|cap| cap.parse::<usize>().ok())
                        .is_some() =>
                    {
                        let cap = leg
                            .strip_prefix("commit-s")
                            .and_then(|cap| cap.parse::<usize>().ok())
                            .expect("checked segment cap");
                        timed_leg(|| {
                            std::env::set_var("JOLT_METAL_G1_SEGMENT_LEN", cap.to_string());
                            std::env::set_var("JOLT_METAL_G1_SEG_SERIAL", "1");
                            commit_once();
                            std::env::remove_var("JOLT_METAL_G1_SEG_SERIAL");
                            std::env::remove_var("JOLT_METAL_G1_SEGMENT_LEN");
                            (None, None)
                        })
                    }
                    _ if leg
                        .strip_prefix("commit-")
                        .and_then(|cap| cap.parse::<usize>().ok())
                        .is_some() =>
                    {
                        let cap = leg
                            .strip_prefix("commit-")
                            .and_then(|cap| cap.parse::<usize>().ok())
                            .expect("checked segment cap");
                        timed_leg(|| {
                            std::env::set_var("JOLT_METAL_G1_SEGMENT_LEN", cap.to_string());
                            commit_once();
                            std::env::remove_var("JOLT_METAL_G1_SEGMENT_LEN");
                            (None, None)
                        })
                    }
                    "walk" => timed_leg(|| {
                        let mut session = backend.begin_proof();
                        std::thread::scope(|scope| {
                            spawn_shared_record_collect::<Fr>(
                                &mut session,
                                &witness as &dyn JoltWitnessPlane<Fr>,
                                log_t,
                                scope,
                            );
                            join_shared_record_for_bench::<Fr>(&mut session, &witness, log_t)
                                .expect("record walk");
                        });
                        (None, None)
                    }),
                    // `corun-bg12`: the E-cluster arm — background QoS at
                    // 12 threads (the walk's package draw floor), interleaved
                    // A/B-safe because the spawn reads the knobs per call.
                    "corun-bg12" => {
                        std::env::set_var("JOLT_RECORD_QOS", "background");
                        std::env::set_var("JOLT_RECORD_BACKGROUND_THREADS", "12");
                        let sample = timed_leg(|| {
                            let mut session = backend.begin_proof();
                            std::thread::scope(|scope| {
                                spawn_shared_record_collect::<Fr>(
                                    &mut session,
                                    &witness as &dyn JoltWitnessPlane<Fr>,
                                    log_t,
                                    scope,
                                );
                                let commit_start = Instant::now();
                                let committed = backend
                                    .commit
                                    .commit_witness(
                                        &mut session,
                                        &witness as &dyn RowSource,
                                        &ids,
                                        grid,
                                        &prover_preprocessing.pcs_setup,
                                    )
                                    .expect("commit");
                                assert_eq!(committed.len(), ids.len());
                                let commit_s = commit_start.elapsed().as_secs_f64();
                                let join_start = Instant::now();
                                join_shared_record_for_bench::<Fr>(&mut session, &witness, log_t)
                                    .expect("record walk");
                                (Some(commit_s), Some(join_start.elapsed().as_secs_f64()))
                            })
                        });
                        std::env::remove_var("JOLT_RECORD_QOS");
                        std::env::remove_var("JOLT_RECORD_BACKGROUND_THREADS");
                        sample
                    }
                    "corun" => timed_leg(|| {
                        let mut session = backend.begin_proof();
                        std::thread::scope(|scope| {
                            spawn_shared_record_collect::<Fr>(
                                &mut session,
                                &witness as &dyn JoltWitnessPlane<Fr>,
                                log_t,
                                scope,
                            );
                            let commit_start = Instant::now();
                            let committed = backend
                                .commit
                                .commit_witness(
                                    &mut session,
                                    &witness as &dyn RowSource,
                                    &ids,
                                    grid,
                                    &prover_preprocessing.pcs_setup,
                                )
                                .expect("commit");
                            assert_eq!(committed.len(), ids.len());
                            let commit_s = commit_start.elapsed().as_secs_f64();
                            let join_start = Instant::now();
                            join_shared_record_for_bench::<Fr>(&mut session, &witness, log_t)
                                .expect("record walk");
                            (Some(commit_s), Some(join_start.elapsed().as_secs_f64()))
                        })
                    }),
                    "soak" | "soak-stream" | "soak-fault" => timed_leg(|| {
                        let stop = AtomicBool::new(false);
                        let fault_mode =
                            leg == "soak-fault" || (leg == "soak" && cli.soak_mode == "fault");
                        let commit_s = std::thread::scope(|scope| {
                            for _ in 0..cli.soak_threads {
                                scope.spawn(|| {
                                    if fault_mode {
                                        const REGION: usize = 64 << 20;
                                        while !stop.load(Ordering::Relaxed) {
                                            let raw = unsafe {
                                                libc::mmap(
                                                    std::ptr::null_mut(),
                                                    REGION,
                                                    libc::PROT_READ | libc::PROT_WRITE,
                                                    libc::MAP_ANON | libc::MAP_PRIVATE,
                                                    -1,
                                                    0,
                                                )
                                            };
                                            assert_ne!(raw, libc::MAP_FAILED, "soak mmap");
                                            let region = raw.cast::<u8>();
                                            for offset in (0..REGION).step_by(1 << 14) {
                                                unsafe { region.add(offset).write(1) };
                                            }
                                            unsafe { libc::munmap(raw, REGION) };
                                        }
                                    } else {
                                        let mut buffer = vec![0u8; 256 << 20];
                                        let mut tick = 0u8;
                                        while !stop.load(Ordering::Relaxed) {
                                            tick = tick.wrapping_add(1);
                                            buffer.fill(tick);
                                            std::hint::black_box(buffer[0]);
                                        }
                                    }
                                });
                            }
                            let commit_start = Instant::now();
                            commit_once();
                            let commit_s = commit_start.elapsed().as_secs_f64();
                            stop.store(true, Ordering::Relaxed);
                            commit_s
                        });
                        (Some(commit_s), None)
                    }),
                    other => panic!("unknown leg {other:?}"),
                };
                emit(leg, iter, &sample);
                totals.push(sample.total_s);
            }
            summarize(leg, &mut totals);
        }
    }
}
