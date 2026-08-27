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
//! - `g1x`     — the tier-1 `jk_g1_seg_sum` attribution matrix (wave 9):
//!   TG-width sweep, fixed-base/gather-only/pipelined variant kernels, and
//!   mul-chain roofs at production and saturated thread counts.
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
            .as_full_arc()
            .expect("full program preprocessing");
        let jolt_program = std::sync::Arc::new(JoltProgram::from_elf_bytes(elf_contents));

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
                    advice_tape: None,
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
            trace_output.advice_tape,
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
        if legs.contains(&"g1x") {
            g1_caps.push(256);
        }
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
            if leg == "g1x" {
                let fixture = g1_fixture.as_ref().expect("g1x fixture");
                let case = g1_cases
                    .iter()
                    .find(|(cap, _)| *cap == 256)
                    .map(|(_, case)| case)
                    .expect("g1x case");
                g1_attribution(fixture, case, cli.iters.max(3));
                continue;
            }
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

    /// Bench-only variant kernels for the wave-9 `jk_g1_seg_sum` gap
    /// attribution. Appended to the production library source, so all
    /// g1.metal helpers are in scope.
    const XV_SRC: &str = r#"
// Fq Montgomery-mul throughput, one dependent chain per thread.
// params: [n_threads, muls_per_thread, n_bases (pow2)].
kernel void xv_mulroof(
    device const uint* bases [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant uint* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params[0]) { return; }
    G1AffinePt p = g1_load_base(bases, tid & (params[2] - 1u));
    Fq256 x = p.x;
    Fq256 y = p.y;
    for (uint i = 0; i < params[1]; i++) {
        x = fq_mul(x, y);
    }
    for (uint i = 0; i < FR_LIMBS; i++) { out[(tid & 8191u) * FR_LIMBS + i] = x.v[i]; }
}

// Same, four independent chains per thread (ILP probe); params[1] is the
// TOTAL muls per thread (split across chains).
kernel void xv_mulroof4(
    device const uint* bases [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant uint* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params[0]) { return; }
    G1AffinePt p = g1_load_base(bases, tid & (params[2] - 1u));
    Fq256 x0 = p.x;
    Fq256 x1 = p.y;
    Fq256 x2 = fq_add(p.x, p.y);
    Fq256 x3 = fq_sub(p.x, p.y);
    Fq256 y = p.y;
    uint iters = params[1] / 4u;
    for (uint i = 0; i < iters; i++) {
        x0 = fq_mul(x0, y);
        x1 = fq_mul(x1, y);
        x2 = fq_mul(x2, y);
        x3 = fq_mul(x3, y);
    }
    x0 = fq_add(fq_add(x0, x1), fq_add(x2, x3));
    for (uint i = 0; i < FR_LIMBS; i++) { out[(tid & 8191u) * FR_LIMBS + i] = x0.v[i]; }
}

// Production segment loop with the gather removed: one base, same adds.
kernel void xv_fixedbase(
    device const uint* bases [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device const uint* seg_bounds [[buffer(2)]],
    device uint* out [[buffer(3)]],
    constant uint* params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint n_segs = params[0];
    if (tid >= n_segs) { return; }
    uint start = seg_bounds[3u * tid];
    uint end = seg_bounds[3u * tid + 1u];
    G1Xyzz acc = g1_xyzz_identity();
    if (start < end) {
        G1AffinePt q = g1_load_base(bases, indices[start] & 0x7fffffffu);
        for (uint i = start; i < end; i++) {
            acc = g1_xyzz_madd(acc, q);
        }
    }
    g1_store_jac(out + tid * (3u * FR_LIMBS), g1_xyzz_to_jac(acc));
}

// Production gather with the EC math removed: XOR-fold the loaded words.
kernel void xv_loadonly(
    device const uint* bases [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device const uint* seg_bounds [[buffer(2)]],
    device uint* out [[buffer(3)]],
    constant uint* params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint n_segs = params[0];
    if (tid >= n_segs) { return; }
    uint start = seg_bounds[3u * tid];
    uint end = seg_bounds[3u * tid + 1u];
    Fq256 sx = fq_zero();
    Fq256 sy = fq_zero();
    for (uint i = start; i < end; i++) {
        uint raw = indices[i];
        G1AffinePt q = g1_load_base(bases, raw & 0x7fffffffu);
        for (uint j = 0; j < FR_LIMBS; j++) {
            sx.v[j] ^= q.x.v[j];
            sy.v[j] ^= q.y.v[j];
        }
    }
    for (uint j = 0; j < FR_LIMBS; j++) {
        out[tid * (3u * FR_LIMBS) + j] = sx.v[j];
        out[tid * (3u * FR_LIMBS) + FR_LIMBS + j] = sy.v[j];
    }
}

// Production loop, software-pipelined: next base load issued before the
// current madd.
kernel void xv_pipelined(
    device const uint* bases [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device const uint* seg_bounds [[buffer(2)]],
    device uint* out [[buffer(3)]],
    constant uint* params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint n_segs = params[0];
    if (tid >= n_segs) { return; }
    uint start = seg_bounds[3u * tid];
    uint end = seg_bounds[3u * tid + 1u];
    G1Xyzz acc = g1_xyzz_identity();
    if (start < end) {
        uint raw = indices[start];
        G1AffinePt q = g1_load_base(bases, raw & 0x7fffffffu);
        for (uint i = start + 1u; i < end; i++) {
            uint nraw = indices[i];
            G1AffinePt nq = g1_load_base(bases, nraw & 0x7fffffffu);
            if (raw >> 31) { q.y = fq_sub(fq_zero(), q.y); }
            acc = g1_xyzz_madd(acc, q);
            q = nq;
            raw = nraw;
        }
        if (raw >> 31) { q.y = fq_sub(fq_zero(), q.y); }
        acc = g1_xyzz_madd(acc, q);
    }
    g1_store_jac(out + tid * (3u * FR_LIMBS), g1_xyzz_to_jac(acc));
}
"#;

    fn g1_attribution(fixture: &G1SegBenchFixture, case: &G1SegBenchCase, iters: usize) {
        use jolt_kernels::metal::{KernelId, MetalContext, JAC_U32S};
        let ctx = MetalContext::global().expect("metal context");
        let (bases_words, row_width) = fixture.bases_words();
        let (indices, bounds, out) = case.raw_parts();
        let segments = bounds.len() / 3;
        let additions: usize = bounds.chunks_exact(3).map(|b| (b[1] - b[0]) as usize).sum();
        let (max_tg, exec_w) = ctx.pipeline_stats(KernelId::G1SegSum);

        // Divergence model: simdgroup utilization in the (sorted) dispatch
        // order vs the original bucket-walk order.
        let lens: Vec<u32> = bounds.chunks_exact(3).map(|b| b[1] - b[0]).collect();
        let simd_util = |lens: &[u32]| -> f64 {
            let (mut work, mut occ) = (0u64, 0u64);
            for group in lens.chunks(32) {
                let max = u64::from(*group.iter().max().unwrap_or(&0));
                work += group.iter().map(|&l| u64::from(l)).sum::<u64>();
                occ += max * 32;
            }
            work as f64 / occ as f64
        };
        // The original bucket-walk order, reconstructed from out slots.
        let mut unsorted_bounds = vec![0u32; bounds.len()];
        for b in bounds.chunks_exact(3) {
            let slot = b[2] as usize;
            unsorted_bounds[3 * slot..3 * slot + 3].copy_from_slice(b);
        }
        let unsorted_lens: Vec<u32> = unsorted_bounds
            .chunks_exact(3)
            .map(|b| b[1] - b[0])
            .collect();
        let full = lens.iter().filter(|&&l| l as usize == 256).count();
        println!(
            "{{\"g1x\":\"shape\",\"segments\":{segments},\"additions\":{additions},\
             \"row_width\":{row_width},\"bases_mib\":{:.2},\"full_segs\":{full},\
             \"simd_util_unsorted\":{:.4},\"simd_util_sorted\":{:.4},\
             \"max_tg\":{max_tg},\"exec_w\":{exec_w}}}",
            (bases_words.len() * 4) as f64 / (1 << 20) as f64,
            simd_util(&unsorted_lens),
            simd_util(&lens),
        );

        let bases_buf = ctx.wrap_slice(bases_words).expect("bases");
        let indices_buf = ctx.wrap_slice(indices).expect("indices");
        let starts_buf = ctx.wrap_slice(bounds).expect("bounds");
        let unsorted_bounds_buf = ctx.wrap_slice(&unsorted_bounds).expect("unsorted bounds");
        let out_buf = out.device_buffer();

        macro_rules! run {
            ($label:expr, $muls_per_add:expr, |$pass:ident| $body:block) => {{
                let mut times = Vec::with_capacity(iters);
                let mut total_muls = 0usize;
                for _ in 0..iters {
                    let mut $pass = ctx.begin_pass().expect("pass");
                    total_muls = $body;
                    times.push($pass.commit().wait_timed().expect("wait").as_secs_f64());
                }
                times.sort_by(f64::total_cmp);
                let median = times[times.len() / 2];
                let muls_per_add: f64 = $muls_per_add;
                let muls = if muls_per_add > 0.0 {
                    additions as f64 * muls_per_add
                } else {
                    total_muls as f64
                };
                println!(
                    "{{\"g1x\":\"{}\",\"n\":{iters},\"min_s\":{:.6},\"median_s\":{median:.6},\
                     \"gmul_s\":{:.3},\"gbps_at_68b\":{:.2}}}",
                    $label,
                    times[0],
                    muls / median / 1e9,
                    additions as f64 * 68.0 / median / 1e9,
                );
            }};
        }

        // Real kernel at threadgroup widths (occupancy shaping).
        for width in [32usize, 64, 128, 256, 512, 1024] {
            if width > max_tg {
                continue;
            }
            let label = format!("real_w{width}");
            run!(&label, 10.0, |pass| {
                pass.dispatch_width(
                    KernelId::G1SegSum,
                    &[segments as u32],
                    &[&bases_buf, &indices_buf, &starts_buf, &out_buf],
                    segments,
                    width,
                );
                0
            });
        }

        // Variant kernels on the production case.
        for (entry, muls_per_add) in [
            ("xv_fixedbase", 10.0),
            ("xv_pipelined", 10.0),
            ("xv_loadonly", 0.0),
        ] {
            let variant = ctx.compile_variant(XV_SRC, entry).expect("variant");
            let (vmax, vw) = variant.stats();
            println!("{{\"g1x\":\"{entry}_stats\",\"max_tg\":{vmax},\"exec_w\":{vw}}}");
            run!(entry, muls_per_add, |pass| {
                pass.dispatch_variant(
                    &variant,
                    &[segments as u32],
                    &[&bases_buf, &indices_buf, &starts_buf, &out_buf],
                    segments,
                    256,
                );
                0
            });
        }

        // Roof reprice: mul-chain throughput at the production thread count
        // and at saturation, ILP 1 and 4.
        let avg_muls = (additions as f64 / segments as f64 * 10.0).round() as u32;
        let sat_threads = 1usize << 19;
        let sat_out = ctx.alloc_u32s(8192 * 8).expect("sat out");
        for (entry, threads, muls_per_thread) in [
            ("xv_mulroof", segments, avg_muls),
            ("xv_mulroof4", segments, avg_muls),
            ("xv_mulroof", sat_threads, 512u32),
            ("xv_mulroof4", sat_threads, 512u32),
        ] {
            let variant = ctx.compile_variant(XV_SRC, entry).expect("variant");
            let (vmax, vw) = variant.stats();
            let label = format!("{entry}_t{threads}");
            println!("{{\"g1x\":\"{label}_stats\",\"max_tg\":{vmax},\"exec_w\":{vw}}}");
            run!(&label, 0.0, |pass| {
                pass.dispatch_variant(
                    &variant,
                    &[threads as u32, muls_per_thread, row_width as u32],
                    &[&bases_buf, &sat_out],
                    threads,
                    256,
                );
                let chains = if entry.ends_with('4') { 4 } else { 1 };
                threads * (muls_per_thread / chains * chains) as usize
            });
        }

        // Unsorted (bucket-walk) dispatch order — the sort ablation.
        for width in [64usize, 256] {
            let label = format!("real_unsorted_w{width}");
            run!(&label, 10.0, |pass| {
                pass.dispatch_width(
                    KernelId::G1SegSum,
                    &[segments as u32],
                    &[&bases_buf, &indices_buf, &unsorted_bounds_buf, &out_buf],
                    segments,
                    width,
                );
                0
            });
        }

        // Base-footprint spread: tile the base array, offset each segment
        // into its own tile — 2^27-like gather footprint at 2^24 shape.
        for tiles in [4usize, 16] {
            let mut spread_bases = Vec::with_capacity(bases_words.len() * tiles);
            for _ in 0..tiles {
                spread_bases.extend_from_slice(bases_words);
            }
            let mut spread_indices = indices.to_vec();
            for (s, b) in bounds.chunks_exact(3).enumerate() {
                let tile = ((s as u32).wrapping_mul(2654435761) as usize) & (tiles - 1);
                let offset = (tile * row_width) as u32;
                for raw in &mut spread_indices[b[0] as usize..b[1] as usize] {
                    let sign = *raw & 0x8000_0000;
                    *raw = ((*raw & 0x7fff_ffff) + offset) | sign;
                }
            }
            let spread_bases_buf = ctx.wrap_slice(&spread_bases).expect("spread bases");
            let spread_indices_buf = ctx.wrap_slice(&spread_indices).expect("spread idx");
            let label = format!("real_spread{tiles}");
            run!(&label, 10.0, |pass| {
                pass.dispatch(
                    KernelId::G1SegSum,
                    &[segments as u32],
                    &[
                        &spread_bases_buf,
                        &spread_indices_buf,
                        &starts_buf,
                        &out_buf,
                    ],
                    segments,
                );
                0
            });
        }
        let _ = JAC_U32S;
    }
}
