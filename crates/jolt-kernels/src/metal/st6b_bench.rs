//! W2-st6b lane harness: the stage-6b lazy-RA device round pipelines in
//! isolation — lazy gathers, the dense adoption, and the fused dense rounds
//! — at production geometry, without an end-to-end prove run.
//!
//! Two slots, driven by the `st6b_rav_microbench` example:
//! - **RAV**: instruction RA virtualization (16 committed = 4 virtual × 4
//!   per-virtual, 8-bit chunks of the 128-bit lookup index), through the
//!   REAL `OptimizedInstructionRaVirtualizationKernel` with the slot's
//!   device driver installed.
//! - **Bool**: the booleanity cycle phase (20 polys = 16 instruction + 2
//!   bytecode + 2 RAM chunk selectors, 8-bit chunks), through the REAL
//!   optimized cycle kernel built from raw parts
//!   ([`crate::optimized::booleanity::booleanity_cycle_kernel_for_bench`]).
//!
//! Both advance under the batch engine's two-phase contract (`begin_round`
//! for every member, then `collect_round`), so `begin_s` is exactly the
//! phase-1 serialization a production round pays before any synchronous CPU
//! member can start.
//!
//! Arms (same binary, the family's `JOLT_*_DEFERRED_ADOPT` knob flipped per
//! kernel build):
//! - `sync`: legacy — third bind materializes at `cycles / 8` inside
//!   `begin_round` (blocking `jk_ra_materialize` + wait), the round message
//!   then re-reads the fresh dense tables.
//! - `deferred`: a fourth lazy round at width 8, then ONE detached fused
//!   adopt-round (`jk_rav_adopt_round` / `jk_bool_adopt_round`)
//!   materializes at `cycles / 16` fused with that round's message.
//!
//! Correctness oracle: a driverless CPU twin runs the same challenge/claim
//! stream; every wire round polynomial and the final output claims must be
//! byte-equal (the exact parity contract the slot tests pin).

use std::sync::Arc;
use std::time::Instant;

use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::relations::instruction::InstructionRaVirtualizationInputClaims;
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_sumcheck::ProveRounds;
use jolt_verifier::stages::relations::SumcheckOutputClaims;
use jolt_verifier::stages::stage6b::booleanity::{Booleanity, BooleanityInputClaims};

use crate::optimized::booleanity::booleanity_cycle_kernel_for_bench;
use crate::optimized::instruction_ra_virtualization::OptimizedInstructionRaVirtualizationKernel;
use crate::optimized::instruction_read_raf::{InstructionCycleRow, InstructionRows};
use crate::SumcheckKernel;

const FR_BYTES: usize = 32;

/// Production stage-6b instruction-RAV geometry at `2^log_t` cycles.
#[derive(Clone, Copy)]
pub struct RavBenchConfig {
    pub log_t: usize,
    pub num_virtual: usize,
    pub per_virtual: usize,
    pub chunk_bits: usize,
    pub seed: u64,
}

impl RavBenchConfig {
    pub fn production(log_t: usize) -> Self {
        Self {
            log_t,
            num_virtual: 4,
            per_virtual: 4,
            chunk_bits: 8,
            seed: 0x57A6_E6B5 ^ log_t as u64,
        }
    }

    fn num_committed(&self) -> usize {
        self.num_virtual * self.per_virtual
    }
}

/// Production stage-6b booleanity-cycle geometry at `2^log_t` cycles: the
/// layout split observed in the campaign CB audit (20-poly adoption =
/// 16 instruction + 2 bytecode + 2 RAM chunk selectors at 8-bit chunks).
#[derive(Clone, Copy)]
pub struct BoolBenchConfig {
    pub log_t: usize,
    pub instruction: usize,
    pub bytecode: usize,
    pub ram: usize,
    pub log_k_chunk: usize,
    pub seed: u64,
}

impl BoolBenchConfig {
    pub fn production(log_t: usize) -> Self {
        Self {
            log_t,
            instruction: 16,
            bytecode: 2,
            ram: 2,
            log_k_chunk: 8,
            seed: 0xB001_EA17 ^ log_t as u64,
        }
    }

    fn num_polys(&self) -> usize {
        self.instruction + self.bytecode + self.ram
    }
}

/// One timed pipeline pass.
pub struct PassTiming {
    /// Per round: (`begin_round` seconds, `collect_round` seconds).
    pub rounds: Vec<(f64, f64)>,
    /// Whether `begin_round` launched a device command buffer for each round.
    pub launched: Vec<bool>,
    /// The first round served from fresh dense state: 3 for the legacy arm,
    /// 4 for the deferred arm.
    pub adopt_round: usize,
    /// The adoption ping-pong's fresh allocation (`cur` + `nxt`), exact
    /// bytes from the arm's geometry — the tail mode's allocation fuel.
    pub adopt_alloc_bytes: usize,
    /// Wall of the full pass: all rounds + `finish_rounds` + output claims.
    pub total_s: f64,
}

impl PassTiming {
    /// The adoption round's `begin_round` wall — the phase-1 stall the
    /// batch engine serializes across every member before synchronous CPU
    /// members may run.
    pub fn adopt_begin_s(&self) -> f64 {
        self.rounds[self.adopt_round].0
    }

    /// The adoption round's full wall (begin + collect).
    pub fn adopt_span_s(&self) -> f64 {
        let (begin, collect) = self.rounds[self.adopt_round];
        begin + collect
    }

    /// Sum of every round's `begin_round` wall.
    pub fn begin_serial_s(&self) -> f64 {
        self.rounds.iter().map(|(begin, _)| begin).sum()
    }

    pub fn launched_wait_s(&self) -> f64 {
        self.rounds
            .iter()
            .zip(&self.launched)
            .filter(|(_, launched)| **launched)
            .map(|((_, collect), _)| collect)
            .sum()
    }

    pub fn lazy_span_s(&self) -> f64 {
        self.rounds[..self.adopt_round]
            .iter()
            .map(|(begin, collect)| begin + collect)
            .sum()
    }

    pub fn dense_device_span_s(&self) -> f64 {
        self.rounds
            .iter()
            .zip(&self.launched)
            .skip(self.adopt_round + 1)
            .filter(|(_, launched)| **launched)
            .map(|((begin, collect), _)| begin + collect)
            .sum()
    }

    pub fn host_tail_span_s(&self) -> f64 {
        self.rounds
            .iter()
            .zip(&self.launched)
            .skip(self.adopt_round + 1)
            .filter(|(_, launched)| !**launched)
            .map(|((begin, collect), _)| begin + collect)
            .sum()
    }
}

/// The adoption ping-pong bytes (`cur` at the arm's dense length plus the
/// half-size `nxt`) for `num_polys` at `2^log_t` cycles.
fn adopt_alloc_bytes(log_t: usize, num_polys: usize, deferred: bool) -> usize {
    let new_len = (1usize << log_t) / if deferred { 16 } else { 8 };
    num_polys * new_len * FR_BYTES * 3 / 2
}

/// Deterministic bench challenge stream (the slot tests' constants).
fn challenge(round: usize) -> Fr {
    Fr::from_u64(0xC0FF_EE11_D00D_F00D ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x2A)
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn point(seed: u64, len: usize) -> Vec<Fr> {
    (0..len)
        .map(|i| Fr::from_u64(seed + 17 * i as u64))
        .collect()
}

/// The shared packed cycle rows (opaque: the row type is crate-internal).
pub struct BenchRows {
    rows: Arc<InstructionRows>,
}

/// Uniform-random lookup indices, cold PC/RAM columns: the RAV gather
/// kernels' data-oblivious (and cache-conservative) case.
pub fn bench_rows(config: &RavBenchConfig) -> BenchRows {
    synth_rows(config.log_t, config.seed, false)
}

/// Booleanity-flavored rows: every cycle's PC hot (production traces
/// execute an instruction every cycle) and ~three-eighths of RAM addresses
/// hot — the sentinel gathers pay their production mix.
pub fn bool_bench_rows(config: &BoolBenchConfig) -> BenchRows {
    synth_rows(config.log_t, config.seed, true)
}

fn synth_rows(log_t: usize, seed: u64, hot_columns: bool) -> BenchRows {
    let mut state = seed;
    BenchRows {
        rows: Arc::new(InstructionRows::new(
            (0..1usize << log_t)
                .map(|j| {
                    let index = match j {
                        0 => 0u128,
                        1 => u128::MAX,
                        _ => ((splitmix(&mut state) as u128) << 64) | splitmix(&mut state) as u128,
                    };
                    let (pc, ram) = if hot_columns {
                        let word = splitmix(&mut state);
                        let pc = Some((word & 0xFFFF) as usize);
                        let ram = (word & 0b111 < 3).then_some((word >> 16) & 0xFFFF);
                        (pc, ram)
                    } else {
                        (None, None)
                    };
                    InstructionCycleRow::new(index, None, false, pc, ram)
                })
                .collect(),
        )),
    }
}

fn build_rav_kernel(
    config: &RavBenchConfig,
    rows: &Arc<InstructionRows>,
    with_driver: bool,
) -> OptimizedInstructionRaVirtualizationKernel<Fr> {
    let instruction_address = point(300, config.num_committed() * config.chunk_bits);
    let r_cycle = point(7000, config.log_t);
    let driver = with_driver
        .then(|| {
            super::slots::rav_driver_for_bench(
                rows,
                config.num_committed(),
                config.per_virtual,
                config.chunk_bits,
            )
        })
        .flatten();
    assert!(
        !with_driver || driver.is_some(),
        "device driver install declined (gate/env)"
    );
    #[expect(clippy::unwrap_used, reason = "bench harness: fail loudly")]
    OptimizedInstructionRaVirtualizationKernel::new_with_driver(
        config.log_t,
        config.num_virtual,
        config.per_virtual,
        &instruction_address,
        &r_cycle,
        config.chunk_bits,
        Arc::clone(rows),
        Fr::from_u64(0xFEED_5EED),
        driver,
    )
    .unwrap()
}

type BoolKernel = Box<dyn SumcheckKernel<Fr, Relation = Booleanity<Fr>>>;

fn build_bool_kernel(
    config: &BoolBenchConfig,
    rows: &Arc<InstructionRows>,
    with_driver: bool,
) -> BoolKernel {
    #[expect(clippy::unwrap_used, reason = "bench harness: fail loudly")]
    let layout =
        JoltRaPolynomialLayout::new(config.instruction, config.bytecode, config.ram).unwrap();
    let r_address = point(110, config.log_k_chunk);
    let reference_address = point(700, config.log_k_chunk);
    let reference_cycle = point(400, config.log_t);
    #[expect(clippy::unwrap_used, reason = "bench harness: fail loudly")]
    booleanity_cycle_kernel_for_bench(
        Arc::clone(rows),
        layout,
        config.log_k_chunk,
        &r_address,
        &reference_address,
        &reference_cycle,
        Fr::from_u64(31),
        |inputs| {
            if !with_driver {
                return None;
            }
            let driver = super::slots::bool_driver_for_bench(&inputs);
            assert!(
                driver.is_some(),
                "device driver install declined (gate/env)"
            );
            driver
        },
    )
    .unwrap()
}

/// Drive one kernel through the engine's two-phase contract, timing each
/// round; returns the wire round polynomials alongside the timings so the
/// caller can pin parity.
fn drive_rounds(
    kernel: &mut dyn ProveRounds<Fr>,
    log_t: usize,
    adopt_round: usize,
    adopt_alloc: usize,
) -> (PassTiming, Vec<Vec<Fr>>) {
    let started = Instant::now();
    let mut claim = Fr::from_u64(0xBEEF);
    let mut rounds = Vec::with_capacity(log_t);
    let mut launched = Vec::with_capacity(log_t);
    let mut polys = Vec::with_capacity(log_t);
    for round in 0..log_t {
        let bind = round.checked_sub(1).map(challenge);
        let begin_at = Instant::now();
        #[expect(clippy::unwrap_used, reason = "bench harness: fail loudly")]
        let device_launched = kernel.begin_round(bind, round, claim).unwrap();
        let begin_s = begin_at.elapsed().as_secs_f64();
        let collect_at = Instant::now();
        #[expect(clippy::unwrap_used, reason = "bench harness: fail loudly")]
        let poly = kernel.collect_round(bind, round, claim).unwrap();
        let collect_s = collect_at.elapsed().as_secs_f64();
        rounds.push((begin_s, collect_s));
        launched.push(device_launched);
        claim = poly.evaluate(challenge(round));
        polys.push(poly.coefficients().to_vec());
    }
    #[expect(clippy::unwrap_used, reason = "bench harness: fail loudly")]
    kernel.finish_rounds(challenge(log_t - 1)).unwrap();
    (
        PassTiming {
            rounds,
            launched,
            adopt_round,
            adopt_alloc_bytes: adopt_alloc,
            total_s: started.elapsed().as_secs_f64(),
        },
        polys,
    )
}

fn drive_rav(
    kernel: &mut OptimizedInstructionRaVirtualizationKernel<Fr>,
    config: &RavBenchConfig,
    deferred: bool,
) -> (PassTiming, Vec<Vec<Fr>>, Vec<Fr>) {
    let adopt_round = if deferred { 4 } else { 3 };
    let alloc = adopt_alloc_bytes(config.log_t, config.num_committed(), deferred);
    let started = Instant::now();
    let (mut timing, polys) = drive_rounds(kernel, config.log_t, adopt_round, alloc);
    let claims = InstructionRaVirtualizationInputClaims {
        instruction_ra: Vec::new(),
    };
    #[expect(clippy::unwrap_used, reason = "bench harness: fail loudly")]
    let outputs = kernel
        .output_claims(&claims)
        .unwrap()
        .committed_instruction_ra;
    timing.total_s = started.elapsed().as_secs_f64();
    (timing, polys, outputs)
}

fn drive_bool(
    kernel: &mut BoolKernel,
    config: &BoolBenchConfig,
    deferred: bool,
) -> (
    PassTiming,
    Vec<Vec<Fr>>,
    SumcheckOutputClaims<Fr, Booleanity<Fr>>,
) {
    let adopt_round = if deferred { 4 } else { 3 };
    let alloc = adopt_alloc_bytes(config.log_t, config.num_polys(), deferred);
    let started = Instant::now();
    let (mut timing, polys) = drive_rounds(kernel.as_mut(), config.log_t, adopt_round, alloc);
    let claims = BooleanityInputClaims {
        address_phase: Fr::from_u64(0),
    };
    #[expect(clippy::unwrap_used, reason = "bench harness: fail loudly")]
    let outputs = kernel.output_claims(&claims).unwrap();
    timing.total_s = started.elapsed().as_secs_f64();
    (timing, polys, outputs)
}

/// One RAV arm: build the device kernel and run one timed pass. `deferred`
/// selects the adoption schedule via the driver-build env knob.
pub fn run_device_pass(config: &RavBenchConfig, rows: &BenchRows, deferred: bool) -> PassTiming {
    std::env::set_var("JOLT_RAV_DEFERRED_ADOPT", if deferred { "1" } else { "0" });
    let mut kernel = build_rav_kernel(config, &rows.rows, true);
    let (timing, _, _) = drive_rav(&mut kernel, config, deferred);
    timing
}

/// RAV correctness oracle for one arm: the device pipeline's wire
/// polynomials and output claims must be byte-equal to the driverless CPU
/// twin's.
pub fn assert_arm_parity(config: &RavBenchConfig, rows: &BenchRows, deferred: bool) {
    std::env::set_var("JOLT_RAV_DEFERRED_ADOPT", if deferred { "1" } else { "0" });
    let mut cpu = build_rav_kernel(config, &rows.rows, false);
    let mut device = build_rav_kernel(config, &rows.rows, true);
    let (_, cpu_polys, cpu_outputs) = drive_rav(&mut cpu, config, deferred);
    let (_, device_polys, device_outputs) = drive_rav(&mut device, config, deferred);
    assert_eq!(
        cpu_polys, device_polys,
        "st6b RAV arm parity: wire polynomials diverged (deferred={deferred})"
    );
    assert_eq!(
        cpu_outputs, device_outputs,
        "st6b RAV arm parity: output claims diverged (deferred={deferred})"
    );
}

/// One booleanity-cycle arm, as [`run_device_pass`].
pub fn run_bool_device_pass(
    config: &BoolBenchConfig,
    rows: &BenchRows,
    deferred: bool,
) -> PassTiming {
    std::env::set_var("JOLT_BOOL_DEFERRED_ADOPT", if deferred { "1" } else { "0" });
    let mut kernel = build_bool_kernel(config, &rows.rows, true);
    let (timing, _, _) = drive_bool(&mut kernel, config, deferred);
    timing
}

/// Booleanity-cycle correctness oracle, as [`assert_arm_parity`].
pub fn assert_bool_arm_parity(config: &BoolBenchConfig, rows: &BenchRows, deferred: bool) {
    std::env::set_var("JOLT_BOOL_DEFERRED_ADOPT", if deferred { "1" } else { "0" });
    let mut cpu = build_bool_kernel(config, &rows.rows, false);
    let mut device = build_bool_kernel(config, &rows.rows, true);
    let (_, cpu_polys, cpu_outputs) = drive_bool(&mut cpu, config, deferred);
    let (_, device_polys, device_outputs) = drive_bool(&mut device, config, deferred);
    assert_eq!(
        cpu_polys, device_polys,
        "st6b Bool arm parity: wire polynomials diverged (deferred={deferred})"
    );
    assert_eq!(
        cpu_outputs, device_outputs,
        "st6b Bool arm parity: output claims diverged (deferred={deferred})"
    );
}
