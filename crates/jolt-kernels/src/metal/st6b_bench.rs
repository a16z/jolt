//! W2-st6b lane harness: the stage-6b instruction-RA-virtualization device
//! round pipeline in isolation — lazy gathers, the dense adoption, and the
//! fused dense rounds — at production geometry (16 committed = 4 virtual ×
//! 4 per-virtual, 8-bit chunks of the 128-bit lookup index), without an
//! end-to-end prove run.
//!
//! Driven by the `st6b_rav_microbench` example. The pipeline advances
//! through the REAL kernel (`OptimizedInstructionRaVirtualizationKernel`
//! with the slot's device driver installed) under the batch engine's
//! two-phase contract (`begin_round` for every member, then
//! `collect_round`), so `begin_s` is exactly the phase-1 serialization a
//! production round pays before any synchronous CPU member can start.
//!
//! Arms (same binary, `JOLT_RAV_DEFERRED_ADOPT` flipped per kernel build):
//! - `sync`: legacy — third bind materializes at `cycles / 8` inside
//!   `begin_round` (blocking `jk_ra_materialize` + wait), the round message
//!   then re-reads the fresh dense tables.
//! - `deferred`: a fourth lazy round at width 8, then ONE detached
//!   `jk_rav_adopt_round` materializes at `cycles / 16` fused with that
//!   round's message.
//!
//! Correctness oracle: a driverless CPU twin runs the same challenge/claim
//! stream; every wire round polynomial and the final output claims must be
//! byte-equal (the exact parity contract the slot tests pin).

use std::sync::Arc;
use std::time::Instant;

use jolt_claims::protocols::jolt::relations::instruction::InstructionRaVirtualizationInputClaims;
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_sumcheck::ProveRounds;

use crate::optimized::instruction_ra_virtualization::OptimizedInstructionRaVirtualizationKernel;
use crate::optimized::instruction_read_raf::{InstructionCycleRow, InstructionRows};
use crate::SumcheckKernel as _;

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

/// One timed pipeline pass.
pub struct RavPassTiming {
    /// Per round: (`begin_round` seconds, `collect_round` seconds).
    pub rounds: Vec<(f64, f64)>,
    /// Round index whose bind lands the dense adoption (`horizon - 1`... the
    /// first round served from `PendingAdopt`/fresh dense state): 3 for the
    /// legacy arm, 4 for the deferred arm.
    pub adopt_round: usize,
    /// Wall of the full pass: all rounds + `finish_rounds` + output claims.
    pub total_s: f64,
}

impl RavPassTiming {
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

/// Uniform-random lookup indices: the gather kernels' data-oblivious (and
/// cache-conservative) case.
pub fn bench_rows(config: &RavBenchConfig) -> BenchRows {
    let mut state = config.seed;
    BenchRows {
        rows: Arc::new(InstructionRows::new(
            (0..1usize << config.log_t)
                .map(|j| {
                    let index = match j {
                        0 => 0u128,
                        1 => u128::MAX,
                        _ => ((splitmix(&mut state) as u128) << 64) | splitmix(&mut state) as u128,
                    };
                    InstructionCycleRow::new(index, None, false, None, None)
                })
                .collect(),
        )),
    }
}

fn build_kernel(
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

/// Drive one kernel through the engine's two-phase contract, timing each
/// round; returns the wire round polynomials alongside the timings so the
/// caller can pin parity.
fn drive(
    kernel: &mut OptimizedInstructionRaVirtualizationKernel<Fr>,
    log_t: usize,
    adopt_round: usize,
) -> (RavPassTiming, Vec<Vec<Fr>>, Vec<Fr>) {
    let started = Instant::now();
    let mut claim = Fr::from_u64(0xBEEF);
    let mut rounds = Vec::with_capacity(log_t);
    let mut polys = Vec::with_capacity(log_t);
    for round in 0..log_t {
        let bind = round.checked_sub(1).map(challenge);
        let begin_at = Instant::now();
        #[expect(clippy::unwrap_used, reason = "bench harness: fail loudly")]
        let _launched = kernel.begin_round(bind, round, claim).unwrap();
        let begin_s = begin_at.elapsed().as_secs_f64();
        let collect_at = Instant::now();
        #[expect(clippy::unwrap_used, reason = "bench harness: fail loudly")]
        let poly = kernel.collect_round(bind, round, claim).unwrap();
        let collect_s = collect_at.elapsed().as_secs_f64();
        rounds.push((begin_s, collect_s));
        claim = poly.evaluate(challenge(round));
        polys.push(poly.coefficients().to_vec());
    }
    #[expect(clippy::unwrap_used, reason = "bench harness: fail loudly")]
    kernel.finish_rounds(challenge(log_t - 1)).unwrap();
    let claims = InstructionRaVirtualizationInputClaims {
        instruction_ra: Vec::new(),
    };
    #[expect(clippy::unwrap_used, reason = "bench harness: fail loudly")]
    let outputs = kernel
        .output_claims(&claims)
        .unwrap()
        .committed_instruction_ra;
    let total_s = started.elapsed().as_secs_f64();
    (
        RavPassTiming {
            rounds,
            adopt_round,
            total_s,
        },
        polys,
        outputs,
    )
}

/// One arm: build the device kernel and run one timed pass. `deferred`
/// selects the adoption schedule via the driver-build env knob.
pub fn run_device_pass(config: &RavBenchConfig, rows: &BenchRows, deferred: bool) -> RavPassTiming {
    std::env::set_var("JOLT_RAV_DEFERRED_ADOPT", if deferred { "1" } else { "0" });
    let mut kernel = build_kernel(config, &rows.rows, true);
    let adopt_round = if deferred { 4 } else { 3 };
    let (timing, _, _) = drive(&mut kernel, config.log_t, adopt_round);
    timing
}

/// Correctness oracle for one arm: the device pipeline's wire polynomials
/// and output claims must be byte-equal to the driverless CPU twin's.
pub fn assert_arm_parity(config: &RavBenchConfig, rows: &BenchRows, deferred: bool) {
    std::env::set_var("JOLT_RAV_DEFERRED_ADOPT", if deferred { "1" } else { "0" });
    let mut cpu = build_kernel(config, &rows.rows, false);
    let mut device = build_kernel(config, &rows.rows, true);
    let adopt_round = if deferred { 4 } else { 3 };
    let (_, cpu_polys, cpu_outputs) = drive(&mut cpu, config.log_t, adopt_round);
    let (_, device_polys, device_outputs) = drive(&mut device, config.log_t, adopt_round);
    assert_eq!(
        cpu_polys, device_polys,
        "st6b RAV arm parity: wire polynomials diverged (deferred={deferred})"
    );
    assert_eq!(
        cpu_outputs, device_outputs,
        "st6b RAV arm parity: output claims diverged (deferred={deferred})"
    );
}
