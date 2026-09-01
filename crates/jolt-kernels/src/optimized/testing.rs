//! Synthetic-trace fixtures and the reference/optimized lockstep parity
//! harness shared by the optimized RAM kernel tests.
//!
//! The fixture replays a small RAM op script into a real [`TraceBackend`]
//! (state-consistent pre/post values, a tiny synthetic memory layout, and
//! the guest-style trailing termination write that keeps `RamValFinal`
//! consistent with the initial state on untouched words — the invariant the
//! optimized kernels' `val_init` reconstruction relies on).

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test support module: fail loudly"
)]

use common::jolt_device::{JoltDevice, MemoryLayout};
use jolt_claims::protocols::jolt::{JoltChallengeId, JoltOneHotConfig};
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::{Field, Fr, Ring};
use jolt_poly::UnivariatePoly;
use jolt_program::execution::{
    JoltProgram, OwnedTrace, RamAccess, RamRead, RamWrite, RegisterRead, RegisterState,
    RegisterWrite, TraceOutput, TraceRow,
};
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing};
use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessPlane, TraceBackend};
use rand_core::SeedableRng;

use crate::{ProverInputs, SumcheckKernel};

/// Word addresses below this index are reserved for the layout's panic (0)
/// and termination (1) words; scripts should use words `>= 2`.
pub(crate) const TERMINATION_WORD: u64 = 1;

/// The fixture's word 0 lives at this byte address (the layout's lowest).
const BASE_ADDRESS: u64 = 0x1000;

/// The fixture layout's lowest mapped address (word 0), as the RAF
/// relation's `lowest_address` expects it.
pub(crate) fn fixture_lowest_address() -> u64 {
    BASE_ADDRESS
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FixtureShape {
    pub log_t: usize,
    pub ram_k: usize,
}

impl FixtureShape {
    pub fn log_k(self) -> usize {
        assert!(self.ram_k.is_power_of_two());
        self.ram_k.trailing_zeros() as usize
    }
}

/// One scripted cycle. Pre-values are replayed from the running RAM state,
/// so scripts stay trace-consistent by construction.
#[derive(Clone, Copy, Debug)]
pub(crate) enum RamOp {
    Read { word: u64 },
    Write { word: u64, post: u64 },
    None,
}

/// Run `f` against a trace backend replaying `ops` (plus the trailing
/// termination write), padded to `2^log_t` cycles.
pub(crate) fn with_ram_fixture<R>(
    shape: FixtureShape,
    ops: Vec<RamOp>,
    f: impl FnOnce(&dyn JoltWitnessPlane<Fr>) -> R,
) -> R {
    with_ram_fixture_init(shape, Vec::new(), ops, f)
}

/// [`with_ram_fixture`] with nonzero initial RAM values: `init_words[i]`
/// seeds word `2 + i` (the reserved panic/termination words stay zero). The
/// values ride in as trusted-advice bytes, which the witness backend
/// populates into BOTH the initial and the final RAM state — so untouched
/// nonzero words keep `RamValFinal` consistent with `val_init` without a
/// final-memory image. WARNING: the final-state advice populate also masks
/// script WRITES to seeded words in `RamValFinal`; only the never-accessed
/// fallback of the optimized `val_init` reconstruction reads those slots, so
/// read-write parity is unaffected, but scripts feeding a val-final-anchored
/// kernel must not write seeded words.
pub(crate) fn with_ram_fixture_init<R>(
    shape: FixtureShape,
    init_words: Vec<u64>,
    ops: Vec<RamOp>,
    f: impl FnOnce(&dyn JoltWitnessPlane<Fr>) -> R,
) -> R {
    assert!(ops.len() < 1usize << shape.log_t, "script too long");
    assert!(
        init_words.is_empty() || 2 + init_words.len() <= shape.ram_k,
        "init words exceed the RAM domain"
    );

    let memory_layout = MemoryLayout {
        trusted_advice_start: BASE_ADDRESS,
        untrusted_advice_start: BASE_ADDRESS,
        panic: BASE_ADDRESS,
        termination: BASE_ADDRESS + 8 * TERMINATION_WORD,
        ..Default::default()
    };

    let load = JoltInstructionRow {
        instruction_kind: JoltInstructionKind::LD,
        address: 0x8000_0000,
        operands: NormalizedOperands {
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: 3,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    };
    let store = JoltInstructionRow {
        instruction_kind: JoltInstructionKind::SD,
        address: 0x8000_0004,
        operands: NormalizedOperands {
            rd: None,
            rs1: Some(2),
            rs2: Some(3),
            imm: 0,
        },
        ..load
    };
    use std::sync::Arc;
    let preprocessing = Arc::new(JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing::preprocess(
            vec![load, store],
            load.address as u64,
            RV64IMAC_JOLT,
        )
        .unwrap(),
        ram: RAMPreprocessing::default(),
        memory_layout: memory_layout.clone(),
        max_padded_trace_length: 1 << shape.log_t,
    });
    let program = Arc::new(JoltProgram::default());

    let mut state = vec![0u64; shape.ram_k];
    let trusted_advice: Vec<u8> = if init_words.is_empty() {
        Vec::new()
    } else {
        // Two zero words keep the reserved panic/termination words zero in
        // the advice populate.
        let mut bytes = vec![0u8; 16];
        for (i, &value) in init_words.iter().enumerate() {
            state[2 + i] = value;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    };
    let mut script = ops;
    if shape.ram_k > TERMINATION_WORD as usize {
        script.push(RamOp::Write {
            word: TERMINATION_WORD,
            post: 1,
        });
    }
    // Build RAM traffic as valid final LD/SD rows.
    let mut rd_value = 0;
    let rows: Vec<TraceRow> = script
        .into_iter()
        .map(|op| {
            let (instruction, registers, ram_access) = match op {
                RamOp::Read { word } => {
                    let address = BASE_ADDRESS + 8 * word;
                    let value = state[word as usize];
                    let registers = RegisterState {
                        rs1: Some(RegisterRead {
                            register: 2,
                            value: address,
                        }),
                        rd: Some(RegisterWrite {
                            register: 1,
                            pre_value: rd_value,
                            post_value: value,
                        }),
                        ..Default::default()
                    };
                    rd_value = value;
                    (load, registers, RamAccess::Read(RamRead { address, value }))
                }
                RamOp::Write { word, post } => {
                    let address = BASE_ADDRESS + 8 * word;
                    let pre_value = state[word as usize];
                    state[word as usize] = post;
                    (
                        store,
                        RegisterState {
                            rs1: Some(RegisterRead {
                                register: 2,
                                value: address,
                            }),
                            rs2: Some(RegisterRead {
                                register: 3,
                                value: post,
                            }),
                            ..Default::default()
                        },
                        RamAccess::Write(RamWrite {
                            address,
                            pre_value,
                            post_value: post,
                        }),
                    )
                }
                RamOp::None => (
                    JoltInstructionRow::default(),
                    RegisterState::default(),
                    RamAccess::NoOp,
                ),
            };
            TraceRow::new(instruction, registers, ram_access).unwrap()
        })
        .collect();

    let device = JoltDevice {
        memory_layout,
        trusted_advice,
        ..Default::default()
    };
    let config = JoltVmWitnessConfig::new(
        shape.log_t,
        shape.ram_k,
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        },
    );
    let inputs = JoltVmWitnessInputs::new(
        &program,
        &preprocessing,
        TraceOutput::new(OwnedTrace::new(rows), device, None, None),
    );
    let backend = TraceBackend::new(config, inputs);
    f(&backend)
}

/// Deterministic scalars for fixture points and challenges.
pub(crate) fn random_scalars(count: usize, seed: u64) -> Vec<Fr> {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
    (0..count).map(|_| Fr::random(&mut rng)).collect()
}

/// Trailing-zero-insensitive round-polynomial coefficients: the engine sums
/// members into `max_degree + 1` slots and trims the batched polynomial, so
/// a member's trailing zeros never reach the wire.
fn trimmed(poly: &UnivariatePoly<Fr>) -> Vec<Fr> {
    let mut coefficients = poly.coefficients().to_vec();
    while coefficients.last() == Some(&Fr::from_u64(0)) {
        let _ = coefficients.pop();
    }
    coefficients
}

/// Drive both kernels through the fused round loop in lockstep with the
/// same deterministic challenges, asserting per-round polynomial equality
/// (up to trailing zeros) and output-claim equality; returns the drawn
/// challenges for the caller's post-loop checks.
pub(crate) fn drive_parity_rounds<R>(
    reference: &mut dyn SumcheckKernel<Fr, Relation = R>,
    optimized: &mut dyn SumcheckKernel<Fr, Relation = R>,
    input_claim: Fr,
    inputs: &ProverInputs<'_, Fr, R>,
    challenge_seed: u64,
) -> Vec<Fr>
where
    R: ConcreteSumcheck<Fr>,
    SumcheckInputClaims<Fr, R>: InputClaims<Fr>,
    SumcheckOutputClaims<Fr, R>: OutputClaims<Fr> + PartialEq + core::fmt::Debug,
    ConcreteSumcheckChallenges<Fr, R>: SumcheckChallenges<Fr, JoltChallengeId>,
{
    let rounds = reference.num_rounds();
    assert_eq!(optimized.num_rounds(), rounds, "round count diverged");
    assert_eq!(inputs.relation.rounds(), rounds, "relation rounds diverged");

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(challenge_seed);
    let mut reference_claim = input_claim;
    let mut optimized_claim = input_claim;
    let mut challenges = Vec::with_capacity(rounds);
    let mut bind = None;
    for round in 0..rounds {
        // The reference (naive) member self-checks s(0) + s(1) against the
        // running claim, so a drifting optimized claim fails loudly here.
        let reference_poly = reference
            .prove_round(bind, round, reference_claim)
            .unwrap_or_else(|error| panic!("reference round {round}: {error}"));
        let optimized_poly = optimized
            .prove_round(bind, round, optimized_claim)
            .unwrap_or_else(|error| panic!("optimized round {round}: {error}"));
        assert_eq!(
            trimmed(&reference_poly),
            trimmed(&optimized_poly),
            "round {round} polynomial diverged"
        );
        let challenge = Fr::random(&mut rng);
        reference_claim = reference_poly.evaluate(challenge);
        optimized_claim = optimized_poly.evaluate(challenge);
        challenges.push(challenge);
        bind = Some(challenge);
    }
    if let Some(challenge) = bind {
        reference.finish_rounds(challenge).unwrap();
        optimized.finish_rounds(challenge).unwrap();
    }

    let reference_outputs = reference.output_claims(inputs.claims).unwrap();
    let optimized_outputs = optimized.output_claims(inputs.claims).unwrap();
    assert_eq!(
        reference_outputs, optimized_outputs,
        "output claims diverged"
    );
    challenges
}

/// [`drive_parity_rounds`] plus both kernels' derived-table self-checks.
pub(crate) fn assert_parity<R>(
    mut reference: Box<dyn SumcheckKernel<Fr, Relation = R>>,
    mut optimized: Box<dyn SumcheckKernel<Fr, Relation = R>>,
    input_claim: Fr,
    inputs: &ProverInputs<'_, Fr, R>,
    challenge_seed: u64,
) where
    R: ConcreteSumcheck<Fr>,
    SumcheckInputClaims<Fr, R>: InputClaims<Fr>,
    SumcheckOutputClaims<Fr, R>: OutputClaims<Fr> + PartialEq + core::fmt::Debug,
    ConcreteSumcheckChallenges<Fr, R>: SumcheckChallenges<Fr, JoltChallengeId>,
{
    let challenges = drive_parity_rounds(
        reference.as_mut(),
        optimized.as_mut(),
        input_claim,
        inputs,
        challenge_seed,
    );
    let output_points = inputs
        .relation
        .derive_opening_points(&challenges, inputs.points)
        .unwrap();
    reference
        .validate_derived_tables(
            inputs.relation,
            inputs.points,
            &output_points,
            inputs.challenges,
        )
        .unwrap();
    optimized
        .validate_derived_tables(
            inputs.relation,
            inputs.points,
            &output_points,
            inputs.challenges,
        )
        .unwrap();
}
