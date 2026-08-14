//! Shared parity-test support for the registers kernel family: a
//! register-consistent synthetic trace behind a full `TraceBackend` witness
//! plane, deterministic challenge sequences, and the engine-mirroring parity
//! driver (bind-then-compute, running claim via `poly.evaluate(challenge)`).

use core::fmt::Debug;

use jolt_claims::protocols::jolt::{JoltChallengeId, JoltOneHotConfig};
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_program::execution::{
    JoltProgram, OwnedTrace, RegisterRead, RegisterState, RegisterWrite, TraceOutput, TraceRow,
};
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing};
use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims,
};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessPlane, TraceBackend};

use crate::reference::ReferenceBackend;
use crate::{PrepareKernel, ProofSession, ProverInputs};

/// Deterministic nonzero field elements (an LCG over odd u64s), used for
/// both fixed points and round challenges.
pub(crate) fn challenge_sequence(len: usize, seed: u64) -> Vec<Fr> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            Fr::from_u64(state | 1)
        })
        .collect()
}

/// The counting pass and the write pass must agree on every operand
/// pattern — including raw index 255, which the old `u8::MAX` sentinel
/// collided with (count 0, write 1 → pass-2 window overrun).
#[cfg(feature = "parallel")]
#[test]
fn cycle_entry_count_matches_cycle_entries() {
    use super::RegisterCycleRow;
    let candidates: [Option<u8>; 5] = [None, Some(0), Some(5), Some(127), Some(255)];
    for rs1 in candidates {
        for rs2 in candidates {
            for rd in candidates {
                let cycle = RegisterCycleRow {
                    rs1: rs1.map(|register| (register, 11)),
                    rs2: rs2.map(|register| (register, 22)),
                    rd: rd.map(|register| (register, 33, 44)),
                };
                let (_, len) = cycle.entries::<Fr>(0);
                assert_eq!(
                    cycle.entry_count(),
                    len,
                    "count/write divergence for {cycle:?}"
                );
            }
        }
    }
}

/// Extraction rejects out-of-domain operand indices with the trace
/// oracle's error (the reference tier's fail-loud contract) instead of
/// scattering out of bounds; with several invalid cycles the collector
/// reports the FIRST one, independent of thread timing.
#[test]
fn collect_rejects_out_of_domain_register_indices() {
    use super::rows::CollectRegisterEntries;
    let mut fixture = TraceFixture::new();
    fixture.noop();
    fixture.op(Some(3), Some(2), None);
    fixture.rows.push(TraceRow {
        registers: RegisterState {
            rs1: None,
            rs2: Some(RegisterRead {
                register: 200,
                value: 7,
            }),
            rd: None,
        },
        ..TraceRow::default()
    });
    fixture.rows.push(TraceRow {
        registers: RegisterState {
            rs1: Some(RegisterRead {
                register: 255,
                value: 1,
            }),
            rs2: None,
            rd: None,
        },
        ..TraceRow::default()
    });
    fixture.with_plane(2, |witness| {
        let error = match CollectRegisterEntries::<Fr>::collect(witness, 1 << 2) {
            Err(error) => error,
            Ok(_) => panic!("collect accepted an out-of-domain register index"),
        };
        let message = format!("{error}");
        assert!(
            message.contains("register index 200"),
            "expected the first invalid cycle's index, got {message:?}"
        );
    });
}

/// A register-consistent trace builder: reads return the current register
/// state, writes advance it, so every witness identity the sumchecks
/// assume holds by construction.
pub(crate) struct TraceFixture {
    rows: Vec<TraceRow>,
    state: [u64; 128],
    counter: u64,
    instruction: JoltInstructionRow,
}

impl TraceFixture {
    pub(crate) fn new() -> Self {
        Self {
            rows: Vec::new(),
            state: [0; 128],
            counter: 0xDEAD_BEEF_0BAD_F00D,
            instruction: JoltInstructionRow {
                instruction_kind: JoltInstructionKind::ADDI,
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
            },
        }
    }

    pub(crate) fn noop(&mut self) {
        let instruction = self.instruction;
        self.rows.push(TraceRow {
            instruction,
            ..TraceRow::default()
        });
    }

    /// One cycle touching the given operands; the write value is a fresh
    /// pseudo-random u64.
    pub(crate) fn op(&mut self, rd: Option<u8>, rs1: Option<u8>, rs2: Option<u8>) {
        let read = |state: &[u64; 128], register: Option<u8>| {
            register.map(|register| RegisterRead {
                register,
                value: state[register as usize],
            })
        };
        let registers = RegisterState {
            rs1: read(&self.state, rs1),
            rs2: read(&self.state, rs2),
            rd: rd.map(|register| {
                self.counter = self
                    .counter
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let pre_value = self.state[register as usize];
                let post_value = self.counter;
                self.state[register as usize] = post_value;
                RegisterWrite {
                    register,
                    pre_value,
                    post_value,
                }
            }),
        };
        let instruction = self.instruction;
        self.rows.push(TraceRow {
            instruction,
            registers,
            ..TraceRow::default()
        });
    }

    /// Run `f` against a trace backend padded to `2^log_t` cycles.
    pub(crate) fn with_plane<R>(
        self,
        log_t: usize,
        f: impl FnOnce(&TraceBackend<OwnedTrace>) -> R,
    ) -> R {
        assert!(self.rows.len() <= 1 << log_t, "fixture overflows 2^log_t");
        use std::sync::Arc;
        let preprocessing = Arc::new(JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(
                vec![self.instruction],
                self.instruction.address as u64,
                RV64IMAC_JOLT,
            )
            .unwrap(),
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: 1 << log_t,
        });
        let program = Arc::new(JoltProgram::default());
        let config = JoltVmWitnessConfig::new(
            log_t,
            64,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        );
        let inputs = JoltVmWitnessInputs::new(
            &program,
            &preprocessing,
            TraceOutput::new(OwnedTrace::new(self.rows), Default::default(), None, None),
        );
        let backend = TraceBackend::new(config, inputs);
        f(&backend)
    }
}

/// A structured register workload: write-then-read chains, `rs1 == rs2`,
/// `rd == rs1` in one cycle, repeated writes, high register indices, and
/// interleaved no-ops. Emits exactly `cycles` rows.
pub(crate) fn structured_fixture(cycles: usize) -> TraceFixture {
    let mut fixture = TraceFixture::new();
    for step in 0..cycles {
        match step % 8 {
            0 => fixture.op(Some(5), Some(2), None),
            1 => fixture.op(Some(7), Some(5), Some(5)),
            2 => fixture.op(Some(5), Some(5), Some(7)),
            3 => fixture.noop(),
            4 => fixture.op(None, Some(7), Some(100)),
            5 => fixture.op(Some(127), Some(0), Some(5)),
            6 => fixture.op(Some(100), None, None),
            _ => fixture.op(Some(7), Some(127), Some(100)),
        }
    }
    fixture
}

/// Prepare the reference and optimized kernels from identical inputs,
/// drive both through the full round sequence asserting byte-identical
/// round polynomials, then assert equal typed output claims and run both
/// kernels' derived-table validation against the relation.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the seam's input decomposition"
)]
pub(crate) fn assert_kernel_parity<R>(
    optimized_slot: &dyn PrepareKernel<Fr, R>,
    witness: &dyn JoltWitnessPlane<Fr>,
    relation: &R,
    claims: &SumcheckInputClaims<Fr, R>,
    points: &SumcheckInputPoints<Fr, R>,
    challenges: &ConcreteSumcheckChallenges<Fr, R>,
    input_claim: Fr,
    round_challenges: &[Fr],
) where
    R: ConcreteSumcheck<Fr>,
    ReferenceBackend: PrepareKernel<Fr, R>,
    SumcheckInputClaims<Fr, R>: InputClaims<Fr>,
    SumcheckOutputClaims<Fr, R>: OutputClaims<Fr> + PartialEq + Debug,
    ConcreteSumcheckChallenges<Fr, R>: SumcheckChallenges<Fr, JoltChallengeId>,
{
    assert_kernel_parity_with_session(
        &mut ProofSession::default(),
        optimized_slot,
        witness,
        relation,
        claims,
        points,
        challenges,
        input_claim,
        round_challenges,
    );
}

/// [`assert_kernel_parity`] with a caller-supplied session for the
/// optimized kernel — exercises cross-member session carries.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the seam's input decomposition"
)]
pub(crate) fn assert_kernel_parity_with_session<R>(
    optimized_session: &mut ProofSession,
    optimized_slot: &dyn PrepareKernel<Fr, R>,
    witness: &dyn JoltWitnessPlane<Fr>,
    relation: &R,
    claims: &SumcheckInputClaims<Fr, R>,
    points: &SumcheckInputPoints<Fr, R>,
    challenges: &ConcreteSumcheckChallenges<Fr, R>,
    input_claim: Fr,
    round_challenges: &[Fr],
) where
    R: ConcreteSumcheck<Fr>,
    ReferenceBackend: PrepareKernel<Fr, R>,
    SumcheckInputClaims<Fr, R>: InputClaims<Fr>,
    SumcheckOutputClaims<Fr, R>: OutputClaims<Fr> + PartialEq + Debug,
    ConcreteSumcheckChallenges<Fr, R>: SumcheckChallenges<Fr, JoltChallengeId>,
{
    let mut reference_session = ProofSession::default();
    let mut reference = ReferenceBackend
        .prepare(
            &mut reference_session,
            witness,
            ProverInputs {
                relation,
                claims,
                points,
                challenges,
            },
        )
        .unwrap();
    let mut optimized = optimized_slot
        .prepare(
            optimized_session,
            witness,
            ProverInputs {
                relation,
                claims,
                points,
                challenges,
            },
        )
        .unwrap();

    let rounds = relation.rounds();
    assert_eq!(reference.num_rounds(), rounds);
    assert_eq!(optimized.num_rounds(), rounds);
    assert_eq!(round_challenges.len(), rounds);

    let mut claim = input_claim;
    for round in 0..rounds {
        let bind = (round > 0).then(|| round_challenges[round - 1]);
        let reference_poly = reference.prove_round(bind, round, claim).unwrap();
        let optimized_poly = optimized.prove_round(bind, round, claim).unwrap();
        assert_eq!(
            reference_poly, optimized_poly,
            "round {round} polynomial mismatch"
        );
        assert_eq!(
            optimized_poly.evaluate(Fr::from_u64(0)) + optimized_poly.evaluate(Fr::from_u64(1)),
            claim,
            "round {round} running-claim mismatch"
        );
        claim = reference_poly.evaluate(round_challenges[round]);
    }
    reference
        .finish_rounds(round_challenges[rounds - 1])
        .unwrap();
    optimized
        .finish_rounds(round_challenges[rounds - 1])
        .unwrap();

    let output_points = relation
        .derive_opening_points(round_challenges, points)
        .unwrap();
    reference
        .validate_derived_tables(relation, points, &output_points, challenges)
        .unwrap();
    optimized
        .validate_derived_tables(relation, points, &output_points, challenges)
        .unwrap();

    let reference_outputs = reference.output_claims(claims).unwrap();
    let optimized_outputs = optimized.output_claims(claims).unwrap();
    assert_eq!(
        reference_outputs, optimized_outputs,
        "output claims mismatch"
    );
}

/// A fixture guard: an all-zero witness would make parity vacuous, so the
/// input claim must be a nontrivial field element.
pub(crate) fn assert_nontrivial(claim: Fr) {
    assert_ne!(
        claim,
        Fr::from_u64(0),
        "degenerate fixture: zero input claim"
    );
    assert_ne!(
        claim,
        Fr::from_u64(1),
        "degenerate fixture: unit input claim"
    );
}
