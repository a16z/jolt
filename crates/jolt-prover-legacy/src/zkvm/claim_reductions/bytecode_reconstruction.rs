//! Bytecode chunk reconstruction sumcheck (lattice/packed mode).
//!
//! Settles the base `BytecodeClaimReduction`'s per-chunk claims (all chunks
//! at one shared `(lane ‖ row)` point) against the precommitted per-lane
//! sub-columns of `W_prog` in ONE γ-batched sumcheck — the chunk polynomials
//! are never PCS-opened.
//!
//! Leg embedding contract (shared with the verifier): the batch binds the
//! widest byte lane's `(byte ‖ place)` variables (`8 + log2(imm_byte_width)`
//! rounds); every narrower column's own variables are the LOW-order tail of
//! that vector, bound by the first rounds, and its grid is zero-extended
//! over the missing high coordinates. The row point stays FIXED at `r_row`
//! (no row rounds): each leg's column table is the sub-column pre-bound at
//! `r_row` — a scatter of `eq(r_row, ·)` over the column's hot cells. A
//! fully-bound (or 0-round flag) leg keeps contributing through the
//! zero-extension as `s·(1 − X)` per remaining round, which is exactly the
//! verifier's `Π (1 − v_i)` zero-pin.
//!
//! Per chunk (scale `γ^chunk`) the legs are: three register selectors
//! (own = log2(REGISTER_COUNT)), every circuit/instruction flag and the RAF
//! flag (own = 0), the lookup-table selector (own = log2 of the padded
//! table block), and the pc/imm byte decodes (own = `8 + log2(places)`).
//! Every leg is a product of two multilinears, hence degree 2.

use allocative::Allocative;
#[cfg(feature = "prover")]
use jolt_riscv::JoltInstructionRow;

use crate::field::JoltField;
#[cfg(feature = "prover")]
use crate::poly::eq_poly::EqPolynomial;
#[cfg(feature = "prover")]
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
#[cfg(feature = "prover")]
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN,
};
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint};
#[cfg(feature = "prover")]
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
#[cfg(feature = "prover")]
use crate::zkvm::bytecode::chunks::BYTECODE_LANE_LAYOUT;
use crate::zkvm::bytecode::chunks::COMMITTED_BYTECODE_LANE_CAPACITY;
#[cfg(feature = "prover")]
use crate::zkvm::instruction::{Flags, InstructionLookup, InterleavedBitsMarker};
#[cfg(feature = "prover")]
use crate::zkvm::lookup_table::LookupTables;
use crate::zkvm::witness::CommittedPolynomial;
#[cfg(feature = "prover")]
use common::constants::XLEN;
#[cfg(feature = "prover")]
use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};

const DEGREE_BOUND: usize = 2;

/// Byte places of one word lane (the unexpanded pc decomposes into 8 bytes).
const WORD_BYTES: usize = 8;
const BYTE_BITS: usize = 8;

#[derive(Allocative, Clone)]
pub struct BytecodeReconstructionSumcheckParams<F: JoltField> {
    /// `γ^chunk` scales, one per bytecode chunk.
    pub gamma_powers: Vec<F>,
    /// Byte places of the immediate lane (the field's canonical byte width).
    pub imm_byte_width: usize,
    /// The lane half of the consumed chunk claims' shared point.
    pub r_lane: Vec<F::Challenge>,
    /// The row half; every produced lane opening is suffixed with it.
    pub r_row: Vec<F::Challenge>,
}

impl<F: JoltField> BytecodeReconstructionSumcheckParams<F> {
    /// Draws the chunk-batching γ and splits the consumed chunk claims'
    /// shared point. `imm_byte_width` is the field's canonical byte width
    /// (the witness lane serializes `F::from_i128(imm)`).
    pub fn new(
        chunk_count: usize,
        imm_byte_width: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = vec![F::one(); chunk_count];
        for i in 1..chunk_count {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        let (point, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::BytecodeChunk(0),
            SumcheckId::BytecodeClaimReduction,
        );
        let lane_vars = COMMITTED_BYTECODE_LANE_CAPACITY.log_2();
        let (r_lane, r_row) = point.r.split_at(lane_vars);

        Self {
            gamma_powers,
            imm_byte_width,
            r_lane: r_lane.to_vec(),
            r_row: r_row.to_vec(),
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for BytecodeReconstructionSumcheckParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.gamma_powers
            .iter()
            .enumerate()
            .map(|(chunk, gamma)| {
                let (_, claim) = accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeChunk(chunk),
                    SumcheckId::BytecodeClaimReduction,
                );
                *gamma * claim
            })
            .sum()
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        BYTE_BITS + WORD_BYTES.log_2().max(self.imm_byte_width.log_2())
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        unimplemented!(
            "zk x lattice is rejected fail-closed; BytecodeChunkReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; BytecodeChunkReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; BytecodeChunkReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, _sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; BytecodeChunkReconstruction carries no BlindFold plumbing"
        )
    }
}

/// One leg's live sumcheck state: two multilinears over the leg's own
/// variables while they bind, then a frozen `(pin-scaled scalar, opening
/// value)` pair contributing `s·(1 − X)` per remaining round.
#[cfg(feature = "prover")]
enum LegState<F: JoltField> {
    Active {
        weight: MultilinearPolynomial<F>,
        column: MultilinearPolynomial<F>,
    },
    Exhausted {
        /// `γ^chunk · W(v_own) · column(v_own ‖ r_row) · Π (1 − r_i)` over
        /// the rounds bound since exhaustion.
        scaled: F,
        /// `column(v_own ‖ r_row)` — the produced opening value.
        value: F,
    },
}

#[cfg(feature = "prover")]
#[derive(Allocative)]
struct Leg<F: JoltField> {
    polynomial: CommittedPolynomial,
    own_vars: usize,
    /// `γ^chunk` — folded into the weight table (and the exhausted scalar).
    #[allocative(skip)]
    state: LegState<F>,
}

/// The prover: per `(chunk, leg)` a weight table (the lane-eq / byte-decode
/// kernel, γ^chunk-scaled) and a column table (the sub-column pre-bound at
/// `r_row`), both over the leg's own variables, zero-extended by bookkeeping
/// rather than materialization.
#[cfg(feature = "prover")]
#[derive(Allocative)]
pub struct BytecodeReconstructionSumcheckProver<F: JoltField> {
    legs: Vec<Leg<F>>,
    pub params: BytecodeReconstructionSumcheckParams<F>,
}

#[cfg(feature = "prover")]
impl<F: JoltField> BytecodeReconstructionSumcheckProver<F> {
    /// Builds the per-chunk leg tables from the materialized bytecode. Row
    /// domain per chunk is `2^log_rows` (bytecode rows, zero-padded); the
    /// byte lanes encode padding rows as byte-0 hot, matching the committed
    /// witness.
    #[tracing::instrument(skip_all, name = "BytecodeReconstructionSumcheckProver::initialize")]
    pub fn initialize(
        params: BytecodeReconstructionSumcheckParams<F>,
        bytecode: &[JoltInstructionRow],
        log_rows: usize,
    ) -> Self {
        let layout = BYTECODE_LANE_LAYOUT;
        let rows = 1usize << log_rows;
        let eq_lane: Vec<F> = EqPolynomial::evals(&params.r_lane);
        let eq_row: Vec<F> = EqPolynomial::evals(&params.r_row);
        debug_assert_eq!(eq_row.len(), rows);
        let register_count = layout.rs2_start - layout.rs1_start;
        let lookup_count = layout.raf_flag_idx - layout.lookup_start;
        let lookup_cells = lookup_count.next_power_of_two();
        let imm_limb_bits = params.imm_byte_width.log_2();
        let place_bits = WORD_BYTES.log_2();

        // The byte-decode weight `byte · 256^place` over `(byte ‖ place)`
        // cells, shared by the pc and imm lanes up to their place counts.
        let byte_kernel = |limb_bits: usize| -> Vec<F> {
            let mut kernel = vec![F::zero(); 1 << (BYTE_BITS + limb_bits)];
            let mut place_value = F::one();
            for place in 0..(1usize << limb_bits) {
                for byte in 0..(1usize << BYTE_BITS) {
                    kernel[(byte << limb_bits) | place] = F::from_u64(byte as u64) * place_value;
                }
                place_value *= F::from_u64(256);
            }
            kernel
        };

        let imm_bytes = |imm: i128| -> Vec<u8> {
            let value = F::from_i128(imm);
            let mut bytes = Vec::new();
            value
                .serialize_compressed(&mut bytes)
                .expect("field element serialization is infallible for Vec sinks");
            debug_assert!(
                bytes[params.imm_byte_width..].iter().all(|byte| *byte == 0),
                "imm must fit the canonical {}-byte lane",
                params.imm_byte_width
            );
            bytes.truncate(params.imm_byte_width);
            bytes
        };

        let mut legs = Vec::new();
        for (chunk, gamma) in params.gamma_powers.iter().enumerate() {
            let start = (chunk * rows).min(bytecode.len());
            let end = ((chunk + 1) * rows).min(bytecode.len());
            let chunk_rows = &bytecode[start..end];

            // Register selectors: one-hot per row when the operand exists.
            for (lane, block_start) in [
                (0usize, layout.rs1_start),
                (1, layout.rs2_start),
                (2, layout.rd_start),
            ] {
                let weight: Vec<F> = (0..register_count)
                    .map(|register| *gamma * eq_lane[block_start + register])
                    .collect();
                let mut column = vec![F::zero(); register_count];
                for (row, instruction) in chunk_rows.iter().enumerate() {
                    let register = match lane {
                        0 => instruction.operands.rs1,
                        1 => instruction.operands.rs2,
                        _ => instruction.operands.rd,
                    };
                    if let Some(register) = register {
                        column[register as usize] += eq_row[row];
                    }
                }
                legs.push(Leg {
                    polynomial: CommittedPolynomial::BytecodeRegisterSelector(chunk, lane),
                    own_vars: register_count.log_2(),
                    state: LegState::Active {
                        weight: weight.into(),
                        column: column.into(),
                    },
                });
            }

            // Flag lanes: 0-round legs — scalar weight × the flag column
            // bound at r_row.
            let mut flag_leg = |polynomial: CommittedPolynomial, lane: usize, value: F| {
                legs.push(Leg {
                    polynomial,
                    own_vars: 0,
                    state: LegState::Exhausted {
                        scaled: *gamma * eq_lane[lane] * value,
                        value,
                    },
                });
            };
            for flag in 0..NUM_CIRCUIT_FLAGS {
                let value = chunk_rows
                    .iter()
                    .enumerate()
                    .filter(|(_, instruction)| instruction.circuit_flags()[flag])
                    .map(|(row, _)| eq_row[row])
                    .sum();
                flag_leg(
                    CommittedPolynomial::BytecodeCircuitFlag(chunk, flag),
                    layout.circuit_start + flag,
                    value,
                );
            }
            for flag in 0..NUM_INSTRUCTION_FLAGS {
                let value = chunk_rows
                    .iter()
                    .enumerate()
                    .filter(|(_, instruction)| instruction.instruction_flags()[flag])
                    .map(|(row, _)| eq_row[row])
                    .sum();
                flag_leg(
                    CommittedPolynomial::BytecodeInstructionFlag(chunk, flag),
                    layout.instr_start + flag,
                    value,
                );
            }
            let raf_value = chunk_rows
                .iter()
                .enumerate()
                .filter(|(_, instruction)| {
                    !InterleavedBitsMarker::is_interleaved_operands(&instruction.circuit_flags())
                })
                .map(|(row, _)| eq_row[row])
                .sum();
            flag_leg(
                CommittedPolynomial::BytecodeRafFlag(chunk),
                layout.raf_flag_idx,
                raf_value,
            );

            // Lookup-table selector: one-hot over the padded table block.
            let weight: Vec<F> = (0..lookup_cells)
                .map(|table| {
                    if table < lookup_count {
                        *gamma * eq_lane[layout.lookup_start + table]
                    } else {
                        F::zero()
                    }
                })
                .collect();
            let mut column = vec![F::zero(); lookup_cells];
            for (row, instruction) in chunk_rows.iter().enumerate() {
                if let Some(table) = InstructionLookup::<XLEN>::lookup_table(instruction) {
                    column[LookupTables::<XLEN>::enum_index(&table)] += eq_row[row];
                }
            }
            legs.push(Leg {
                polynomial: CommittedPolynomial::BytecodeLookupSelector(chunk),
                own_vars: lookup_cells.log_2(),
                state: LegState::Active {
                    weight: weight.into(),
                    column: column.into(),
                },
            });

            // Unexpanded-pc byte decode: padding rows scatter byte 0,
            // matching the committed witness (weight 0·256^place keeps the
            // claim unaffected).
            let mut column = vec![F::zero(); 1 << (BYTE_BITS + place_bits)];
            for row in 0..rows {
                let address = chunk_rows
                    .get(row)
                    .map_or(0, |instruction| instruction.address as u64);
                for place in 0..WORD_BYTES {
                    let byte = ((address >> (8 * place)) & 0xff) as usize;
                    column[(byte << place_bits) | place] += eq_row[row];
                }
            }
            legs.push(Leg {
                polynomial: CommittedPolynomial::BytecodeUnexpandedPcBytes(chunk),
                own_vars: BYTE_BITS + place_bits,
                state: LegState::Active {
                    weight: byte_kernel(place_bits)
                        .into_iter()
                        .map(|w| *gamma * eq_lane[layout.unexp_pc_idx] * w)
                        .collect::<Vec<F>>()
                        .into(),
                    column: column.into(),
                },
            });

            // Immediate byte decode over the canonical field bytes.
            let mut column = vec![F::zero(); 1 << (BYTE_BITS + imm_limb_bits)];
            for row in 0..rows {
                let bytes = match chunk_rows.get(row) {
                    Some(instruction) => imm_bytes(instruction.operands.imm),
                    None => vec![0u8; params.imm_byte_width],
                };
                for (place, byte) in bytes.into_iter().enumerate() {
                    column[((byte as usize) << imm_limb_bits) | place] += eq_row[row];
                }
            }
            legs.push(Leg {
                polynomial: CommittedPolynomial::BytecodeImmBytes(chunk),
                own_vars: BYTE_BITS + imm_limb_bits,
                state: LegState::Active {
                    weight: byte_kernel(imm_limb_bits)
                        .into_iter()
                        .map(|w| *gamma * eq_lane[layout.imm_idx] * w)
                        .collect::<Vec<F>>()
                        .into(),
                    column: column.into(),
                },
            });
        }

        Self { legs, params }
    }

    /// The produced opening for one leg family, in the verifier's canonical
    /// order: `(reverse(challenges[..own]) ‖ r_row)`.
    fn leg_opening_point(&self, own_vars: usize, challenges: &[F::Challenge]) -> Vec<F::Challenge> {
        let mut point: Vec<F::Challenge> = challenges[..own_vars].to_vec();
        point.reverse();
        point.extend_from_slice(&self.params.r_row);
        point
    }
}

#[cfg(feature = "prover")]
impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for BytecodeReconstructionSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "BytecodeReconstructionSumcheckProver::compute_message"
    )]
    fn compute_message(
        &mut self,
        _round: usize,
        previous_claim: F,
    ) -> crate::poly::unipoly::UniPoly<F> {
        let mut eval_zero = F::zero();
        let mut eval_two = F::zero();
        for leg in &self.legs {
            match &leg.state {
                LegState::Active { weight, column } => {
                    let half = weight.len() / 2;
                    for i in 0..half {
                        let w =
                            weight.sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);
                        let c =
                            column.sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);
                        eval_zero += w[0] * c[0];
                        eval_two += w[1] * c[1];
                    }
                }
                LegState::Exhausted { scaled, .. } => {
                    // The zero-extension: `s·(1 − X)`.
                    eval_zero += *scaled;
                    eval_two -= *scaled;
                }
            }
        }
        crate::poly::unipoly::UniPoly::from_evals_and_hint(previous_claim, &[eval_zero, eval_two])
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        for leg in &mut self.legs {
            match &mut leg.state {
                LegState::Active { weight, column } => {
                    weight.bind_parallel(r_j, BindingOrder::LowToHigh);
                    column.bind_parallel(r_j, BindingOrder::LowToHigh);
                    if weight.len() == 1 {
                        let value = column.final_sumcheck_claim();
                        leg.state = LegState::Exhausted {
                            scaled: weight.final_sumcheck_claim() * value,
                            value,
                        };
                    }
                }
                LegState::Exhausted { scaled, .. } => {
                    *scaled *= F::one() - Into::<F>::into(r_j);
                }
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Family-major, chunk-major within — the verifier's canonical
        // output-claim order.
        let families: [&dyn Fn(&CommittedPolynomial) -> bool; 7] = [
            &|p| matches!(p, CommittedPolynomial::BytecodeRegisterSelector(..)),
            &|p| matches!(p, CommittedPolynomial::BytecodeCircuitFlag(..)),
            &|p| matches!(p, CommittedPolynomial::BytecodeInstructionFlag(..)),
            &|p| matches!(p, CommittedPolynomial::BytecodeLookupSelector(_)),
            &|p| matches!(p, CommittedPolynomial::BytecodeRafFlag(_)),
            &|p| matches!(p, CommittedPolynomial::BytecodeUnexpandedPcBytes(_)),
            &|p| matches!(p, CommittedPolynomial::BytecodeImmBytes(_)),
        ];
        for family in families {
            for leg in self.legs.iter().filter(|leg| family(&leg.polynomial)) {
                let LegState::Exhausted { value, .. } = &leg.state else {
                    panic!("every reconstruction leg is exhausted after the final round");
                };
                accumulator.append_dense(
                    leg.polynomial,
                    SumcheckId::BytecodeChunkReconstruction,
                    self.leg_opening_point(leg.own_vars, sumcheck_challenges),
                    *value,
                );
            }
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[cfg(all(test, feature = "prover"))]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::transcripts::Blake2bTranscript;
    use crate::zkvm::bytecode::chunks::{for_each_active_lane_value, ActiveLaneValue};
    use ark_bn254::Fr;
    use ark_std::{One, Zero};
    use jolt_riscv::{JoltInstructionKind, NormalizedOperands};

    type Challenge = <Fr as JoltField>::Challenge;

    const LOG_ROWS: usize = 2;
    const CHUNKS: usize = 2;
    const IMM_BYTES: usize = 32;

    fn row(
        kind: JoltInstructionKind,
        rd: Option<u8>,
        rs1: Option<u8>,
        rs2: Option<u8>,
        imm: i128,
        address: usize,
    ) -> JoltInstructionRow {
        JoltInstructionRow {
            instruction_kind: kind,
            address,
            operands: NormalizedOperands { rd, rs1, rs2, imm },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    fn bytecode() -> Vec<JoltInstructionRow> {
        vec![
            row(
                JoltInstructionKind::ADDI,
                Some(1),
                Some(2),
                None,
                3,
                0x80000000,
            ),
            row(
                JoltInstructionKind::ADD,
                Some(5),
                Some(6),
                Some(7),
                0,
                0x80000004,
            ),
            row(
                JoltInstructionKind::SD,
                None,
                Some(8),
                Some(9),
                -16,
                0x80000008,
            ),
            row(
                JoltInstructionKind::BEQ,
                None,
                Some(3),
                Some(4),
                64,
                0x8000000c,
            ),
            row(
                JoltInstructionKind::XOR,
                Some(10),
                Some(11),
                Some(12),
                0,
                0x80000010,
            ),
        ]
    }

    /// `eq(point, bits(index))`, point big-endian (msb first).
    fn eq_index(point: &[Challenge], index: usize) -> Fr {
        point.iter().enumerate().fold(Fr::one(), |acc, (bit, r)| {
            let r: Fr = (*r).into();
            if (index >> (point.len() - 1 - bit)) & 1 == 1 {
                acc * r
            } else {
                acc * (Fr::one() - r)
            }
        })
    }

    /// The full round loop against synthetic bytecode: every round folds the
    /// previous claim, and the final claim equals the verifier's closed form
    /// `Σ_c γ^c Σ_legs public(bound) · opening` over the cached openings —
    /// with the input claim pinned to the direct lane-value evaluation
    /// (`for_each_active_lane_value`), the base-reduction ground truth.
    #[test]
    fn round_loop_reconstructs_the_chunk_claims() {
        let bytecode = bytecode();
        let rows = 1usize << LOG_ROWS;
        let lane_vars = COMMITTED_BYTECODE_LANE_CAPACITY.log_2();

        let r_lane: Vec<Challenge> = (0..lane_vars)
            .map(|i| Challenge::from((3 + 7 * i as u64) as u128))
            .collect();
        let r_row: Vec<Challenge> = (0..LOG_ROWS)
            .map(|i| Challenge::from((101 + 13 * i as u64) as u128))
            .collect();
        let shared_point: Vec<Challenge> = r_lane.iter().chain(r_row.iter()).copied().collect();

        // Ground-truth chunk claims: the direct lane-value evaluation.
        let eq_lane_f = EqPolynomial::<Fr>::evals(&r_lane);
        let eq_row_f = EqPolynomial::<Fr>::evals(&r_row);
        let chunk_claim = |chunk: usize| -> Fr {
            let start = (chunk * rows).min(bytecode.len());
            let end = ((chunk + 1) * rows).min(bytecode.len());
            let mut claim = Fr::zero();
            for (row, instruction) in bytecode[start..end].iter().enumerate() {
                for_each_active_lane_value::<Fr>(instruction, |lane, value| {
                    let value = match value {
                        ActiveLaneValue::One => Fr::one(),
                        ActiveLaneValue::Scalar(v) => v,
                    };
                    claim += eq_lane_f[lane] * eq_row_f[row] * value;
                });
            }
            claim
        };

        let mut accumulator = ProverOpeningAccumulator::<Fr>::new(LOG_ROWS + lane_vars);
        for chunk in 0..CHUNKS {
            accumulator.append_dense(
                CommittedPolynomial::BytecodeChunk(chunk),
                SumcheckId::BytecodeClaimReduction,
                shared_point.clone(),
                chunk_claim(chunk),
            );
        }

        let mut transcript = Blake2bTranscript::new(b"bytecode-reconstruction-test");
        let params = BytecodeReconstructionSumcheckParams::<Fr>::new(
            CHUNKS,
            IMM_BYTES,
            &accumulator,
            &mut transcript,
        );
        let mut prover =
            BytecodeReconstructionSumcheckProver::initialize(params.clone(), &bytecode, LOG_ROWS);

        let mut claim = params.input_claim(&accumulator);
        let total_rounds = params.num_rounds();
        assert_eq!(total_rounds, BYTE_BITS + IMM_BYTES.log_2());
        let mut challenges = Vec::new();
        for round in 0..total_rounds {
            let message = SumcheckInstanceProver::<Fr, Blake2bTranscript>::compute_message(
                &mut prover,
                round,
                claim,
            );
            assert_eq!(
                message.eval_at_zero() + message.eval_at_one(),
                claim,
                "round {round} message must fold the previous claim"
            );
            let r_j = Challenge::from((100 + 7 * round) as u128);
            claim = message.evaluate(&r_j);
            challenges.push(r_j);
            SumcheckInstanceProver::<Fr, Blake2bTranscript>::ingest_challenge(
                &mut prover,
                r_j,
                round,
            );
        }
        SumcheckInstanceProver::<Fr, Blake2bTranscript>::cache_openings(
            &prover,
            &mut accumulator,
            &challenges,
        );

        // The verifier's closed form over the cached openings.
        let layout = BYTECODE_LANE_LAYOUT;
        let register_count = layout.rs2_start - layout.rs1_start;
        let lookup_count = layout.raf_flag_idx - layout.lookup_start;
        let bound: Vec<Challenge> = challenges.iter().rev().copied().collect();
        let leg_point = |own: usize| -> &[Challenge] { &bound[total_rounds - own..] };
        let zero_pin = |own: usize| -> Fr { eq_index(&bound[..total_rounds - own], 0) };
        let selector_block_weight = |start: usize, own: usize, count: usize| -> Fr {
            (0..count)
                .map(|t| eq_index(&r_lane, start + t) * eq_index(leg_point(own), t))
                .sum()
        };
        let byte_decode_weight = |own: usize| -> Fr {
            let (r_byte, r_place) = leg_point(own).split_at(BYTE_BITS);
            let byte: Fr = (0..1usize << BYTE_BITS)
                .map(|b| Fr::from_u64(b as u64) * eq_index(r_byte, b))
                .sum();
            let place: Fr = (0..1usize << r_place.len())
                .map(|p| {
                    let mut value = Fr::one();
                    for _ in 0..p {
                        value *= Fr::from_u64(256);
                    }
                    value * eq_index(r_place, p)
                })
                .sum();
            byte * place
        };
        let opening = |polynomial: CommittedPolynomial| -> Fr {
            accumulator
                .try_get_committed_polynomial_opening(
                    polynomial,
                    SumcheckId::BytecodeChunkReconstruction,
                )
                .unwrap()
                .1
        };

        let mut expected = Fr::zero();
        for (chunk, gamma) in params.gamma_powers.iter().enumerate() {
            let selector_vars = register_count.log_2();
            for (lane, start) in [
                (0usize, layout.rs1_start),
                (1, layout.rs2_start),
                (2, layout.rd_start),
            ] {
                expected += *gamma
                    * selector_block_weight(start, selector_vars, register_count)
                    * zero_pin(selector_vars)
                    * opening(CommittedPolynomial::BytecodeRegisterSelector(chunk, lane));
            }
            for flag in 0..NUM_CIRCUIT_FLAGS {
                expected += *gamma
                    * eq_index(&r_lane, layout.circuit_start + flag)
                    * zero_pin(0)
                    * opening(CommittedPolynomial::BytecodeCircuitFlag(chunk, flag));
            }
            for flag in 0..NUM_INSTRUCTION_FLAGS {
                expected += *gamma
                    * eq_index(&r_lane, layout.instr_start + flag)
                    * zero_pin(0)
                    * opening(CommittedPolynomial::BytecodeInstructionFlag(chunk, flag));
            }
            let lookup_vars = lookup_count.next_power_of_two().log_2();
            expected += *gamma
                * selector_block_weight(layout.lookup_start, lookup_vars, lookup_count)
                * zero_pin(lookup_vars)
                * opening(CommittedPolynomial::BytecodeLookupSelector(chunk));
            expected += *gamma
                * eq_index(&r_lane, layout.raf_flag_idx)
                * zero_pin(0)
                * opening(CommittedPolynomial::BytecodeRafFlag(chunk));
            let pc_vars = BYTE_BITS + WORD_BYTES.log_2();
            expected += *gamma
                * eq_index(&r_lane, layout.unexp_pc_idx)
                * byte_decode_weight(pc_vars)
                * zero_pin(pc_vars)
                * opening(CommittedPolynomial::BytecodeUnexpandedPcBytes(chunk));
            let imm_vars = BYTE_BITS + IMM_BYTES.log_2();
            expected += *gamma
                * eq_index(&r_lane, layout.imm_idx)
                * byte_decode_weight(imm_vars)
                * zero_pin(imm_vars)
                * opening(CommittedPolynomial::BytecodeImmBytes(chunk));
        }

        assert_eq!(
            claim, expected,
            "final claim must equal the verifier's closed form over the produced openings"
        );
    }
}
