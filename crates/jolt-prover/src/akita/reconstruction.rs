//! The packed reconstruction phase (the head of the stage-8 region): one
//! batched sumcheck settling every virtualized word/chunk claim against its
//! committed one-hot decomposition — members in canonical commitment-object
//! order (untrusted advice, trusted advice, bytecode chunks, program image),
//! present exactly when their object exists in the public shape.
//!
//! Pure orchestration mirroring `stage8::reconstruction::verify`: the batch
//! and its consumed claims come from the verifier's promoted
//! [`build_reconstruction_parts`], the challenges from the generated batch
//! draw, and the members prove through the shared generated stage driver.
//! The member kernels live here (naive tier over the relations' own
//! expressions): the reconstruction relations exist only on the packed
//! build, so their `PrepareKernel` impls hang off [`JoltAkitaBackend`]
//! directly rather than a `jolt-kernels` slot.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::{
    committed_lane_vars, BYTECODE_LANE_LAYOUT,
};
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::lattice::geometry::{byte_place_vars, BYTE_BITS, WORD_BYTES};
use jolt_claims::protocols::jolt::lattice::relations::advice_reconstruction::{
    trusted_advice_bytes_opening, untrusted_advice_bytes_opening,
};
use jolt_claims::protocols::jolt::lattice::relations::bytecode_reconstruction::{
    bytecode_circuit_flag_opening, bytecode_imm_bytes_opening, bytecode_instruction_flag_opening,
    bytecode_lookup_selector_opening, bytecode_raf_flag_opening,
    bytecode_register_selector_opening, bytecode_unexpanded_pc_bytes_opening,
};
use jolt_claims::protocols::jolt::{
    BytecodeChunkReconstructionPublic, BytecodeRegisterLane, JoltDerivedId,
    ProgramImageReconstructionPublic, TrustedAdviceReconstructionPublic,
    UntrustedAdviceReconstructionPublic,
};
use jolt_field::{CanonicalBytes, Field, FixedByteSize};
use jolt_kernels::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel,
};
use jolt_lookup_tables::{InstructionLookupTable, XLEN};
use jolt_openings::CommitmentScheme;
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial};
use jolt_riscv::{Flags, InterleavedBitsMarker, JoltInstructionRow, CIRCUIT_FLAGS};
use jolt_sumcheck::{ClearSumcheckRecorder, SumcheckProof};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
use jolt_verifier::stages::stage7::outputs::Stage7ClearOutput;
use jolt_verifier::stages::stage8::reconstruction::{
    build_reconstruction_parts, BytecodeChunkReconstructionInstance,
    ProgramImageReconstructionInstance, ReconstructionClearOutput, ReconstructionOutputClaims,
    ReconstructionOutputPoints, ReconstructionParts, TrustedAdviceReconstructionInstance,
    UntrustedAdviceReconstructionInstance,
};
use jolt_verifier::CheckedInputs;
use jolt_witness::JoltWitnessPlane;

use super::witness::{decode_row, INSTRUCTION_FLAG_ORDER};
use super::JoltAkitaBackend;
use crate::{ProverError, StageProver as _};

mod drivers {
    use jolt_verifier::stages::stage8::reconstruction::{
        BytecodeChunkReconstructionInstance, ProgramImageReconstructionInstance,
        ReconstructionChallenges, ReconstructionInputClaims, ReconstructionInputPoints,
        ReconstructionOutputClaims, ReconstructionOutputPoints, ReconstructionSumchecks,
        TrustedAdviceReconstructionInstance, UntrustedAdviceReconstructionInstance,
    };

    use crate::driver::impl_stage_prover;

    jolt_verifier::reconstruction_sumchecks_members!(impl_stage_prover);
}

/// The reconstruction phase's outputs: the wire proof (`None` exactly when
/// the phase is absent), the wire claims, and the verifier-typed carrier the
/// packed stage-8 opening consumes.
pub struct ReconstructionProverOutput<F: Field, C> {
    pub sumcheck_proof: Option<SumcheckProof<F, C>>,
    pub claims: ReconstructionOutputClaims<F>,
    pub clear_output: ReconstructionClearOutput<F>,
}

/// Prove the reconstruction phase on `transcript` (positioned at the stage-7
/// boundary). Zero transcript interaction when the phase is absent (the span
/// still fires, so the taxonomy presence set is workload-independent).
#[tracing::instrument(skip_all)]
pub fn prove_reconstruction<F, PCS, C, T>(
    backend: &JoltAkitaBackend<F, PCS>,
    session: &mut ProofSession,
    checked: &CheckedInputs,
    stage6b: &Stage6bClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    witness: &dyn JoltWitnessPlane<F>,
    transcript: &mut T,
) -> Result<ReconstructionProverOutput<F, C>, ProverError<F>>
where
    F: Field + CanonicalBytes,
    PCS: CommitmentScheme<Field = F>,
    C: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let Some(ReconstructionParts {
        sumchecks,
        input_values,
        input_points,
    }) = build_reconstruction_parts(checked, stage6b, stage7)?
    else {
        return Ok(ReconstructionProverOutput {
            sumcheck_proof: None,
            claims: ReconstructionOutputClaims {
                untrusted_advice: None,
                trusted_advice: None,
                bytecode: None,
                program_image: None,
            },
            clear_output: ReconstructionClearOutput {
                output_values: ReconstructionOutputClaims {
                    untrusted_advice: None,
                    trusted_advice: None,
                    bytecode: None,
                    program_image: None,
                },
                output_points: ReconstructionOutputPoints {
                    untrusted_advice: None,
                    trusted_advice: None,
                    bytecode: None,
                    program_image: None,
                },
            },
        });
    };

    let challenges = sumchecks.draw_challenges(transcript)?;
    let proved = sumchecks.prove(
        backend,
        session,
        witness,
        &input_values,
        &input_points,
        &challenges,
        ClearSumcheckRecorder::<F, C>::new(),
        transcript,
    )?;

    Ok(ReconstructionProverOutput {
        sumcheck_proof: Some(proved.recorded.proof),
        claims: proved.output_claims.clone(),
        clear_output: ReconstructionClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
        },
    })
}

fn eq_table<F: Field>(point: &[F]) -> Vec<F> {
    EqPolynomial::new(point.to_vec()).evaluations()
}

/// The byte-decode weight table over a `(byte ‖ place)` block:
/// `T[(byte << place_bits) | place] = byte · 256^place`.
fn byte_decode_table<F: Field>(place_bits: usize) -> Vec<F> {
    let mut table = vec![F::zero(); 1usize << (BYTE_BITS + place_bits)];
    for byte in 0..(1usize << BYTE_BITS) {
        for place in 0..(1usize << place_bits) {
            table[(byte << place_bits) | place] = F::from_u64(byte as u64) * F::pow2(8 * place);
        }
    }
    table
}

impl<F, PCS> PrepareKernel<F, UntrustedAdviceReconstructionInstance<F>> for JoltAkitaBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, UntrustedAdviceReconstructionInstance<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = UntrustedAdviceReconstructionInstance<F>>>,
        KernelError<F>,
    > {
        use jolt_verifier::stages::relations::ConcreteSumcheck as _;

        let relation = inputs.relation;
        let rounds = relation.rounds();
        let r_word = inputs.points.word.as_slice();
        let word_vars = rounds - byte_place_vars();
        if r_word.len() != word_vars {
            return Err(KernelError::InvariantViolation {
                reason: "untrusted advice word point arity disagrees with the cell domain",
            });
        }
        let r_reference = &inputs.challenges.r_reference;
        if r_reference.len() != rounds {
            return Err(KernelError::InvariantViolation {
                reason: "untrusted advice reference point arity disagrees with the cell domain",
            });
        }

        let byte_column = witness.oracle_table(untrusted_advice_bytes_opening().polynomial_id())?;

        // The publics, materialized over the big-endian (byte ‖ place ‖ word)
        // cell domain; LowToHigh binding reproduces the verifier's
        // reversed-point evaluations.
        let place_bits = WORD_BYTES.ilog2() as usize;
        let cells = 1usize << rounds;
        let eq_full = eq_table(r_reference);
        let eq_place_word_base = eq_table(&r_reference[BYTE_BITS..]);
        let eq_word_base = eq_table(r_word);
        let decode_block = byte_decode_table::<F>(place_bits);
        let mut eq_place_word = vec![F::zero(); cells];
        let mut eq_word = vec![F::zero(); cells];
        let mut decode = vec![F::zero(); cells];
        let low_mask = (1usize << (rounds - BYTE_BITS)) - 1;
        let word_mask = (1usize << word_vars) - 1;
        for cell in 0..cells {
            eq_place_word[cell] = eq_place_word_base[cell & low_mask];
            eq_word[cell] = eq_word_base[cell & word_mask];
            decode[cell] = decode_block[cell >> word_vars];
        }

        let opening_tables = BTreeMap::from([(
            untrusted_advice_bytes_opening(),
            Polynomial::new(byte_column),
        )]);
        let derived_tables = BTreeMap::from([
            (
                JoltDerivedId::from(UntrustedAdviceReconstructionPublic::EqBytePlaceWord),
                Polynomial::new(eq_full),
            ),
            (
                JoltDerivedId::from(UntrustedAdviceReconstructionPublic::EqPlaceWord),
                Polynomial::new(eq_place_word),
            ),
            (
                JoltDerivedId::from(UntrustedAdviceReconstructionPublic::ByteDecode),
                Polynomial::new(decode),
            ),
            (
                JoltDerivedId::from(UntrustedAdviceReconstructionPublic::EqWord),
                Polynomial::new(eq_word),
            ),
        ]);
        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}

impl<F, PCS> PrepareKernel<F, TrustedAdviceReconstructionInstance<F>> for JoltAkitaBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, TrustedAdviceReconstructionInstance<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = TrustedAdviceReconstructionInstance<F>>>,
        KernelError<F>,
    > {
        let r_word = inputs.points.word.as_slice();
        let byte_column: Vec<F> =
            witness.oracle_table(trusted_advice_bytes_opening().polynomial_id())?;
        let folded = fold_word_dimension(&byte_column, r_word)?;

        let place_bits = WORD_BYTES.ilog2() as usize;
        let opening_tables =
            BTreeMap::from([(trusted_advice_bytes_opening(), Polynomial::new(folded))]);
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(TrustedAdviceReconstructionPublic::ByteDecode),
            Polynomial::new(byte_decode_table::<F>(place_bits)),
        )]);
        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}

impl<F, PCS> PrepareKernel<F, ProgramImageReconstructionInstance<F>> for JoltAkitaBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, ProgramImageReconstructionInstance<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = ProgramImageReconstructionInstance<F>>>,
        KernelError<F>,
    > {
        let r_word = inputs.points.word.as_slice();
        let image_words =
            super::witness::program_image_words_padded(witness.program_preprocessing());
        if image_words.len() != 1usize << r_word.len() {
            return Err(KernelError::InvariantViolation {
                reason: "program image word point arity disagrees with the padded image",
            });
        }
        // The word-folded byte one-hot cells, built directly from the padded
        // image words: T[(byte ‖ place)] = Σ_w eq(r_word, w) · [byte_of(w, place) = byte].
        let eq_word = eq_table(r_word);
        let place_bits = WORD_BYTES.ilog2() as usize;
        let mut folded = vec![F::zero(); 1usize << byte_place_vars()];
        for (word_index, word) in image_words.iter().enumerate() {
            for place in 0..WORD_BYTES {
                let byte = (word >> (8 * place)) as u8 as usize;
                folded[(byte << place_bits) | place] += eq_word[word_index];
            }
        }

        let opening_tables = BTreeMap::from([(
            jolt_claims::protocols::jolt::lattice::relations::program_image_reconstruction::program_image_bytes_opening(),
            Polynomial::new(folded),
        )]);
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(ProgramImageReconstructionPublic::ByteDecode),
            Polynomial::new(byte_decode_table::<F>(place_bits)),
        )]);
        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}

/// Fold the word (low-index) dimension of a `(byte ‖ place ‖ word)` cell
/// table at `r_word`: `out[bp] = Σ_w eq(r_word, w) · cells[(bp << wv) | w]`.
fn fold_word_dimension<F: Field>(cells: &[F], r_word: &[F]) -> Result<Vec<F>, KernelError<F>> {
    let word_vars = r_word.len();
    let blocks = 1usize << byte_place_vars();
    if cells.len() != blocks << word_vars {
        return Err(KernelError::InvariantViolation {
            reason: "advice byte cell table arity disagrees with the word point",
        });
    }
    let eq_word = eq_table(r_word);
    Ok((0..blocks)
        .map(|bp| {
            (0..1usize << word_vars)
                .map(|w| cells[(bp << word_vars) | w] * eq_word[w])
                .sum()
        })
        .collect())
}

impl<F, PCS> PrepareKernel<F, BytecodeChunkReconstructionInstance<F>> for JoltAkitaBackend<F, PCS>
where
    F: Field + CanonicalBytes,
    PCS: CommitmentScheme<Field = F>,
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeChunkReconstructionInstance<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = BytecodeChunkReconstructionInstance<F>>>,
        KernelError<F>,
    > {
        use jolt_verifier::stages::relations::ConcreteSumcheck as _;

        let relation = inputs.relation;
        let rounds = relation.rounds();
        let chunks = inputs.claims.chunks.len();
        let shared_point = inputs
            .points
            .chunks
            .first()
            .ok_or(KernelError::InvariantViolation {
                reason: "bytecode reconstruction consumed no chunk points",
            })?;
        let lane_vars = committed_lane_vars();
        if shared_point.len() < lane_vars {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode chunk point is below the lane prefix",
            });
        }
        let (r_lane, r_row) = shared_point.split_at(lane_vars);
        let log_rows = r_row.len();
        let eq_lane = eq_table(r_lane);
        let eq_row = eq_table(r_row);
        let rows = 1usize << log_rows;
        let bytecode = &witness.program_preprocessing().bytecode.bytecode;
        let chunk_rows = |chunk: usize| -> &[JoltInstructionRow] {
            let start = (chunk * rows).min(bytecode.len());
            let end = ((chunk + 1) * rows).min(bytecode.len());
            &bytecode[start..end]
        };

        let layout = BYTECODE_LANE_LAYOUT;
        let imm_byte_width = <F as FixedByteSize>::NUM_BYTES;
        let imm_place_bits = imm_byte_width.ilog2() as usize;
        let pc_place_bits = WORD_BYTES.ilog2() as usize;
        // The lookup block pads the (non-power-of-two) table count up; the
        // padded cells carry zero weight and no column mass.
        let lookup_vars = (layout.raf_flag_idx - layout.lookup_start)
            .next_power_of_two()
            .ilog2() as usize;
        let selector_vars = REGISTER_ADDRESS_BITS;
        let pc_vars = BYTE_BITS + pc_place_bits;
        let imm_vars = BYTE_BITS + imm_place_bits;
        debug_assert_eq!(rounds, imm_vars.max(pc_vars));

        // Every leg's own variables are the LOW-order bits of the sumcheck
        // index. The missing high coordinates are zero-pinned through the
        // DERIVED weight (`Π (1 − v_i)` — the verifier folds the pin into the
        // leg's public), so the COLUMN table is constant in them: replicated
        // across the high bits, its bound value is exactly
        // `column(v_own ‖ r_row)` — the packed-slot claim the final opening
        // consumes. Zero-extending the column too would square the pin and
        // shift every claim off the committed column.
        let cells = 1usize << rounds;
        let zero_extended = |own_values: Vec<F>| -> Vec<F> {
            let mut table = vec![F::zero(); cells];
            table[..own_values.len()].copy_from_slice(&own_values);
            table
        };
        let replicated = |own_values: Vec<F>| -> Vec<F> {
            debug_assert!(cells.is_multiple_of(own_values.len().max(1)));
            let mut table = Vec::with_capacity(cells);
            while table.len() < cells {
                table.extend_from_slice(&own_values);
            }
            table.truncate(cells);
            table
        };

        let mut opening_tables = BTreeMap::new();
        for chunk in 0..chunks {
            let instructions = chunk_rows(chunk);
            let fold_flag = |predicate: &dyn Fn(&JoltInstructionRow) -> bool| -> Vec<F> {
                let mut value = F::zero();
                for (row, instruction) in instructions.iter().enumerate() {
                    if predicate(instruction) {
                        value += eq_row[row];
                    }
                }
                vec![value]
            };
            for lane in BytecodeRegisterLane::ALL {
                let mut own = vec![F::zero(); 1usize << selector_vars];
                for (row, instruction) in instructions.iter().enumerate() {
                    let register = match lane {
                        BytecodeRegisterLane::Rs1 => instruction.operands.rs1,
                        BytecodeRegisterLane::Rs2 => instruction.operands.rs2,
                        BytecodeRegisterLane::Rd => instruction.operands.rd,
                    };
                    if let Some(register) = register {
                        own[register as usize] += eq_row[row];
                    }
                }
                let _ = opening_tables.insert(
                    bytecode_register_selector_opening(chunk, lane),
                    Polynomial::new(replicated(own)),
                );
            }
            for (flag, circuit_flag) in CIRCUIT_FLAGS.iter().enumerate() {
                let _ = opening_tables.insert(
                    bytecode_circuit_flag_opening(chunk, flag),
                    Polynomial::new(replicated(fold_flag(&|instruction| {
                        decode_row(instruction).circuit_flags()[*circuit_flag]
                    }))),
                );
            }
            for (flag, instruction_flag) in INSTRUCTION_FLAG_ORDER.iter().enumerate() {
                let _ = opening_tables.insert(
                    bytecode_instruction_flag_opening(chunk, flag),
                    Polynomial::new(replicated(fold_flag(&|instruction| {
                        decode_row(instruction).instruction_flags()[*instruction_flag]
                    }))),
                );
            }
            {
                let mut own = vec![F::zero(); 1usize << lookup_vars];
                for (row, instruction) in instructions.iter().enumerate() {
                    if let Some(table) =
                        InstructionLookupTable::<XLEN>::lookup_table(&decode_row(instruction))
                    {
                        own[table.index()] += eq_row[row];
                    }
                }
                let _ = opening_tables.insert(
                    bytecode_lookup_selector_opening(chunk),
                    Polynomial::new(replicated(own)),
                );
            }
            let _ = opening_tables.insert(
                bytecode_raf_flag_opening(chunk),
                Polynomial::new(replicated(fold_flag(&|instruction| {
                    !decode_row(instruction)
                        .circuit_flags()
                        .is_interleaved_operands()
                }))),
            );
            {
                // PC bytes: hot at (byte(place), place) per row; padding rows
                // land on byte 0 like the committed witness.
                let mut own = vec![F::zero(); 1usize << pc_vars];
                for place in 0..WORD_BYTES {
                    for (row, eq) in eq_row.iter().enumerate().take(rows) {
                        let byte = instructions.get(row).map_or(0, |instruction| {
                            ((instruction.address as u64) >> (8 * place)) as u8
                        }) as usize;
                        own[(byte << pc_place_bits) | place] += *eq;
                    }
                }
                let _ = opening_tables.insert(
                    bytecode_unexpanded_pc_bytes_opening(chunk),
                    Polynomial::new(replicated(own)),
                );
            }
            {
                // Imm bytes: the field's canonical little-endian bytes of
                // `from_i128(imm)`; padding rows land on byte 0.
                let mut own = vec![F::zero(); 1usize << imm_vars];
                for (row, eq) in eq_row.iter().enumerate().take(rows) {
                    let bytes: Vec<u8> = match instructions.get(row) {
                        Some(instruction) => {
                            F::from_i128(instruction.operands.imm).to_bytes_le_vec()
                        }
                        None => vec![0u8; imm_byte_width],
                    };
                    for place in 0..imm_byte_width {
                        let byte = bytes.get(place).copied().unwrap_or(0) as usize;
                        own[(byte << imm_place_bits) | place] += *eq;
                    }
                }
                let _ = opening_tables.insert(
                    bytecode_imm_bytes_opening(chunk),
                    Polynomial::new(replicated(own)),
                );
            }
        }

        // The derived weight tables: each leg's lane-eq weight at its own low
        // indices (zero-pinned high coordinates), matching
        // `derive_output_term`'s multilinear form.
        let mut derived_tables = BTreeMap::new();
        for lane in BytecodeRegisterLane::ALL {
            let block_start = match lane {
                BytecodeRegisterLane::Rs1 => layout.rs1_start,
                BytecodeRegisterLane::Rs2 => layout.rs2_start,
                BytecodeRegisterLane::Rd => layout.rd_start,
            };
            let own: Vec<F> = (0..1usize << selector_vars)
                .map(|register| eq_lane[block_start + register])
                .collect();
            let _ = derived_tables.insert(
                JoltDerivedId::from(BytecodeChunkReconstructionPublic::RegisterSelectorWeight(
                    lane,
                )),
                Polynomial::new(zero_extended(own)),
            );
        }
        for flag in 0..CIRCUIT_FLAGS.len() {
            let _ = derived_tables.insert(
                JoltDerivedId::from(BytecodeChunkReconstructionPublic::LaneWeight(
                    layout.circuit_start + flag,
                )),
                Polynomial::new(zero_extended(vec![eq_lane[layout.circuit_start + flag]])),
            );
        }
        for flag in 0..INSTRUCTION_FLAG_ORDER.len() {
            let _ = derived_tables.insert(
                JoltDerivedId::from(BytecodeChunkReconstructionPublic::LaneWeight(
                    layout.instr_start + flag,
                )),
                Polynomial::new(zero_extended(vec![eq_lane[layout.instr_start + flag]])),
            );
        }
        {
            let own: Vec<F> = (0..(layout.raf_flag_idx - layout.lookup_start))
                .map(|value| eq_lane[layout.lookup_start + value])
                .collect();
            let _ = derived_tables.insert(
                JoltDerivedId::from(BytecodeChunkReconstructionPublic::LookupSelectorWeight),
                Polynomial::new(zero_extended(own)),
            );
        }
        let _ = derived_tables.insert(
            JoltDerivedId::from(BytecodeChunkReconstructionPublic::LaneWeight(
                layout.raf_flag_idx,
            )),
            Polynomial::new(zero_extended(vec![eq_lane[layout.raf_flag_idx]])),
        );
        {
            let decode = byte_decode_table::<F>(pc_place_bits);
            let own: Vec<F> = decode
                .into_iter()
                .map(|weight| eq_lane[layout.unexp_pc_idx] * weight)
                .collect();
            let _ = derived_tables.insert(
                JoltDerivedId::from(BytecodeChunkReconstructionPublic::PcByteDecode),
                Polynomial::new(zero_extended(own)),
            );
        }
        {
            let decode = byte_decode_table::<F>(imm_place_bits);
            let own: Vec<F> = decode
                .into_iter()
                .map(|weight| eq_lane[layout.imm_idx] * weight)
                .collect();
            let _ = derived_tables.insert(
                JoltDerivedId::from(BytecodeChunkReconstructionPublic::ImmByteDecode),
                Polynomial::new(zero_extended(own)),
            );
        }

        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
