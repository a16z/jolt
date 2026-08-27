//! Stage 0: input validation, the Fiat-Shamir preamble, and witness
//! commitment.
//!
//! The transcript work is the verifier's own exported code
//! ([`validate_inputs_from_parts`], [`absorb_transcript_preamble`],
//! [`absorb_transcript_commitments`]) — the two sides share the absorb
//! sequence structurally, so stage-0 Fiat-Shamir drift is impossible by
//! construction. The commitment compute is delegated to the `jolt-kernels`
//! witness-commitment kernel; only the absorbs happen here.

use common::jolt_device::JoltDevice;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineCommittedPolynomial;
use jolt_claims::protocols::jolt::JoltPolynomialId;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, TracePolynomialOrder};
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_kernels::reference::bytecode_read_raf::BytecodeReadRafWitness;
use jolt_kernels::reference::instruction_read_raf::InstructionReadRafWitness;
use jolt_kernels::{CommitmentGrid, JoltBackend, ProofSession, WitnessCommitment};
use jolt_openings::CommitmentScheme;
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::proof::JoltCommitments;
#[cfg(feature = "field-inline")]
use jolt_verifier::proof::{FieldInlineCommitments, FieldRegistersCommitments};
use jolt_verifier::{
    absorb_committed_program_commitments, absorb_transcript_commitments,
    absorb_transcript_preamble, validate_inputs_from_parts, CheckedInputs, ProofTranscriptConfig,
};
use jolt_witness::{
    validate_servable, JoltWitnessOracle, JoltWitnessPlane, RowSource, WitnessBundle,
};

use crate::config::advice_total_vars;
use crate::{CommittedProgramCandidates, JoltProverPreprocessing, ProverConfig, ProverError};

/// The externally supplied trusted-advice commitment (produced at
/// preprocessing time, before any proving) and its opening hint. Mirrors
/// legacy's prover-constructor pair: the commitment is absorbed in stage 0 and
/// batched in stage 8, and the hint joins the stage-8 hint combination.
pub struct TrustedAdviceCommitment<PCS: CommitmentScheme> {
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
}

/// Stage 0's outputs: the validated inputs, the seeded transcript (positioned
/// exactly where the verifier's `verify_until_stage1` leaves its own), the
/// witness commitments in wire form, the untrusted-advice commitment (proved
/// at prove time, carried on the proof), and the per-polynomial opening hints
/// the stage-8 joint opening will consume (advice hints included).
pub struct Stage0Output<PCS, T>
where
    PCS: CommitmentScheme,
{
    pub checked: CheckedInputs,
    pub transcript: T,
    pub commitments: JoltCommitments<PCS::Output>,
    pub untrusted_advice_commitment: Option<PCS::Output>,
    pub hints: Vec<(JoltCommittedPolynomial, PCS::OpeningHint)>,
    /// The field-inline opening hints, id-disjoint from the jolt hints; the
    /// FR joint-opening wiring consumes them in a later unit.
    #[cfg(feature = "field-inline")]
    pub field_inline_hints: Vec<(FieldInlineCommittedPolynomial, PCS::OpeningHint)>,
}

/// Validate inputs, seed the transcript, commit the witness (the untrusted
/// advice polynomial in its own balanced grid), and absorb the commitments
/// (main, untrusted advice, trusted advice, then the preprocessing-held
/// committed-program chunk/image commitments — the verifier's own absorb
/// order).
#[tracing::instrument(skip_all)]
pub fn prove_stage0<F, PCS, VC, T, W>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&TrustedAdviceCommitment<PCS>>,
    witness: &W,
    public_io: &JoltDevice,
) -> Result<Stage0Output<PCS, T>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: AppendToTranscript,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
    W: JoltWitnessPlane<F>,
{
    // Committed-program mode needs the prover-retained full program + hints;
    // require presence to agree with the verifier preprocessing's mode.
    if preprocessing.verifier.program.committed().is_some()
        != preprocessing.committed_program.is_some()
    {
        return Err(ProverError::Unsupported {
            reason: "committed-program prover data presence disagrees with the preprocessing mode",
        });
    }
    // The chunk commitments bake their trace order in at preprocessing time;
    // a disagreeing proof config would transpose the rebuilt chunk tables
    // against the absorbed commitments and fail only at verification.
    if preprocessing
        .committed_program
        .as_ref()
        .is_some_and(|committed| committed.trace_order != config.trace_polynomial_order)
    {
        return Err(ProverError::Unsupported {
            reason: "committed-program preprocessing was built for a different trace layout",
        });
    }
    let untrusted_advice_present = !public_io.untrusted_advice.is_empty();
    // Trusted-advice presence rides on the external commitment argument;
    // require it to agree with the advice bytes so a mismatch fails here
    // rather than as an opaque stage-4 sumcheck error (bytes without a
    // commitment) or as a nonstandard proof over the zero advice polynomial
    // (a commitment without bytes).
    if trusted_advice.is_some() == public_io.trusted_advice.is_empty() {
        return Err(ProverError::Unsupported {
            reason: "trusted-advice commitment presence disagrees with the trusted advice bytes",
        });
    }
    // The verifier's own input validation doubles as the prover's self-check
    // and produces the normalized `CheckedInputs` the preamble absorbs. The
    // zk axis is the compiled feature — the co-compiled verifier's
    // `SELECTED_ZK_CONFIG` flips with the same feature, so both sides always
    // agree.
    let checked = validate_inputs_from_parts(
        &preprocessing.verifier,
        public_io,
        config.trace_length,
        config.ram_K,
        config.trace_polynomial_order,
        config.one_hot_config,
        trusted_advice.is_some(),
        untrusted_advice_present,
        cfg!(feature = "zk"),
    )?;

    // The dominant-advice regime (an advice grid wider than every other
    // commitment-grid candidate) has no e2e coverage anywhere; guard it off
    // until an oracle-backed test exists. Committed-program candidates count
    // toward the grid width, so advice wider than the main matrix but inside
    // a committed candidate is fine.
    {
        let mut grid_without_advice =
            config.one_hot_config.committed_chunk_bits() + config.trace_length.ilog2() as usize;
        if let Some(candidates) = CommittedProgramCandidates::from_schedule(&checked.precommitted) {
            grid_without_advice = grid_without_advice
                .max(candidates.bytecode_chunk_vars)
                .max(candidates.program_image_vars);
        }
        let advice_dominates = |max_size: u64| advice_total_vars(max_size) > grid_without_advice;
        if (trusted_advice.is_some()
            && advice_dominates(public_io.memory_layout.max_trusted_advice_size))
            || (untrusted_advice_present
                && advice_dominates(public_io.memory_layout.max_untrusted_advice_size))
        {
            return Err(ProverError::Unsupported {
                reason: "dominant advice (advice grid wider than the main commitment grid) is not yet supported",
            });
        }
    }

    let mut transcript = T::new(b"Jolt");
    absorb_transcript_preamble(
        &checked,
        ProofTranscriptConfig {
            rw_config: config.rw_config,
            one_hot_config: config.one_hot_config,
            trace_polynomial_order: config.trace_polynomial_order,
        },
        &mut transcript,
    );

    let ids: Vec<JoltCommittedPolynomial> = witness
        .committed_order()?
        .into_iter()
        .filter(|id| {
            !matches!(
                id,
                JoltCommittedPolynomial::TrustedAdvice | JoltCommittedPolynomial::UntrustedAdvice
            )
        })
        .collect();
    // Stage-0 validation: every id the proof will request — the committed
    // set and each bundle's annotated set — must be servable by the backend
    // before witness generation starts.
    let requested = ids
        .iter()
        .map(|&id| JoltPolynomialId::Committed(id))
        .chain(InstructionReadRafWitness::annotated_ids())
        .chain(BytecodeReadRafWitness::annotated_ids());
    validate_servable(witness as &dyn JoltWitnessOracle<F>, requested)?;

    let grid = CommitmentGrid {
        total_vars: config.commitment_total_vars(
            &public_io.memory_layout,
            trusted_advice.is_some(),
            untrusted_advice_present,
            CommittedProgramCandidates::from_schedule(&checked.precommitted),
        ),
        log_t: config.trace_length.ilog2() as usize,
        log_k_chunk: config.one_hot_config.committed_chunk_bits(),
        order: config.trace_polynomial_order,
    };
    // The `commit_witness` kernel-seam span sits at this call boundary, not
    // on any one backend impl, so every `CommitWitness` backend inherits it
    // (the taxonomy advertises it as backend-neutral).
    let committed = tracing::info_span!(
        "commit_witness",
        columns = ids.len(),
        total_vars = grid.total_vars
    )
    .in_scope(|| {
        backend.commit.commit_witness(
            session,
            witness as &dyn RowSource,
            &ids,
            grid,
            &preprocessing.pcs_setup,
        )
    })?;
    let (commitments, mut hints) = assemble_commitments::<PCS>(committed)?;

    // The field-inline committed columns follow the base commitments and
    // precede the advice commitments — the same appended-extension position
    // `absorb_transcript_commitments` absorbs them in.
    #[cfg(feature = "field-inline")]
    let (commitments, field_inline_hints) = {
        let (field_inline, field_inline_hints) = commit_field_inline::<F, PCS>(
            backend,
            session,
            witness as &dyn JoltWitnessPlane<F>,
            grid,
            &preprocessing.pcs_setup,
        )?;
        (
            commitments.with_field_inline(field_inline),
            field_inline_hints,
        )
    };

    // The untrusted advice polynomial is committed at prove time in its OWN
    // balanced grid (its variable count comes from the memory layout's maximum
    // advice size, independent of the main grid); the trusted commitment
    // arrived from preprocessing.
    let untrusted_advice_commitment = if untrusted_advice_present {
        let advice_grid = CommitmentGrid {
            total_vars: advice_total_vars(public_io.memory_layout.max_untrusted_advice_size),
            log_t: 0,
            log_k_chunk: 0,
            // Advice grids always place cycle-major — see `CommitmentGrid`.
            order: TracePolynomialOrder::CycleMajor,
        };
        // Backend-neutral seam span, like `commit_witness` above.
        let advice = tracing::info_span!(
            "commit_advice",
            id = ?JoltCommittedPolynomial::UntrustedAdvice
        )
        .in_scope(|| {
            backend.commit.commit_advice(
                session,
                witness as &dyn JoltWitnessOracle<F>,
                JoltCommittedPolynomial::UntrustedAdvice,
                advice_grid,
                &preprocessing.pcs_setup,
            )
        })?;
        hints.push((advice.id, advice.hint));
        Some(advice.commitment)
    } else {
        None
    };
    if let Some(trusted) = trusted_advice {
        hints.push((JoltCommittedPolynomial::TrustedAdvice, trusted.hint.clone()));
    }
    // The committed-program hints ride from preprocessing (the chunk/image
    // commitments were produced there, before any proving).
    if let Some(committed) = &preprocessing.committed_program {
        let expected_chunks = checked
            .precommitted
            .bytecode
            .as_ref()
            .map_or(0, |layout| layout.chunk_count());
        if committed.bytecode_chunk_hints.len() != expected_chunks {
            return Err(ProverError::Unsupported {
                reason: "committed-program chunk hint count disagrees with the bytecode schedule",
            });
        }
        for (index, hint) in committed.bytecode_chunk_hints.iter().enumerate() {
            hints.push((JoltCommittedPolynomial::BytecodeChunk(index), hint.clone()));
        }
        hints.push((
            JoltCommittedPolynomial::ProgramImageInit,
            committed.program_image_hint.clone(),
        ));
    }

    absorb_transcript_commitments(
        &commitments,
        untrusted_advice_commitment.as_ref(),
        trusted_advice.map(|trusted| &trusted.commitment),
        &mut transcript,
    );
    if let Some(committed) = preprocessing.verifier.program.committed() {
        absorb_committed_program_commitments(
            &committed.bytecode_chunk_commitments,
            &committed.program_image_commitment,
            &mut transcript,
        );
    }

    Ok(Stage0Output {
        checked,
        transcript,
        commitments,
        untrusted_advice_commitment,
        hints,
        #[cfg(feature = "field-inline")]
        field_inline_hints,
    })
}

/// Commit the field-inline columns off the plane's field-inline oracle and
/// assemble the proof's FR commitment payload. Fails closed when the plane
/// serves no field-inline oracle: an FR-on build proves only FR-profile
/// witnesses (a non-FR guest has no honest FR columns to commit).
#[cfg(feature = "field-inline")]
#[expect(
    clippy::type_complexity,
    reason = "the wire payload paired with its opening hints"
)]
fn commit_field_inline<F, PCS>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    grid: CommitmentGrid,
    setup: &PCS::ProverSetup,
) -> Result<
    (
        FieldInlineCommitments<PCS::Output>,
        Vec<(FieldInlineCommittedPolynomial, PCS::OpeningHint)>,
    ),
    ProverError<F>,
>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let Some(field_inline) = witness.field_inline() else {
        return Err(ProverError::Unsupported {
            reason: "field-inline proving requires a witness plane serving the field-inline \
                     oracle (an FR-profile guest)",
        });
    };
    let ids = field_inline.committed_order();
    // Backend-neutral seam span, like `commit_witness`.
    let committed = tracing::info_span!("commit_field_inline_witness", columns = ids.len())
        .in_scope(|| {
            backend
                .commit
                .commit_field_inline_witness(session, witness, &ids, grid, setup)
        })?;

    let mut rd_inc = None;
    let mut hints = Vec::with_capacity(committed.len());
    for entry in committed {
        match entry.id {
            FieldInlineCommittedPolynomial::FieldRdInc => rd_inc = Some(entry.commitment),
        }
        hints.push((entry.id, entry.hint));
    }
    let Some(rd_inc) = rd_inc else {
        return Err(ProverError::InvariantViolation {
            reason: "witness did not produce the FieldRdInc commitment",
        });
    };
    Ok((
        FieldInlineCommitments {
            field_registers: FieldRegistersCommitments { rd_inc },
        },
        hints,
    ))
}

/// Split the kernel's flat id-ordered output into the proof's wire shape.
#[expect(
    clippy::type_complexity,
    reason = "the wire aggregate paired with its opening hints"
)]
fn assemble_commitments<PCS: CommitmentScheme>(
    committed: Vec<WitnessCommitment<PCS>>,
) -> Result<
    (
        JoltCommitments<PCS::Output>,
        Vec<(JoltCommittedPolynomial, PCS::OpeningHint)>,
    ),
    ProverError<PCS::Field>,
> {
    let mut rd_inc = None;
    let mut ram_inc = None;
    let mut instruction = Vec::new();
    let mut ram = Vec::new();
    let mut bytecode = Vec::new();
    let mut hints = Vec::with_capacity(committed.len());

    for entry in committed {
        let WitnessCommitment {
            id,
            commitment,
            hint,
        } = entry;
        match id {
            JoltCommittedPolynomial::RdInc => rd_inc = Some(commitment),
            JoltCommittedPolynomial::RamInc => ram_inc = Some(commitment),
            JoltCommittedPolynomial::InstructionRa(_) => instruction.push(commitment),
            JoltCommittedPolynomial::RamRa(_) => ram.push(commitment),
            JoltCommittedPolynomial::BytecodeRa(_) => bytecode.push(commitment),
            other => {
                return Err(ProverError::InvariantViolation {
                    reason: match other {
                        JoltCommittedPolynomial::TrustedAdvice
                        | JoltCommittedPolynomial::UntrustedAdvice => {
                            "advice polynomials are absorbed separately, not as main commitments"
                        }
                        _ => "precommitted polynomials are not main witness commitments",
                    },
                });
            }
        }
        hints.push((id, hint));
    }

    let (Some(rd_inc), Some(ram_inc)) = (rd_inc, ram_inc) else {
        return Err(ProverError::InvariantViolation {
            reason: "witness did not produce the RdInc/RamInc commitments",
        });
    };
    Ok((
        JoltCommitments::new(rd_inc, ram_inc, instruction, ram, bytecode),
        hints,
    ))
}

// Transparent mode only: the zk streaming finishes blind their commitments,
// so two independent commits of the same column are not comparable.
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_tests {
    use std::sync::Arc;

    use common::constants::RAM_START_ADDRESS;
    use jolt_claims::protocols::field_inline::FieldInlinePolynomialId;
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_dory::DoryCommitment;
    use jolt_dory::DoryScheme;
    use jolt_field::{Fr, Ring};
    use jolt_kernels::finish_streamed;
    use jolt_openings::{CommitmentScheme, StreamingCommitment};
    use jolt_program::execution::{JoltProgram, OwnedTrace, TraceOutput, TraceRow};
    use jolt_program::field_inline::{
        FieldEncodedValue, FieldInlineTraceData, FieldRegisterRead, FieldRegisterWrite,
    };
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_riscv::{
        FieldInlineOp, JoltInstructionKind, JoltInstructionProfile, JoltInstructionRow,
        NormalizedOperands, RV64IMAC_JOLT_FIELD_INLINE,
    };
    use jolt_transcript::LegacyBlake2bTranscript;
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

    use super::*;

    const ENTRY: u64 = RAM_START_ADDRESS;
    const LOG_T: usize = 2;

    fn instruction(
        instruction_kind: JoltInstructionKind,
        offset: usize,
        rd: Option<u8>,
        rs1: Option<u8>,
        rs2: Option<u8>,
        imm: i128,
    ) -> JoltInstructionRow {
        JoltInstructionRow {
            instruction_kind,
            address: ENTRY as usize + offset * 4,
            operands: NormalizedOperands { rd, rs1, rs2, imm },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    fn fr_backend(
        bytecode: Vec<JoltInstructionRow>,
        rows: Vec<TraceRow>,
    ) -> TraceBackend<OwnedTrace> {
        let profile: JoltInstructionProfile = RV64IMAC_JOLT_FIELD_INLINE;
        let program = Arc::new(JoltProgram::from_parts_with_profile(
            Vec::new(),
            bytecode.clone(),
            Vec::new(),
            ENTRY + 4,
            ENTRY,
            profile,
        ));
        let preprocessing = Arc::new(JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(bytecode, ENTRY, profile).unwrap(),
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: 1 << LOG_T,
        });
        TraceBackend::new(
            JoltVmWitnessConfig::new(
                LOG_T,
                64,
                JoltOneHotConfig {
                    log_k_chunk: 4,
                    lookups_ra_virtual_log_k_chunk: 16,
                },
            ),
            JoltVmWitnessInputs::new(
                &program,
                &preprocessing,
                TraceOutput::new(OwnedTrace::new(rows), Default::default(), None, None),
            ),
        )
    }

    fn enc(value: u64) -> FieldEncodedValue {
        FieldEncodedValue::from_u64(value)
    }

    fn field_row(instruction: JoltInstructionRow, data: FieldInlineTraceData) -> TraceRow {
        TraceRow {
            instruction,
            field_inline: Some(data.into()),
            ..TraceRow::default()
        }
    }

    /// Two field loads and a multiply: `FieldRdInc` = [13, 17, 221, 0].
    fn arithmetic_backend() -> TraceBackend<OwnedTrace> {
        let load_a = instruction(
            JoltInstructionKind::FIELD_LOAD_IMM,
            0,
            Some(1),
            None,
            None,
            13,
        );
        let load_b = instruction(
            JoltInstructionKind::FIELD_LOAD_IMM,
            1,
            Some(2),
            None,
            None,
            17,
        );
        let mul = instruction(
            JoltInstructionKind::FIELD_MUL,
            2,
            Some(3),
            Some(1),
            Some(2),
            0,
        );
        let rows = vec![
            field_row(
                load_a,
                FieldInlineTraceData {
                    op: Some(FieldInlineOp::LoadImm),
                    rd: Some(FieldRegisterWrite {
                        register: 1,
                        pre_value: enc(0),
                        post_value: enc(13),
                    }),
                    ..FieldInlineTraceData::default()
                },
            ),
            field_row(
                load_b,
                FieldInlineTraceData {
                    op: Some(FieldInlineOp::LoadImm),
                    rd: Some(FieldRegisterWrite {
                        register: 2,
                        pre_value: enc(0),
                        post_value: enc(17),
                    }),
                    ..FieldInlineTraceData::default()
                },
            ),
            field_row(
                mul,
                FieldInlineTraceData {
                    op: Some(FieldInlineOp::Mul),
                    rs1: Some(FieldRegisterRead {
                        register: 1,
                        value: enc(13),
                    }),
                    rs2: Some(FieldRegisterRead {
                        register: 2,
                        value: enc(17),
                    }),
                    rd: Some(FieldRegisterWrite {
                        register: 3,
                        pre_value: enc(0),
                        post_value: enc(221),
                    }),
                    product: Some(enc(221)),
                    ..FieldInlineTraceData::default()
                },
            ),
        ];
        fr_backend(vec![load_a, load_b, mul], rows)
    }

    /// An FR-profile guest that executes zero FR instructions.
    fn no_fr_instruction_backend() -> TraceBackend<OwnedTrace> {
        let addi = instruction(JoltInstructionKind::ADDI, 0, Some(1), Some(2), None, 3);
        let rows = vec![TraceRow {
            instruction: addi,
            ..TraceRow::default()
        }];
        fr_backend(vec![addi], rows)
    }

    fn grid() -> CommitmentGrid {
        CommitmentGrid {
            total_vars: 4 + LOG_T,
            log_t: LOG_T,
            log_k_chunk: 4,
            order: TracePolynomialOrder::CycleMajor,
        }
    }

    /// Commit `values` directly through the streaming PCS calls the dense
    /// grid columns use — the placement spec the kernel must match.
    fn direct_dense_commitment(
        values: &[Fr],
        setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
    ) -> DoryCommitment {
        let mut partial = <DoryScheme as StreamingCommitment>::begin(setup);
        for row in values.chunks(grid().num_columns()) {
            <DoryScheme as StreamingCommitment>::feed(&mut partial, row, setup);
        }
        finish_streamed::<DoryScheme>(partial, setup).0
    }

    /// The prover attaches the FR payload and absorbs it through the
    /// verifier's own `absorb_transcript_commitments` — pinned by asserting
    /// the payload is `Some`, that both sides' absorbs agree byte-for-byte
    /// (equal challenge streams), and that stripping the payload diverges
    /// (the FR commitment is Fiat-Shamir-bound).
    #[test]
    fn stage0_attaches_and_absorbs_the_field_inline_payload() {
        let witness = arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let setup = DoryScheme::setup_prover(grid().total_vars);

        let ids: Vec<JoltCommittedPolynomial> = witness.committed_polynomial_order().unwrap();
        let committed = backend
            .commit
            .commit_witness(
                &mut session,
                &witness as &dyn JoltWitnessPlane<Fr>,
                &ids,
                grid(),
                &setup,
            )
            .unwrap();
        let (commitments, _hints) = assemble_commitments::<DoryScheme>(committed).unwrap();

        let (field_inline, field_inline_hints) = commit_field_inline::<Fr, DoryScheme>(
            &backend,
            &mut session,
            &witness as &dyn JoltWitnessPlane<Fr>,
            grid(),
            &setup,
        )
        .unwrap();
        let commitments = commitments.with_field_inline(field_inline);

        assert!(commitments.field_inline.is_some());
        assert_eq!(
            field_inline_hints
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>(),
            vec![FieldInlineCommittedPolynomial::FieldRdInc]
        );

        // The FR commitment is the dense trace-domain column committed with
        // the same placement as the jolt increment columns.
        let column = witness
            .field_inline_witness()
            .unwrap()
            .oracle_table::<Fr>(FieldInlinePolynomialId::Committed(
                FieldInlineCommittedPolynomial::FieldRdInc,
            ))
            .unwrap();
        assert_eq!(
            column,
            [13u64, 17, 221, 0].map(Fr::from_u64).to_vec(),
            "fixture column"
        );
        assert_eq!(
            commitments
                .field_inline
                .as_ref()
                .unwrap()
                .field_registers
                .rd_inc,
            direct_dense_commitment(&column, &setup)
        );

        // Byte-for-byte absorb parity between the prover-side call and the
        // verifier's own absorb (the same shared fn on the same payload).
        let mut prover_transcript = LegacyBlake2bTranscript::<Fr>::new(b"Jolt");
        absorb_transcript_commitments(&commitments, None, None, &mut prover_transcript);
        let mut verifier_transcript = LegacyBlake2bTranscript::<Fr>::new(b"Jolt");
        jolt_verifier::absorb_transcript_commitments(
            &commitments,
            None,
            None,
            &mut verifier_transcript,
        );
        assert_eq!(
            prover_transcript.challenge(),
            verifier_transcript.challenge()
        );

        let mut stripped = commitments.clone();
        stripped.field_inline = None;
        let mut stripped_transcript = LegacyBlake2bTranscript::<Fr>::new(b"Jolt");
        absorb_transcript_commitments(&stripped, None, None, &mut stripped_transcript);
        assert_ne!(
            prover_transcript.challenge(),
            stripped_transcript.challenge(),
            "the FR payload must be Fiat-Shamir-bound"
        );
    }

    /// Zero-short-circuit sanity: an FR-profile guest with no FR instructions
    /// still serves the FR committed order and commits the all-zero column.
    #[test]
    fn fr_guest_with_no_fr_instructions_commits_the_zero_column() {
        let witness = no_fr_instruction_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let setup = DoryScheme::setup_prover(grid().total_vars);

        let provider = witness.field_inline_witness().unwrap();
        assert_eq!(
            provider.committed_order(),
            vec![FieldInlineCommittedPolynomial::FieldRdInc]
        );
        let column = provider
            .oracle_table::<Fr>(FieldInlinePolynomialId::Committed(
                FieldInlineCommittedPolynomial::FieldRdInc,
            ))
            .unwrap();
        assert_eq!(column, vec![Fr::from_u64(0); 1 << LOG_T]);

        let (field_inline, _hints) = commit_field_inline::<Fr, DoryScheme>(
            &backend,
            &mut session,
            &witness as &dyn JoltWitnessPlane<Fr>,
            grid(),
            &setup,
        )
        .unwrap();
        assert_eq!(
            field_inline.field_registers.rd_inc,
            direct_dense_commitment(&column, &setup)
        );
    }

    /// D1 fail-closed: a plane without the field-inline oracle cannot start
    /// an FR-on proof.
    #[test]
    fn stage0_fails_closed_without_the_field_inline_oracle() {
        let witness = arithmetic_backend();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let setup = DoryScheme::setup_prover(grid().total_vars);

        let result = commit_field_inline::<Fr, DoryScheme>(
            &backend,
            &mut session,
            &witness as &dyn JoltWitnessPlane<Fr>,
            grid(),
            &setup,
        );
        assert!(matches!(result, Err(ProverError::Unsupported { .. })));
    }
}
