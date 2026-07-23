//! The packed reconstruction phase, strictly after stage 7: one batched
//! sumcheck settling every virtualized word/chunk claim against its committed
//! one-hot decomposition, producing the packed final claims for the advice
//! byte columns and the `ProgramOneHot` lane columns. Members in canonical
//! commitment-object order: untrusted advice, trusted advice, bytecode
//! chunks, program image. The phase is entirely absent (zero transcript
//! interaction) when no advice is present and the program is full.
//!
//! Each member consumes the *completed* claim of its base reduction — the
//! stage-7 address-phase output, or the stage-6b cycle-phase output when the
//! reduction had no address rounds.

use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::{
    committed_lane_vars, BYTECODE_LANE_LAYOUT,
};
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::lattice::geometry::{
    byte_decode_weight, selector_block_weight, word_byte_num_vars, BYTE_BITS,
};
use jolt_claims::protocols::jolt::lattice::relations::advice_reconstruction::{
    AdviceReconstructionDimensions, TrustedAdviceReconstruction as TrustedSymbolic,
    TrustedAdviceReconstructionInputClaims, TrustedAdviceReconstructionOutputClaims,
    UntrustedAdviceReconstruction as UntrustedSymbolic, UntrustedAdviceReconstructionChallenges,
    UntrustedAdviceReconstructionInputClaims, UntrustedAdviceReconstructionOutputClaims,
};
use jolt_claims::protocols::jolt::lattice::relations::bytecode_reconstruction::{
    BytecodeChunkReconstruction as BytecodeSymbolic, BytecodeChunkReconstructionChallenges,
    BytecodeChunkReconstructionInputClaims, BytecodeChunkReconstructionOutputClaims,
    BytecodeReconstructionDimensions,
};
use jolt_claims::protocols::jolt::lattice::relations::program_image_reconstruction::{
    ProgramImageReconstruction as ProgramImageSymbolic, ProgramImageReconstructionInputClaims,
    ProgramImageReconstructionOutputClaims,
};
use jolt_claims::protocols::jolt::{
    BytecodeChunkReconstructionPublic, BytecodeRegisterLane, JoltAdviceKind, JoltDerivedId,
    JoltRelationId, ProgramImageReconstructionPublic, TrustedAdviceReconstructionPublic,
    UntrustedAdviceReconstructionPublic,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::{Field, FixedByteSize};
use jolt_poly::math::Math;
use jolt_poly::{eq_index_msb, try_eq_mle};
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;

use crate::stages::relations::{ConcreteSumcheck, SumcheckBatch};
use crate::stages::stage6b::Stage6bClearOutput;
use crate::stages::stage7::Stage7ClearOutput;
use crate::verifier::CheckedInputs;
use crate::VerifierError;

fn public_input_failed(stage: JoltRelationId, reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage,
        reason: reason.to_string(),
    }
}

fn untrusted_public_failed(reason: impl ToString) -> VerifierError {
    public_input_failed(JoltRelationId::UntrustedAdviceReconstruction, reason)
}

fn trusted_public_failed(reason: impl ToString) -> VerifierError {
    public_input_failed(JoltRelationId::TrustedAdviceReconstruction, reason)
}

fn bytecode_public_failed(reason: impl ToString) -> VerifierError {
    public_input_failed(JoltRelationId::BytecodeChunkReconstruction, reason)
}

fn image_public_failed(reason: impl ToString) -> VerifierError {
    public_input_failed(JoltRelationId::ProgramImageReconstruction, reason)
}

/// The single-leg decode publics shared by the trusted-advice and
/// program-image instances: [`byte_decode_weight`] at the bound
/// `(byte ‖ place)` prefix of the produced opening point.
fn byte_decode_leg<F: Field>(
    opening_point: &[F],
    bound: usize,
    fail: fn(&'static str) -> VerifierError,
) -> Result<F, VerifierError> {
    if opening_point.len() < bound {
        return Err(fail("cell point is below the (byte ‖ place) prefix"));
    }
    let (r_byte, r_place) = opening_point[..bound].split_at(BYTE_BITS);
    Ok(byte_decode_weight(r_byte, r_place))
}

/// The untrusted advice reconstruction: booleanity + hamming + decode legs
/// over the full `(byte ‖ place ‖ word)` cell domain.
#[derive(Clone)]
pub struct UntrustedAdviceReconstructionInstance<F: Field> {
    symbolic: UntrustedSymbolic,
    /// The fresh reference point the booleanity/hamming kernels compare
    /// against, drawn over the full cell domain before the instance gammas.
    r_reference: Vec<F>,
}

impl<F: Field> ConcreteSumcheck<F> for UntrustedAdviceReconstructionInstance<F> {
    type Symbolic = UntrustedSymbolic;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &UntrustedAdviceReconstructionInputClaims<Vec<F>>,
    ) -> Result<UntrustedAdviceReconstructionOutputClaims<Vec<F>>, VerifierError> {
        // Word variables bind first (low bits of the `(byte ‖ place ‖ word)`
        // cell order), so the reversed sumcheck point is msb-first.
        Ok(UntrustedAdviceReconstructionOutputClaims {
            bytes: sumcheck_point.iter().rev().copied().collect(),
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        input_points: &UntrustedAdviceReconstructionInputClaims<Vec<F>>,
        output_points: &UntrustedAdviceReconstructionOutputClaims<Vec<F>>,
        _challenges: &UntrustedAdviceReconstructionChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::UntrustedAdviceReconstruction(public) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let opening_point = output_points.bytes();
        // The `(byte ‖ place)` prefix of the cell layout; the word variables
        // follow it.
        let byte_place_split = word_byte_num_vars(0);
        if opening_point.len() < byte_place_split || self.r_reference.len() != opening_point.len() {
            return Err(untrusted_public_failed(format!(
                "cell point has {} variables, reference has {}",
                opening_point.len(),
                self.r_reference.len()
            )));
        }
        let (byte_place, r_word) = opening_point.split_at(byte_place_split);
        let (r_byte, r_place) = byte_place.split_at(BYTE_BITS);
        match public {
            UntrustedAdviceReconstructionPublic::EqBytePlaceWord => {
                try_eq_mle(opening_point, &self.r_reference).map_err(untrusted_public_failed)
            }
            UntrustedAdviceReconstructionPublic::EqPlaceWord => {
                try_eq_mle(&opening_point[BYTE_BITS..], &self.r_reference[BYTE_BITS..])
                    .map_err(untrusted_public_failed)
            }
            UntrustedAdviceReconstructionPublic::ByteDecode => {
                Ok(byte_decode_weight(r_byte, r_place))
            }
            UntrustedAdviceReconstructionPublic::EqWord => {
                try_eq_mle(r_word, input_points.word()).map_err(untrusted_public_failed)
            }
        }
    }
}

/// The trusted advice reconstruction: the decode leg alone over the
/// `(byte ‖ place)` variables, the word point fixed by the incoming claim.
#[derive(Clone)]
pub struct TrustedAdviceReconstructionInstance<F: Field> {
    symbolic: TrustedSymbolic,
    _field: core::marker::PhantomData<F>,
}

impl<F: Field> ConcreteSumcheck<F> for TrustedAdviceReconstructionInstance<F> {
    type Symbolic = TrustedSymbolic;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        input_points: &TrustedAdviceReconstructionInputClaims<Vec<F>>,
    ) -> Result<TrustedAdviceReconstructionOutputClaims<Vec<F>>, VerifierError> {
        let bound = sumcheck_point.iter().rev().copied();
        Ok(TrustedAdviceReconstructionOutputClaims {
            bytes: bound.chain(input_points.word().iter().copied()).collect(),
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &TrustedAdviceReconstructionInputClaims<Vec<F>>,
        output_points: &TrustedAdviceReconstructionOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::TrustedAdviceReconstruction(
            TrustedAdviceReconstructionPublic::ByteDecode,
        ) = id
        else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        byte_decode_leg(
            output_points.bytes(),
            ConcreteSumcheck::<F>::symbolic(self).rounds(),
            trusted_public_failed,
        )
    }
}

/// The bytecode chunk reconstruction: rebuilds every chunk claim from the
/// per-lane sub-columns.
///
/// Leg embedding contract (shared with the prover): the batch binds the
/// widest byte lane's `(byte ‖ place)` variables; every narrower column's own
/// variables are the LOW-order tail of that vector (bound by the first
/// rounds, interpreted by the column's own layout), and its missing high
/// coordinates are zero-pinned through the leg's public — the column grid is
/// zero-extended, so its bound value factors as
/// `Π_missing (1 − v_i) · column(v_own ‖ r_row)`, with the `Π` folded into
/// the derived and the claim landing at the column's own packed-slot point.
#[derive(Clone)]
pub struct BytecodeChunkReconstructionInstance<F: Field> {
    symbolic: BytecodeSymbolic,
    dimensions: BytecodeReconstructionDimensions,
    /// The lane half of the completed chunk claims' shared point.
    r_lane: Vec<F>,
    /// The row half; every produced lane opening is suffixed with it.
    r_row: Vec<F>,
}

impl<F: Field> BytecodeChunkReconstructionInstance<F> {
    fn own_vars(&self) -> BytecodeLegVars {
        BytecodeLegVars {
            total: SymbolicSumcheck::rounds(&self.symbolic),
            selector: REGISTER_ADDRESS_BITS,
            lookup: (BYTECODE_LANE_LAYOUT.raf_flag_idx - BYTECODE_LANE_LAYOUT.lookup_start).log_2(),
            pc: BYTE_BITS + 8usize.log_2(),
            imm: BYTE_BITS + self.dimensions.imm_byte_width.log_2(),
        }
    }
}

struct BytecodeLegVars {
    total: usize,
    selector: usize,
    lookup: usize,
    pc: usize,
    imm: usize,
}

impl BytecodeLegVars {
    /// The leg's own point: the low-order `own` tail of the bound vector.
    fn leg_point<'a, F>(&self, bound: &'a [F], own: usize) -> &'a [F] {
        &bound[self.total - own..]
    }

    /// The zero-pin factor of a leg's missing high coordinates:
    /// `eq(v_missing, 0) = Π (1 − v_i)`.
    fn zero_pin<F: Field>(&self, bound: &[F], own: usize) -> F {
        eq_index_msb(&bound[..self.total - own], 0)
    }
}

impl<F: Field> ConcreteSumcheck<F> for BytecodeChunkReconstructionInstance<F> {
    type Symbolic = BytecodeSymbolic;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &BytecodeChunkReconstructionInputClaims<Vec<F>>,
    ) -> Result<BytecodeChunkReconstructionOutputClaims<Vec<F>>, VerifierError> {
        let vars = self.own_vars();
        let bound = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        let chunks = self.dimensions.chunks;
        let leg = |own: usize| [vars.leg_point(&bound, own), self.r_row.as_slice()].concat();
        Ok(BytecodeChunkReconstructionOutputClaims {
            register_selectors: vec![leg(vars.selector); chunks * BytecodeRegisterLane::ALL.len()],
            circuit_flags: vec![self.r_row.clone(); chunks * jolt_riscv::NUM_CIRCUIT_FLAGS],
            instruction_flags: vec![self.r_row.clone(); chunks * jolt_riscv::NUM_INSTRUCTION_FLAGS],
            lookup_selectors: vec![leg(vars.lookup); chunks],
            raf_flags: vec![self.r_row.clone(); chunks],
            pc_bytes: vec![leg(vars.pc); chunks],
            imm_bytes: vec![leg(vars.imm); chunks],
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &BytecodeChunkReconstructionInputClaims<Vec<F>>,
        output_points: &BytecodeChunkReconstructionOutputClaims<Vec<F>>,
        _challenges: &BytecodeChunkReconstructionChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::BytecodeChunkReconstruction(public) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let vars = self.own_vars();
        // Recover the full bound vector from the widest byte lane's produced
        // point (its own variables cover every bound round).
        let widest = if vars.imm >= vars.pc {
            (output_points.imm_bytes.first(), vars.imm)
        } else {
            (output_points.pc_bytes.first(), vars.pc)
        };
        let (Some(widest_point), widest_vars) = widest else {
            return Err(bytecode_public_failed(
                "reconstruction produced no byte-lane openings",
            ));
        };
        debug_assert_eq!(widest_vars, vars.total);
        let bound = widest_point.get(..vars.total).ok_or_else(|| {
            bytecode_public_failed("byte-lane opening point is below the bound prefix")
        })?;
        let layout = BYTECODE_LANE_LAYOUT;
        let register_count = 1usize << REGISTER_ADDRESS_BITS;
        Ok(match public {
            BytecodeChunkReconstructionPublic::RegisterSelectorWeight(lane) => {
                let block_start = match lane {
                    BytecodeRegisterLane::Rs1 => layout.rs1_start,
                    BytecodeRegisterLane::Rs2 => layout.rs2_start,
                    BytecodeRegisterLane::Rd => layout.rd_start,
                };
                selector_block_weight(
                    &self.r_lane,
                    block_start,
                    vars.leg_point(bound, vars.selector),
                    register_count,
                ) * vars.zero_pin(bound, vars.selector)
            }
            BytecodeChunkReconstructionPublic::LaneWeight(lane) => {
                eq_index_msb::<F>(&self.r_lane, *lane as u128) * vars.zero_pin(bound, 0)
            }
            BytecodeChunkReconstructionPublic::LookupSelectorWeight => {
                selector_block_weight(
                    &self.r_lane,
                    layout.lookup_start,
                    vars.leg_point(bound, vars.lookup),
                    layout.raf_flag_idx - layout.lookup_start,
                ) * vars.zero_pin(bound, vars.lookup)
            }
            BytecodeChunkReconstructionPublic::PcByteDecode => {
                let leg = vars.leg_point(bound, vars.pc);
                let (r_byte, r_place) = leg.split_at(BYTE_BITS);
                eq_index_msb::<F>(&self.r_lane, layout.unexp_pc_idx as u128)
                    * byte_decode_weight(r_byte, r_place)
                    * vars.zero_pin(bound, vars.pc)
            }
            BytecodeChunkReconstructionPublic::ImmByteDecode => {
                let leg = vars.leg_point(bound, vars.imm);
                let (r_byte, r_place) = leg.split_at(BYTE_BITS);
                eq_index_msb::<F>(&self.r_lane, layout.imm_idx as u128)
                    * byte_decode_weight(r_byte, r_place)
                    * vars.zero_pin(bound, vars.imm)
            }
        })
    }
}

/// The program-image reconstruction: the trusted-advice decode shape over the
/// program image byte column.
#[derive(Clone)]
pub struct ProgramImageReconstructionInstance<F: Field> {
    symbolic: ProgramImageSymbolic,
    _field: core::marker::PhantomData<F>,
}

impl<F: Field> ConcreteSumcheck<F> for ProgramImageReconstructionInstance<F> {
    type Symbolic = ProgramImageSymbolic;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        input_points: &ProgramImageReconstructionInputClaims<Vec<F>>,
    ) -> Result<ProgramImageReconstructionOutputClaims<Vec<F>>, VerifierError> {
        let bound = sumcheck_point.iter().rev().copied();
        Ok(ProgramImageReconstructionOutputClaims {
            bytes: bound.chain(input_points.word().iter().copied()).collect(),
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &ProgramImageReconstructionInputClaims<Vec<F>>,
        output_points: &ProgramImageReconstructionOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::ProgramImageReconstruction(ProgramImageReconstructionPublic::ByteDecode) =
            id
        else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        byte_decode_leg(
            output_points.bytes(),
            ConcreteSumcheck::<F>::symbolic(self).rounds(),
            image_public_failed,
        )
    }
}

/// The reconstruction batch, members in canonical commitment-object order.
/// Each is present exactly when its object exists in the public shape.
#[derive(SumcheckBatch)]
#[sumcheck_batch(crate = "crate")]
pub struct ReconstructionSumchecks<F: Field> {
    pub untrusted_advice: Option<UntrustedAdviceReconstructionInstance<F>>,
    pub trusted_advice: Option<TrustedAdviceReconstructionInstance<F>>,
    pub bytecode: Option<BytecodeChunkReconstructionInstance<F>>,
    pub program_image: Option<ProgramImageReconstructionInstance<F>>,
}

pub struct ReconstructionClearOutput<F: Field> {
    pub output_values: ReconstructionOutputClaims<F>,
    pub output_points: ReconstructionOutputPoints<F>,
}

impl<F: Field> ReconstructionClearOutput<F> {
    fn empty() -> Self {
        Self {
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
        }
    }
}

/// A completed claim of a base precommitted reduction: the stage-7
/// address-phase output when the reduction ran one, else its stage-6b
/// cycle-phase terminus.
struct CompletedClaim<F> {
    value: F,
    point: Vec<F>,
}

/// The address-phase-else-cycle-terminus fallback shared by every completed
/// claim: take the stage-7 pair when the address phase ran, else the stage-6b
/// pair, else fail with `error`.
fn completed<F: Field, V>(
    address_phase: Option<(V, &[F])>,
    cycle_phase: Option<(V, &[F])>,
    error: impl FnOnce() -> VerifierError,
) -> Result<(V, Vec<F>), VerifierError> {
    address_phase
        .or(cycle_phase)
        .map(|(value, point)| (value, point.to_vec()))
        .ok_or_else(error)
}

fn completed_advice_claim<F: Field>(
    kind: JoltAdviceKind,
    stage6b: &Stage6bClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
) -> Result<CompletedClaim<F>, VerifierError> {
    let address_value = match kind {
        JoltAdviceKind::Trusted => stage7
            .output_values
            .trusted_advice
            .as_ref()
            .map(|claims| claims.trusted),
        JoltAdviceKind::Untrusted => stage7
            .output_values
            .untrusted_advice
            .as_ref()
            .map(|claims| claims.untrusted),
    };
    let (value, point) = completed(
        address_value.zip(stage7.output_points.advice_point(kind)),
        stage6b
            .output_values
            .advice_cycle_phase_claim(kind)
            .zip(stage6b.output_points.advice_cycle_phase_opening_point(kind)),
        || {
            let message =
                format!("no completed {kind:?} advice claim for the reconstruction phase");
            match kind {
                JoltAdviceKind::Trusted => trusted_public_failed(message),
                JoltAdviceKind::Untrusted => untrusted_public_failed(message),
            }
        },
    )?;
    Ok(CompletedClaim { value, point })
}

fn completed_chunk_claims<F: Field>(
    stage6b: &Stage6bClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
) -> Result<(Vec<F>, Vec<F>), VerifierError> {
    completed(
        stage7
            .output_values
            .bytecode_address_phase
            .as_ref()
            .map(|claims| claims.chunks.clone())
            .zip(stage7.output_points.bytecode_point()),
        stage6b
            .output_values
            .bytecode_reduction
            .as_ref()
            .filter(|claims| !claims.chunks.is_empty())
            .map(|claims| claims.chunks.clone())
            .zip(stage6b.output_points.bytecode_reduction_opening_point()),
        || {
            bytecode_public_failed(
                "no completed bytecode chunk claims for the reconstruction phase",
            )
        },
    )
}

fn completed_program_image_claim<F: Field>(
    stage6b: &Stage6bClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
) -> Result<CompletedClaim<F>, VerifierError> {
    let (value, point) = completed(
        stage7
            .output_values
            .program_image_address_phase
            .as_ref()
            .map(|claims| claims.program_image)
            .zip(stage7.output_points.program_image_point()),
        stage6b
            .output_values
            .program_image_reduction
            .as_ref()
            .map(|claims| claims.program_image)
            .zip(stage6b.output_points.program_image_opening_point()),
        || image_public_failed("no completed program-image claim for the reconstruction phase"),
    )?;
    Ok(CompletedClaim { value, point })
}

pub fn verify<F, C, T>(
    checked: &CheckedInputs,
    sumcheck_proof: Option<&SumcheckProof<F, C>>,
    claims: &ReconstructionOutputClaims<F>,
    transcript: &mut T,
    stage6b: &Stage6bClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
) -> Result<ReconstructionClearOutput<F>, VerifierError>
where
    F: Field,
    C: Clone + jolt_transcript::AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let untrusted_layout = checked.precommitted.untrusted_advice.as_ref();
    let trusted_layout = checked.precommitted.trusted_advice.as_ref();
    let bytecode_layout = checked.precommitted.bytecode.as_ref();
    let image_layout = checked.precommitted.program_image.as_ref();

    let phase_runs = untrusted_layout.is_some()
        || trusted_layout.is_some()
        || bytecode_layout.is_some()
        || image_layout.is_some();
    if !phase_runs {
        // Fail-closed: no advice and a full program in the public shape means
        // no reconstruction anywhere in the proof.
        if sumcheck_proof.is_some()
            || claims.untrusted_advice.is_some()
            || claims.trusted_advice.is_some()
            || claims.bytecode.is_some()
            || claims.program_image.is_some()
        {
            return Err(public_input_failed(
                JoltRelationId::UntrustedAdviceReconstruction,
                "reconstruction phase present without advice or a committed program",
            ));
        }
        return Ok(ReconstructionClearOutput::empty());
    }
    let Some(sumcheck_proof) = sumcheck_proof else {
        return Err(public_input_failed(
            JoltRelationId::UntrustedAdviceReconstruction,
            "advice or a committed program is present but the reconstruction phase is missing",
        ));
    };

    let advice_word_vars = |layout: &jolt_claims::protocols::jolt::AdviceClaimReductionLayout| {
        layout.advice_shape().total_vars()
    };

    // The untrusted booleanity/hamming reference point is drawn over the full
    // cell domain before every instance gamma (the generated draws follow).
    let untrusted = untrusted_layout
        .map(|layout| -> Result<_, VerifierError> {
            let word_vars = advice_word_vars(layout);
            let word = completed_advice_claim(JoltAdviceKind::Untrusted, stage6b, stage7)?;
            if word.point.len() != word_vars {
                return Err(public_input_failed(
                    JoltRelationId::UntrustedAdviceReconstruction,
                    format!(
                        "completed untrusted advice claim has {} variables, expected {word_vars}",
                        word.point.len()
                    ),
                ));
            }
            let instance = UntrustedAdviceReconstructionInstance {
                symbolic: UntrustedSymbolic::new(AdviceReconstructionDimensions { word_vars }),
                r_reference: transcript.challenge_vector(word_byte_num_vars(word_vars)),
            };
            Ok((instance, word))
        })
        .transpose()?;
    let trusted = trusted_layout
        .map(|layout| -> Result<_, VerifierError> {
            let word_vars = advice_word_vars(layout);
            let word = completed_advice_claim(JoltAdviceKind::Trusted, stage6b, stage7)?;
            if word.point.len() != word_vars {
                return Err(public_input_failed(
                    JoltRelationId::TrustedAdviceReconstruction,
                    format!(
                        "completed trusted advice claim has {} variables, expected {word_vars}",
                        word.point.len()
                    ),
                ));
            }
            let instance = TrustedAdviceReconstructionInstance {
                symbolic: TrustedSymbolic::new(()),
                _field: core::marker::PhantomData,
            };
            Ok((instance, word))
        })
        .transpose()?;
    let bytecode = bytecode_layout
        .map(|layout| -> Result<_, VerifierError> {
            let (chunk_values, shared_point) = completed_chunk_claims(stage6b, stage7)?;
            if chunk_values.len() != layout.chunk_count() {
                return Err(public_input_failed(
                    JoltRelationId::BytecodeChunkReconstruction,
                    format!(
                        "completed chunk claim count mismatch: expected {}, got {}",
                        layout.chunk_count(),
                        chunk_values.len()
                    ),
                ));
            }
            let lane_vars = committed_lane_vars();
            if shared_point.len() < lane_vars {
                return Err(public_input_failed(
                    JoltRelationId::BytecodeChunkReconstruction,
                    format!(
                        "bytecode chunk point has {} variables, below the {lane_vars}-variable lane prefix",
                        shared_point.len()
                    ),
                ));
            }
            let (r_lane, r_row) = shared_point.split_at(lane_vars);
            let dimensions = BytecodeReconstructionDimensions {
                chunks: chunk_values.len(),
                imm_byte_width: <F as FixedByteSize>::NUM_BYTES,
            };
            let instance = BytecodeChunkReconstructionInstance {
                symbolic: BytecodeSymbolic::new(dimensions),
                dimensions,
                r_lane: r_lane.to_vec(),
                r_row: r_row.to_vec(),
            };
            Ok((instance, chunk_values, shared_point))
        })
        .transpose()?;
    let program_image = image_layout
        .map(|_| -> Result<_, VerifierError> {
            let word = completed_program_image_claim(stage6b, stage7)?;
            let instance = ProgramImageReconstructionInstance {
                symbolic: ProgramImageSymbolic::new(()),
                _field: core::marker::PhantomData,
            };
            Ok((instance, word))
        })
        .transpose()?;

    let input_values = ReconstructionInputClaims {
        untrusted_advice: untrusted
            .as_ref()
            .map(|(_, word)| UntrustedAdviceReconstructionInputClaims { word: word.value }),
        trusted_advice: trusted
            .as_ref()
            .map(|(_, word)| TrustedAdviceReconstructionInputClaims { word: word.value }),
        bytecode: bytecode
            .as_ref()
            .map(|(_, chunks, _)| BytecodeChunkReconstructionInputClaims {
                chunks: chunks.clone(),
            }),
        program_image: program_image
            .as_ref()
            .map(|(_, word)| ProgramImageReconstructionInputClaims { word: word.value }),
    };
    let input_points = ReconstructionInputPoints {
        untrusted_advice: untrusted.as_ref().map(|(_, word)| {
            UntrustedAdviceReconstructionInputClaims {
                word: word.point.clone(),
            }
        }),
        trusted_advice: trusted
            .as_ref()
            .map(|(_, word)| TrustedAdviceReconstructionInputClaims {
                word: word.point.clone(),
            }),
        bytecode: bytecode.as_ref().map(|(_, chunks, shared_point)| {
            BytecodeChunkReconstructionInputClaims {
                chunks: vec![shared_point.clone(); chunks.len()],
            }
        }),
        program_image: program_image.as_ref().map(|(_, word)| {
            ProgramImageReconstructionInputClaims {
                word: word.point.clone(),
            }
        }),
    };

    let sumchecks = ReconstructionSumchecks {
        untrusted_advice: untrusted.map(|(instance, _)| instance),
        trusted_advice: trusted.map(|(instance, _)| instance),
        bytecode: bytecode.map(|(instance, _, _)| instance),
        program_image: program_image.map(|(instance, _)| instance),
    };

    let challenges = sumchecks.draw_challenges(transcript)?;
    sumchecks.validate_output_claims(claims)?;

    let output_points = sumchecks.verify_clear(
        &input_values,
        &input_points,
        &challenges,
        claims,
        sumcheck_proof,
        transcript,
        7,
    )?;

    sumchecks.append_output_claims(transcript, claims);

    Ok(ReconstructionClearOutput {
        output_values: claims.clone(),
        output_points,
    })
}
