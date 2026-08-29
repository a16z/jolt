//! The stage 6a bytecode read-RAF address-phase sumcheck instance.
//!
//! The **address phase** binds the `log_k` address variables. Its input claim is
//! the gamma-folded bind of the entire prior proof (every stage-1..5 opening plus
//! the two PC claims), wired by
//! [`bytecode_read_raf_address_phase_input_values_from_upstream`]. Its output is
//! the staged `BytecodeReadRafAddrClaim` intermediate (consumed by the stage-6b
//! cycle phase) followed, in committed mode, by the `BytecodeValClaim` openings.
//!
//! Under the `akita` feature the symbolic swaps to the lattice address phase,
//! whose input fold additionally consumes the four reduced `Inc` claims

#[cfg(feature = "field-inline")]
use std::sync::OnceLock;

#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::relations;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
pub use jolt_claims::protocols::jolt::relations::bytecode::{
    BytecodeReadRafAddressPhaseInputClaims, BytecodeReadRafAddressPhaseOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::{
        bytecode::BytecodeReadRafDimensions, claim_reductions::bytecode as bytecode_reduction,
        dimensions::REGISTER_ADDRESS_BITS,
    },
    JoltOpeningId, JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;

#[cfg(feature = "field-inline")]
use super::field_inline::{FieldInlineBytecodeReadRafGeometry, FieldInlineBytecodeReadRafInputs};
#[cfg(feature = "field-inline")]
use crate::stages::relations::SumcheckInputClaims;
use crate::stages::relations::{ConcreteSumcheck, SumcheckInputPoints};
use crate::stages::stage2::Stage2BatchOutputPoints;
use crate::stages::stage3::outputs::Stage3OutputPoints;
use crate::stages::stage4::outputs::Stage4OutputPoints;
use crate::stages::stage5::outputs::Stage5OutputPoints;
use crate::stages::stage6_checked_split;
use crate::stages::{
    stage1::Stage1BatchOutputClaims, stage2::Stage2BatchOutputClaims, stage3::Stage3OutputClaims,
    stage4::Stage4OutputClaims, stage5::Stage5OutputClaims,
};
use crate::VerifierError;

/// The bytecode read-RAF upstream cycle points shared by this address phase
/// and the stage-6b batch build: the five per-stage cycle bindings (the
/// stage-1 binding is the raw remainder tail, re-reversed), plus the register
/// opening points whose 7-var address prefixes feed the stage-value folds.
#[derive(Clone)]
pub struct BytecodeStagePoints<F: JoltField> {
    pub stage_cycle_points: [Vec<F>; 5],
    pub register_read_write_point: Vec<F>,
    pub register_val_evaluation_point: Vec<F>,
    /// The packed fused-inc consumer cycle points in stage order (`γ^5..8`):
    /// RAM read-write, RAM val-check, registers read-write, registers
    /// val-evaluation — the four reduced `Inc` claims' own cycle points, which
    /// the prover's address-phase kernel weights its fused pushforwards by.
    /// Empty on the base build (populated only under `akita`).
    pub fused_inc_cycle_points: Vec<Vec<F>>,
}

impl<F: JoltField> BytecodeStagePoints<F> {
    /// The stage-4 register read-write cycle leg (`stage_cycle_points[3]`).
    pub fn register_read_write_cycle(&self) -> &[F] {
        &self.stage_cycle_points[3]
    }

    /// The stage-5 register value-evaluation cycle leg (`stage_cycle_points[4]`).
    pub fn register_val_evaluation_cycle(&self) -> &[F] {
        &self.stage_cycle_points[4]
    }
}

/// Derive the [`BytecodeStagePoints`] from the mode-agnostic upstream opening
/// points. Shared by the stage-6a and stage-6b batch builds (both proving
/// modes, both fronts), single-sourcing the five-leg wiring on the clear-mode
/// paths. The BlindFold ZK input derivation (`crate::stages::zk::blindfold`)
/// assembles its own legs from the committed consistency points and does not
/// route through this helper.
pub fn bytecode_stage_points<F: JoltField>(
    stage1_cycle_binding: &[F],
    stage2: &Stage2BatchOutputPoints<F>,
    stage3: &Stage3OutputPoints<F>,
    stage4: &Stage4OutputPoints<F>,
    stage5: &Stage5OutputPoints<F>,
) -> Result<BytecodeStagePoints<F>, VerifierError> {
    let register_read_write_point = stage4.registers_read_write_point().to_vec();
    let register_val_evaluation_point = stage5.registers_opening_point().to_vec();
    let (_, register_read_write_cycle) = stage6_checked_split(
        "Stage 6 stage4 register read-write opening",
        &register_read_write_point,
        REGISTER_ADDRESS_BITS,
        JoltRelationId::BytecodeReadRaf,
    )?;
    let (_, register_val_evaluation_cycle) = stage6_checked_split(
        "Stage 6 stage5 register value-evaluation opening",
        &register_val_evaluation_point,
        REGISTER_ADDRESS_BITS,
        JoltRelationId::BytecodeReadRaf,
    )?;
    let stage_cycle_points = [
        stage1_cycle_binding.iter().rev().copied().collect(),
        stage2.product_remainder_point().to_vec(),
        stage3.shift_opening_point().to_vec(),
        register_read_write_cycle.to_vec(),
        register_val_evaluation_cycle.to_vec(),
    ];
    #[cfg(not(feature = "akita"))]
    let fused_inc_cycle_points = Vec::new();
    #[cfg(feature = "akita")]
    let fused_inc_cycle_points = {
        // The RAM legs' recorded points are `(address ‖ cycle)`; the fused
        // pushforwards bind their `log_t` cycle suffixes (the register legs
        // are the already-split stage 4/5 cycle legs).
        let log_t = register_read_write_cycle.len();
        let cycle_suffix = |label: &'static str, point: &[F]| {
            stage6_checked_split(
                label,
                point,
                point.len().checked_sub(log_t).ok_or_else(|| {
                    VerifierError::StageClaimPublicInputFailed {
                        stage: JoltRelationId::BytecodeReadRaf,
                        reason: format!("{label} is shorter than the cycle domain"),
                    }
                })?,
                JoltRelationId::BytecodeReadRaf,
            )
            .map(|(_, cycle)| cycle.to_vec())
        };
        vec![
            cycle_suffix(
                "Stage 6 RAM read-write inc opening",
                stage2.ram_read_write.inc(),
            )?,
            cycle_suffix(
                "Stage 6 RAM value-check inc opening",
                stage4.ram_val_check.ram_inc(),
            )?,
            register_read_write_cycle.to_vec(),
            register_val_evaluation_cycle.to_vec(),
        ]
    };
    Ok(BytecodeStagePoints {
        stage_cycle_points,
        register_read_write_point,
        register_val_evaluation_point,
        fused_inc_cycle_points,
    })
}

#[cfg(not(feature = "akita"))]
type AddressPhaseSymbolic = relations::bytecode::ReadRafAddressPhase;
#[cfg(feature = "akita")]
type AddressPhaseSymbolic =
    jolt_claims::protocols::jolt::lattice::relations::read_raf::LatticeReadRafAddressPhase;

/// Wire the prior-proof opening *values* the address-phase input claim binds
/// (every stage-1..5 opening folded by the `read_raf_address_phase` input `Expr`,
/// plus the two PC claims). Each Spartan-outer circuit flag is a direct
/// field-for-field read from the stage-1 outer remainder; the input claim reads
/// only values, so the consumed input *points* are the generated all-empty
/// `empty_input_points`.
pub fn bytecode_read_raf_address_phase_input_values_from_upstream<F: JoltField>(
    stage1: &Stage1BatchOutputClaims<F>,
    stage2: &Stage2BatchOutputClaims<F>,
    stage3: &Stage3OutputClaims<F>,
    stage4: &Stage4OutputClaims<F>,
    stage5: &Stage5OutputClaims<F>,
) -> BytecodeReadRafAddressPhaseInputClaims<F> {
    let outer = &stage1.outer_remainder;
    let product = &stage2.product_remainder;
    let instruction_input = &stage3.instruction_input;
    let shift = &stage3.shift;
    let registers_read_write = &stage4.registers_read_write;
    let instruction_read_raf = &stage5.instruction_read_raf;
    BytecodeReadRafAddressPhaseInputClaims {
        outer_unexpanded_pc: outer.unexpanded_pc,
        outer_imm: outer.imm,
        outer_add_operands: outer.add_operands,
        outer_subtract_operands: outer.subtract_operands,
        outer_multiply_operands: outer.multiply_operands,
        outer_load: outer.load,
        outer_store: outer.store,
        outer_jump: outer.jump,
        outer_write_lookup_output_to_rd: outer.write_lookup_output_to_rd,
        outer_virtual_instruction: outer.virtual_instruction,
        outer_assert: outer.assert,
        outer_do_not_update_unexpanded_pc: outer.do_not_update_unexpanded_pc,
        outer_advice: outer.advice,
        outer_is_compressed: outer.is_compressed,
        outer_is_first_in_sequence: outer.is_first_in_sequence,
        outer_is_last_in_sequence: outer.is_last_in_sequence,
        outer_pc: outer.pc,
        product_jump: product.jump_flag,
        product_branch: product.branch_flag,
        product_write_lookup_output_to_rd: product.write_lookup_output_to_rd,
        product_virtual_instruction: product.virtual_instruction,
        instruction_input_imm: instruction_input.imm,
        shift_unexpanded_pc: shift.unexpanded_pc,
        left_operand_is_rs1_value: instruction_input.left_operand_is_rs1,
        left_operand_is_pc: instruction_input.left_operand_is_pc,
        right_operand_is_rs2_value: instruction_input.right_operand_is_rs2,
        right_operand_is_imm: instruction_input.right_operand_is_imm,
        is_noop: shift.is_noop,
        shift_virtual_instruction: shift.is_virtual,
        shift_is_first_in_sequence: shift.is_first_in_sequence,
        shift_pc: shift.pc,
        rd_wa_read_write: registers_read_write.rd_wa,
        rs1_ra: registers_read_write.rs1_ra,
        rs2_ra: registers_read_write.rs2_ra,
        rd_wa_val_evaluation: stage5.registers_val_evaluation.rd_wa,
        instruction_raf_flag: instruction_read_raf.instruction_raf_flag,
        lookup_table_flags: instruction_read_raf.lookup_table_flags.clone(),
    }
}

#[derive(Clone)]
pub struct BytecodeReadRafAddressPhase<F: JoltField> {
    symbolic: AddressPhaseSymbolic,
    dimensions: BytecodeReadRafDimensions,
    /// Committed-program mode stages the `BytecodeValClaim` wire claims.
    committed_program: bool,
    /// The upstream cycle points and register opening points the address-phase
    /// kernel's PC pushforwards and stage-value folds bind against (the same
    /// [`BytecodeStagePoints`] wiring the stage-6b cycle phase carries). The
    /// verifier constructs the relation with full geometry; only the prover's
    /// kernel reads these.
    stage_points: BytecodeStagePoints<F>,
    entry_bytecode_index: usize,
    /// The FR opening values the composed input claim folds, set by the
    /// stage-6a fronts from the stage-1/4/5 clear outputs before the input
    /// claim is computed. See
    /// [`field_inline::FieldInlineBytecodeReadRafInputs`](super::field_inline::FieldInlineBytecodeReadRafInputs).
    #[cfg(feature = "field-inline")]
    field_inline_inputs: OnceLock<FieldInlineBytecodeReadRafInputs<F>>,
    /// The FR side table and opening points the address-phase kernel folds
    /// over, set by both fronts right after the batch build. See
    /// [`field_inline::FieldInlineBytecodeReadRafGeometry`](super::field_inline::FieldInlineBytecodeReadRafGeometry).
    #[cfg(feature = "field-inline")]
    field_inline_geometry: OnceLock<FieldInlineBytecodeReadRafGeometry<F>>,
}

impl<F: JoltField> BytecodeReadRafAddressPhase<F> {
    pub fn new(
        dimensions: BytecodeReadRafDimensions,
        committed_program: bool,
        stage_points: BytecodeStagePoints<F>,
        entry_bytecode_index: usize,
    ) -> Self {
        Self {
            symbolic: AddressPhaseSymbolic::new(dimensions),
            dimensions,
            committed_program,
            stage_points,
            entry_bytecode_index,
            #[cfg(feature = "field-inline")]
            field_inline_inputs: OnceLock::new(),
            #[cfg(feature = "field-inline")]
            field_inline_geometry: OnceLock::new(),
        }
    }

    /// Supply the FR kernel geometry (rejects a second set at different
    /// contents — one proof per relation instance).
    #[cfg(feature = "field-inline")]
    pub fn set_field_inline_geometry(
        &self,
        geometry: FieldInlineBytecodeReadRafGeometry<F>,
    ) -> Result<(), VerifierError> {
        let stored = self.field_inline_geometry.get_or_init(|| geometry.clone());
        if *stored != geometry {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: "field-inline bytecode read-RAF geometry already set to different \
                         contents"
                    .to_string(),
            });
        }
        Ok(())
    }

    /// The carried FR kernel geometry, fail-closed when the front never
    /// attached one.
    #[cfg(feature = "field-inline")]
    pub fn field_inline_geometry(
        &self,
    ) -> Result<&FieldInlineBytecodeReadRafGeometry<F>, VerifierError> {
        self.field_inline_geometry
            .get()
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: "field-inline bytecode read-RAF geometry was never attached".to_string(),
            })
    }

    /// Supply the FR opening values the composed input claim folds. Must be
    /// called before `input_claim`; rejects a second set at different values
    /// (one proof per relation instance).
    #[cfg(feature = "field-inline")]
    pub fn set_field_inline_inputs(
        &self,
        values: FieldInlineBytecodeReadRafInputs<F>,
    ) -> Result<(), VerifierError> {
        let stored = self.field_inline_inputs.get_or_init(|| values.clone());
        if *stored != values {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: "field-inline bytecode read-RAF inputs already set to different values"
                    .to_string(),
            });
        }
        Ok(())
    }

    pub fn committed_program(&self) -> bool {
        self.committed_program
    }

    pub fn dimensions(&self) -> BytecodeReadRafDimensions {
        self.dimensions
    }

    pub fn stage_cycle_points(&self) -> &[Vec<F>; 5] {
        &self.stage_points.stage_cycle_points
    }

    /// The packed fused-inc consumer cycle points (`γ^5..8` stage order);
    /// empty on the base build. See [`BytecodeStagePoints`].
    pub fn fused_inc_cycle_points(&self) -> &[Vec<F>] {
        &self.stage_points.fused_inc_cycle_points
    }

    /// The full stage-4 register read-write opening point (address prefix ‖
    /// cycle); the stage-value fold reads its `REGISTER_ADDRESS_BITS` prefix.
    pub fn register_read_write_point(&self) -> &[F] {
        &self.stage_points.register_read_write_point
    }

    /// The full stage-5 register value-evaluation opening point (address
    /// prefix ‖ cycle); the stage-value fold reads its
    /// `REGISTER_ADDRESS_BITS` prefix.
    pub fn register_val_evaluation_point(&self) -> &[F] {
        &self.stage_points.register_val_evaluation_point
    }

    pub fn entry_bytecode_index(&self) -> usize {
        self.entry_bytecode_index
    }

    /// The staged `BytecodeValClaim` wire-claim count: all
    /// `NUM_BYTECODE_VAL_STAGES` in committed-program mode, none in full mode.
    fn num_val_stages(&self) -> usize {
        if self.committed_program {
            bytecode_reduction::NUM_BYTECODE_VAL_STAGES
        } else {
            0
        }
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for BytecodeReadRafAddressPhase<F> {
    type Symbolic = AddressPhaseSymbolic;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn wire_output_openings(&self) -> std::collections::BTreeSet<JoltOpeningId> {
        // Committed-program mode absorbs the staged `BytecodeValClaim` columns
        // beyond the output-`Expr` set (the address-phase intermediate); their
        // constraining fold happens in stage 6b's bytecode claim reduction.
        let mut openings = self.symbolic().expected_output_openings::<F>();
        openings
            .extend((0..self.num_val_stages()).map(bytecode_reduction::bytecode_val_stage_opening));
        openings
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &SumcheckInputPoints<F, Self>,
    ) -> Result<BytecodeReadRafAddressPhaseOutputClaims<Vec<F>>, VerifierError> {
        // `bytecode_r_address` is the reversed address sumcheck point; the
        // intermediate and every staged Val column open there.
        let r_address = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(BytecodeReadRafAddressPhaseOutputClaims {
            intermediate: r_address.clone(),
            val_stages: vec![r_address; self.num_val_stages()],
        })
    }

    /// The composed input claim: the ordinary gamma-folded bind (the jolt
    /// symbolic input `Expr`) plus the FR appendage extension — the jolt
    /// symbolic expression cannot name the FR openings, so the composed form
    /// adds [`super::field_inline::input_claim_extension`] over the supplied
    /// appendage (spec: `field-inline-protocol.md`, "Stage 6 Composition").
    #[cfg(feature = "field-inline")]
    fn input_claim(
        &self,
        input_values: &SumcheckInputClaims<F, Self>,
        challenges: &BytecodeReadRafAddressPhaseChallenges<F>,
    ) -> Result<F, VerifierError> {
        use jolt_claims::{InputClaims as _, SumcheckChallenges as _};

        let ordinary = self.symbolic().input_expression::<F>().try_evaluate(
            |id| {
                input_values
                    .resolve_input(id)
                    .ok_or(VerifierError::MissingOpeningClaim { id: (*id).into() })
            },
            |id| {
                challenges
                    .resolve_challenge(id)
                    .ok_or(VerifierError::MissingStageClaimChallenge { id: (*id).into() })
            },
            |id| self.derive_input_term(id, challenges),
        )?;

        let field_inline = self.field_inline_inputs.get().ok_or_else(|| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: "field-inline bytecode read-RAF inputs not set (the stage-6a front must \
                         supply them from the stage-1/4/5 outputs before the input claim)"
                    .to_string(),
            }
        })?;
        Ok(ordinary + super::field_inline::input_claim_extension(field_inline, challenges)?)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test code indexes its own fixed-size fixtures"
)]
mod tests {
    use super::*;
    use crate::stages::relations::draw_recording::{record, DrawEvent};
    use jolt_field::Fr;
    use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
    use jolt_riscv::NUM_CIRCUIT_FLAGS;
    use jolt_transcript::Transcript;

    // The address phase has the only multi-field `Challenges` (gamma + five stage
    // gammas), so it exercises that the default draws one `challenge_scalar` per
    // field in declaration order. Each inline draw is a `challenge_scalar_powers(..)`
    // whose single squeeze's degree-1 power equals that squeezed scalar, so the
    // default's six `challenge_scalar` squeezes reproduce the inline byte stream
    // (six squeezes) and the six stored values. The cycle and committed variants are
    // single-field and use the same default path.
    #[test]
    fn default_draw_challenges_matches_inline_bytecode_address_gammas() {
        let relation = BytecodeReadRafAddressPhase::<Fr>::new(
            BytecodeReadRafDimensions::new(3, 4, 2),
            false,
            BytecodeStagePoints {
                stage_cycle_points: Default::default(),
                register_read_write_point: Vec::new(),
                register_val_evaluation_point: Vec::new(),
                fused_inc_cycle_points: Vec::new(),
            },
            0,
        );

        // Inline: six `challenge_scalar_powers(..)`, each contributing its
        // degree-1 power.
        let (inline_events, inline_gammas) = record(|t| {
            [
                t.challenge_scalar_powers(8)[1],
                t.challenge_scalar_powers(2 + NUM_CIRCUIT_FLAGS)[1],
                t.challenge_scalar_powers(4)[1],
                t.challenge_scalar_powers(9)[1],
                t.challenge_scalar_powers(3)[1],
                t.challenge_scalar_powers(2 + LookupTableKind::<RISCV_XLEN>::COUNT)[1],
            ]
        });
        let (draw_events, challenges) = record(|t| relation.draw_challenges(t).unwrap());

        // Six squeezes in the same order, byte-for-byte.
        assert_eq!(draw_events, inline_events);
        assert_eq!(
            draw_events,
            (1..=6).map(DrawEvent::Squeeze).collect::<Vec<_>>()
        );
        // Each field stores the corresponding inline degree-1 power.
        assert_eq!(
            [
                challenges.gamma,
                challenges.stage1_gamma,
                challenges.stage2_gamma,
                challenges.stage3_gamma,
                challenges.stage4_gamma,
                challenges.stage5_gamma,
            ],
            inline_gammas,
        );
    }
}

// The dory-shaped composition pins (base input-claims struct, five stage
// points); the packed composition is covered by the prover's FR stage
// round-trips and the packed e2e suite.
#[cfg(all(test, feature = "field-inline", not(feature = "akita")))]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    reason = "test code indexes its own fixed-size fixtures and uses plain arithmetic on fixture data"
)]
mod field_inline_tests {
    use super::super::field_inline::FieldInlineBytecodeReadRafInputs;
    use super::*;
    use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
    use jolt_claims::{InputClaims as _, SumcheckChallenges as _};
    use jolt_field::{Fr, Ring};
    use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
    use jolt_riscv::NUM_CIRCUIT_FLAGS;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn relation() -> BytecodeReadRafAddressPhase<Fr> {
        BytecodeReadRafAddressPhase::new(
            BytecodeReadRafDimensions::new(3, 4, 2),
            false,
            BytecodeStagePoints {
                stage_cycle_points: Default::default(),
                fused_inc_cycle_points: Vec::new(),
                register_read_write_point: Vec::new(),
                register_val_evaluation_point: Vec::new(),
            },
            0,
        )
    }

    fn input_values() -> BytecodeReadRafAddressPhaseInputClaims<Fr> {
        let mut inputs = BytecodeReadRafAddressPhaseInputClaims {
            lookup_table_flags: vec![Fr::from_u64(0); LookupTableKind::<RISCV_XLEN>::COUNT],
            ..Default::default()
        };
        // Distinct sentinels on a spread of ordinary openings so the ordinary
        // leg of the pin is non-trivial.
        inputs.outer_unexpanded_pc = fr(3);
        inputs.outer_imm = fr(5);
        inputs.outer_jump = fr(7);
        inputs.product_branch = fr(11);
        inputs.instruction_input_imm = fr(13);
        inputs.rd_wa_read_write = fr(17);
        inputs.rs1_ra = fr(19);
        inputs.rs2_ra = fr(23);
        inputs.rd_wa_val_evaluation = fr(29);
        inputs.instruction_raf_flag = fr(31);
        for (index, flag) in inputs.lookup_table_flags.iter_mut().enumerate() {
            *flag = fr(100 + index as u64);
        }
        inputs
    }

    fn field_inline_inputs() -> FieldInlineBytecodeReadRafInputs<Fr> {
        FieldInlineBytecodeReadRafInputs {
            field_op_flags: core::array::from_fn(|index| fr(200 + index as u64)),
            rd_wa_read_write: fr(301),
            rs1_ra: fr(302),
            rs2_ra: fr(303),
            rd_wa_val_evaluation: fr(304),
        }
    }

    fn challenges() -> BytecodeReadRafAddressPhaseChallenges<Fr> {
        BytecodeReadRafAddressPhaseChallenges {
            gamma: fr(401),
            stage1_gamma: fr(402),
            stage2_gamma: fr(403),
            stage3_gamma: fr(404),
            stage4_gamma: fr(405),
            stage5_gamma: fr(406),
        }
    }

    fn powers(gamma: Fr, len: usize) -> Vec<Fr> {
        let mut powers = vec![Fr::from_u64(1); len];
        for index in 1..len {
            powers[index] = powers[index - 1] * gamma;
        }
        powers
    }

    /// The composed input claim equals the from-scratch fold: the ordinary
    /// symbolic bind plus the FR terms at the extended stage-1/4/5 power
    /// indices, each stage extension riding the same outer gamma power as its
    /// ordinary stage claim (spec: `field-inline-protocol.md`, "Stage 6
    /// Composition" — Stage1 powers gain the eight `FieldOpFlag`s, Stage4
    /// powers gain `FieldRdWa`/`FieldRs1Ra`/`FieldRs2Ra`, Stage5 powers gain
    /// the val-evaluation `FieldRdWa`).
    #[test]
    fn composed_input_claim_matches_from_scratch_fold() {
        let relation = relation();
        let inputs = input_values();
        let challenges = challenges();
        let field_inline = field_inline_inputs();
        relation
            .set_field_inline_inputs(field_inline.clone())
            .unwrap();

        let ordinary = relation
            .symbolic()
            .input_expression::<Fr>()
            .try_evaluate(
                |id| {
                    inputs
                        .resolve_input(id)
                        .ok_or(VerifierError::MissingOpeningClaim { id: (*id).into() })
                },
                |id| {
                    challenges
                        .resolve_challenge(id)
                        .ok_or(VerifierError::MissingStageClaimChallenge { id: (*id).into() })
                },
                |_| {
                    Err(VerifierError::StageClaimPublicInputFailed {
                        stage: JoltRelationId::BytecodeReadRaf,
                        reason: "no input deriveds".to_string(),
                    })
                },
            )
            .unwrap();

        let stage1_powers = powers(challenges.stage1_gamma, 2 + NUM_CIRCUIT_FLAGS + 8);
        let stage4_powers = powers(challenges.stage4_gamma, 6);
        let stage5_powers = powers(
            challenges.stage5_gamma,
            2 + LookupTableKind::<RISCV_XLEN>::COUNT + 1,
        );
        let fr_stage1: Fr = field_inline
            .field_op_flags
            .iter()
            .enumerate()
            .map(|(index, flag)| stage1_powers[2 + NUM_CIRCUIT_FLAGS + index] * *flag)
            .sum();
        let fr_stage4 = stage4_powers[3] * field_inline.rd_wa_read_write
            + stage4_powers[4] * field_inline.rs1_ra
            + stage4_powers[5] * field_inline.rs2_ra;
        let fr_stage5 = stage5_powers[2 + LookupTableKind::<RISCV_XLEN>::COUNT]
            * field_inline.rd_wa_val_evaluation;
        let gamma = challenges.gamma;
        let expected = ordinary
            + fr_stage1
            + gamma * gamma * gamma * fr_stage4
            + gamma * gamma * gamma * gamma * fr_stage5;

        let composed = relation.input_claim(&inputs, &challenges).unwrap();
        assert_eq!(composed, expected);
    }

    /// With a zeroed FR appendage the composed input claim reduces to the
    /// ordinary symbolic bind — pinning the override's ordinary leg to the
    /// symbolic source of truth.
    #[test]
    fn composed_input_claim_reduces_to_symbolic_form_without_field_terms() {
        let relation = relation();
        let inputs = input_values();
        let challenges = challenges();
        relation
            .set_field_inline_inputs(FieldInlineBytecodeReadRafInputs {
                field_op_flags: [Fr::from_u64(0); 8],
                rd_wa_read_write: Fr::from_u64(0),
                rs1_ra: Fr::from_u64(0),
                rs2_ra: Fr::from_u64(0),
                rd_wa_val_evaluation: Fr::from_u64(0),
            })
            .unwrap();

        let ordinary = relation
            .symbolic()
            .input_expression::<Fr>()
            .try_evaluate(
                |id| {
                    inputs
                        .resolve_input(id)
                        .ok_or(VerifierError::MissingOpeningClaim { id: (*id).into() })
                },
                |id| {
                    challenges
                        .resolve_challenge(id)
                        .ok_or(VerifierError::MissingStageClaimChallenge { id: (*id).into() })
                },
                |_| {
                    Err(VerifierError::StageClaimPublicInputFailed {
                        stage: JoltRelationId::BytecodeReadRaf,
                        reason: "no input deriveds".to_string(),
                    })
                },
            )
            .unwrap();
        assert_eq!(
            relation.input_claim(&inputs, &challenges).unwrap(),
            ordinary
        );
    }

    /// An unset FR appendage fails closed rather than computing the ordinary
    /// claim (which would desynchronize the composed transcript).
    #[test]
    fn composed_input_claim_requires_the_field_inline_appendage() {
        let relation = relation();
        assert!(matches!(
            relation.input_claim(&input_values(), &challenges()),
            Err(VerifierError::StageClaimPublicInputFailed { .. })
        ));
    }
}
