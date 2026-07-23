use std::collections::BTreeSet;

use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_field::{Field, Fr};
use jolt_verifier::{
    proof::ClearProofClaims,
    stages::{stage1, stage2, stage3, stage4, stage5, stage6a, stage6b, stage7},
    VerifierError,
};
use serde_json::Value;

use crate::support::VerifierPhase;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TamperMode {
    Standard,
    Zk,
    Both,
}

impl TamperMode {
    pub fn includes(self, zk: bool) -> bool {
        matches!(
            (self, zk),
            (Self::Both, _) | (Self::Standard, false) | (Self::Zk, true)
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(
    clippy::enum_variant_names,
    reason = "the shared `Checked` prefix records where a tampered field is caught; dropping it would obscure the disposition semantics"
)]
pub enum TamperDisposition {
    CheckedAtStage,
    CheckedByLaterStage(VerifierPhase),
    CheckedByFinalOpenings,
    CheckedByZk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutationStrategy {
    OffsetScalar,
    FlipBool,
    ChangeEnumVariant,
    RemoveItem,
    DuplicateItem,
    SwapOrder,
    TruncateVector,
    ExtendVector,
    ReplaceProofPayload,
    ReplaceModePayload,
    ReplaceSetup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TamperCoverage {
    Active,
    IgnoredUntilFixture,
    Deferred,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TamperTarget {
    pub name: &'static str,
    pub path: &'static str,
    pub mode: TamperMode,
    pub checked_at: VerifierPhase,
    pub disposition: TamperDisposition,
    pub strategy: MutationStrategy,
    pub coverage: TamperCoverage,
    pub reason: &'static str,
}

const fn checked_standard(
    name: &'static str,
    path: &'static str,
    checked_at: VerifierPhase,
    strategy: MutationStrategy,
    coverage: TamperCoverage,
    reason: &'static str,
) -> TamperTarget {
    TamperTarget {
        name,
        path,
        mode: TamperMode::Standard,
        checked_at,
        disposition: TamperDisposition::CheckedAtStage,
        strategy,
        coverage,
        reason,
    }
}

const fn checked_both(
    name: &'static str,
    path: &'static str,
    checked_at: VerifierPhase,
    strategy: MutationStrategy,
    coverage: TamperCoverage,
    reason: &'static str,
) -> TamperTarget {
    TamperTarget {
        name,
        path,
        mode: TamperMode::Both,
        checked_at,
        disposition: TamperDisposition::CheckedAtStage,
        strategy,
        coverage,
        reason,
    }
}

const fn later_standard(
    name: &'static str,
    path: &'static str,
    later_phase: VerifierPhase,
    strategy: MutationStrategy,
    coverage: TamperCoverage,
    reason: &'static str,
) -> TamperTarget {
    TamperTarget {
        name,
        path,
        mode: TamperMode::Standard,
        checked_at: later_phase,
        disposition: TamperDisposition::CheckedByLaterStage(later_phase),
        strategy,
        coverage,
        reason,
    }
}

const fn final_opening_standard(
    name: &'static str,
    path: &'static str,
    strategy: MutationStrategy,
    coverage: TamperCoverage,
    reason: &'static str,
) -> TamperTarget {
    TamperTarget {
        name,
        path,
        mode: TamperMode::Standard,
        checked_at: VerifierPhase::Stage8Openings,
        disposition: TamperDisposition::CheckedByFinalOpenings,
        strategy,
        coverage,
        reason,
    }
}

const fn final_opening_zk(
    name: &'static str,
    path: &'static str,
    strategy: MutationStrategy,
    coverage: TamperCoverage,
    reason: &'static str,
) -> TamperTarget {
    TamperTarget {
        name,
        path,
        mode: TamperMode::Zk,
        checked_at: VerifierPhase::Stage8Openings,
        disposition: TamperDisposition::CheckedByFinalOpenings,
        strategy,
        coverage,
        reason,
    }
}

const fn zk_target(
    name: &'static str,
    path: &'static str,
    strategy: MutationStrategy,
    coverage: TamperCoverage,
    reason: &'static str,
) -> TamperTarget {
    TamperTarget {
        name,
        path,
        mode: TamperMode::Zk,
        checked_at: VerifierPhase::Zk,
        disposition: TamperDisposition::CheckedByZk,
        strategy,
        coverage,
        reason,
    }
}

pub const PREAMBLE_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "public_io.inputs",
        "public_io.inputs",
        VerifierPhase::Stage1,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "absorbed into the transcript; the first concrete rejection is a stage sumcheck mismatch",
    ),
    checked_standard(
        "public_io.outputs",
        "public_io.outputs",
        VerifierPhase::Stage2,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "real fixture output mutation; rejected no later than the RAM output check",
    ),
    checked_standard(
        "public_io.panic",
        "public_io.panic",
        VerifierPhase::Stage2,
        MutationStrategy::FlipBool,
        TamperCoverage::Active,
        "panic materializes as public memory for the RAM output check; real fixture bool flip",
    ),
    checked_both(
        "public_io.memory_layout",
        "public_io.memory_layout",
        VerifierPhase::Preamble,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "validate_inputs rejects layout mismatch before transcript use",
    ),
    checked_standard(
        "proof.trace_length",
        "proof.trace_length",
        VerifierPhase::Preamble,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "invalid and excessive trace lengths are checked during input validation",
    ),
    checked_standard(
        "proof.ram_K",
        "proof.ram_K",
        VerifierPhase::Preamble,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "invalid RAM domains are checked during input validation",
    ),
    checked_standard(
        "proof.rw_config",
        "proof.rw_config",
        VerifierPhase::Stage2,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "stage 2 consumes RAM read-write phase lengths when slicing batched points",
    ),
    later_standard(
        "proof.one_hot_config",
        "proof.one_hot_config",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "transcript-bound and substantively checked by later RA virtualization stages; real fixture config offset",
    ),
    checked_standard(
        "proof.trace_polynomial_order",
        "proof.trace_polynomial_order",
        VerifierPhase::Stage1,
        MutationStrategy::ChangeEnumVariant,
        TamperCoverage::Active,
        "transcript-bound; real fixture enum flip diverges challenges at the first stage sumcheck",
    ),
    checked_standard(
        "preprocessing.preprocessing_digest",
        "preprocessing.preprocessing_digest",
        VerifierPhase::Stage1,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "transcript-bound; real fixture digest offset diverges challenges at the first stage sumcheck",
    ),
    checked_standard(
        "preprocessing.program.bytecode.entry_address",
        "preprocessing.program.bytecode.entry_address",
        VerifierPhase::Stage1,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "transcript-bound; real fixture entry-address offset diverges challenges at the first stage sumcheck",
    ),
];

pub const COMMITMENT_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "proof.commitments.value",
        "proof.commitments[*]",
        VerifierPhase::Stage1,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "real verifier fixture replaces one commitment with another and rejects before acceptance",
    ),
    checked_standard(
        "proof.commitments.missing",
        "proof.commitments",
        VerifierPhase::Stage1,
        MutationStrategy::RemoveItem,
        TamperCoverage::Active,
        "real verifier fixture removes one commitment and rejects before acceptance",
    ),
    checked_standard(
        "proof.commitments.extra",
        "proof.commitments",
        VerifierPhase::Stage1,
        MutationStrategy::DuplicateItem,
        TamperCoverage::Active,
        "real verifier fixture duplicates one commitment and rejects before acceptance",
    ),
    checked_standard(
        "proof.commitments.order",
        "proof.commitments",
        VerifierPhase::Stage1,
        MutationStrategy::SwapOrder,
        TamperCoverage::Active,
        "real verifier fixture swaps commitments and rejects before acceptance",
    ),
    checked_standard(
        "proof.untrusted_advice_commitment",
        "proof.untrusted_advice_commitment",
        VerifierPhase::Stage1,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "advice fixture replaces the untrusted advice commitment with a valid but wrong commitment",
    ),
    checked_standard(
        "trusted_advice_commitment",
        "trusted_advice_commitment",
        VerifierPhase::Stage1,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "advice fixture replaces the trusted advice commitment with a valid but wrong commitment",
    ),
    final_opening_standard(
        "proof.joint_opening_proof",
        "proof.joint_opening_proof",
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "real verifier fixture replaces the joint opening proof with another valid proof payload",
    ),
];

pub const PROOF_SHAPE_TARGETS: &[TamperTarget] = &[
    checked_both(
        "proof.stages.clear_vs_committed",
        "proof.stages.*",
        VerifierPhase::Preamble,
        MutationStrategy::ReplaceModePayload,
        TamperCoverage::Active,
        "validate_proof_consistency rejects mixed clear/committed mode",
    ),
    checked_both(
        "proof.claims.mode_payload",
        "proof.claims",
        VerifierPhase::Preamble,
        MutationStrategy::ReplaceModePayload,
        TamperCoverage::Active,
        "validate_proof_consistency rejects clear/ZK payload mismatches",
    ),
];

pub const STAGE1_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "stage1.uni_skip.round_polynomial",
        "proof.stages.stage1_uni_skip_first_round_proof.round_polynomials[*]",
        VerifierPhase::Stage1,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "prover-fixture test mutates every full uni-skip round polynomial",
    ),
    checked_standard(
        "stage1.uni_skip.round_count.missing",
        "proof.stages.stage1_uni_skip_first_round_proof.round_polynomials",
        VerifierPhase::Stage1,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "prover-fixture test removes a uni-skip round",
    ),
    checked_standard(
        "stage1.uni_skip.round_count.extra",
        "proof.stages.stage1_uni_skip_first_round_proof.round_polynomials",
        VerifierPhase::Stage1,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "prover-fixture test appends a uni-skip round",
    ),
    checked_standard(
        "stage1.remainder.round_polynomial",
        "proof.stages.stage1_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage1,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "prover-fixture test mutates every compressed remainder round polynomial",
    ),
    checked_standard(
        "stage1.remainder.round_count.missing",
        "proof.stages.stage1_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage1,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "prover-fixture test removes a remainder round",
    ),
    checked_standard(
        "stage1.remainder.round_count.extra",
        "proof.stages.stage1_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage1,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "prover-fixture test appends a remainder round",
    ),
    checked_standard(
        "stage1.claims.uniskip_output_claim",
        "claims.stage1.uniskip_output_claim",
        VerifierPhase::Stage1,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets the Spartan outer uni-skip opening claim",
    ),
    later_standard(
        "stage1.claims.outer",
        "claims.stage1.outer.*",
        VerifierPhase::Stage2,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "outer output claims are consumed as stage-2 input claims (product \
         virtualization); the PCS binds them again at final openings",
    ),
];

pub const STAGE2_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "stage2.product_uniskip.round_polynomial",
        "proof.stages.stage2_uni_skip_first_round_proof.round_polynomials[*]",
        VerifierPhase::Stage2,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "prover-fixture test mutates every product uni-skip round polynomial",
    ),
    checked_standard(
        "stage2.product_uniskip.round_count.missing",
        "proof.stages.stage2_uni_skip_first_round_proof.round_polynomials",
        VerifierPhase::Stage2,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "prover-fixture test removes a product uni-skip round",
    ),
    checked_standard(
        "stage2.product_uniskip.round_count.extra",
        "proof.stages.stage2_uni_skip_first_round_proof.round_polynomials",
        VerifierPhase::Stage2,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "prover-fixture test appends a product uni-skip round",
    ),
    checked_standard(
        "stage2.batch.round_polynomial",
        "proof.stages.stage2_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage2,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "prover-fixture test mutates every compressed Stage 2 batch round polynomial",
    ),
    checked_standard(
        "stage2.batch.round_count.missing",
        "proof.stages.stage2_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage2,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "prover-fixture test removes a Stage 2 batch round",
    ),
    checked_standard(
        "stage2.batch.round_count.extra",
        "proof.stages.stage2_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage2,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "prover-fixture test appends a Stage 2 batch round",
    ),
    checked_standard(
        "stage2.claims.product_uniskip_output_claim",
        "claims.stage2.product_uniskip_output_claim",
        VerifierPhase::Stage2,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets the product uni-skip opening claim",
    ),
    checked_standard(
        "stage2.claims.batch_outputs.ram_read_write",
        "claims.stage2.batch_outputs.ram_read_write.*",
        VerifierPhase::Stage2,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets each RAM read-write output claim",
    ),
    checked_standard(
        "stage2.claims.batch_outputs.product_remainder.checked",
        "claims.stage2.batch_outputs.product_remainder.{left_instruction_input,right_instruction_input,jump_flag,lookup_output,branch_flag,next_is_noop}",
        VerifierPhase::Stage2,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets every product remainder output used by the Stage 2 formula",
    ),
    checked_standard(
        "stage2.claims.batch_outputs.product_remainder.write_lookup_output_to_rd",
        "claims.stage2.batch_outputs.product_remainder.write_lookup_output_to_rd",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "Stage 6 bytecode read-RAF consumes this Stage 2 pass-through claim",
    ),
    checked_standard(
        "stage2.claims.batch_outputs.product_remainder.virtual_instruction",
        "claims.stage2.batch_outputs.product_remainder.virtual_instruction",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "Stage 6 bytecode read-RAF consumes this Stage 2 pass-through claim",
    ),
    checked_standard(
        "stage2.claims.batch_outputs.instruction_claim_reduction",
        "claims.stage2.batch_outputs.instruction_claim_reduction.*",
        VerifierPhase::Stage2,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets each instruction claim-reduction output (aliased cells are rejected by the generated validate_aliases)",
    ),
    checked_standard(
        "stage2.claims.batch_outputs.ram_raf_evaluation",
        "claims.stage2.batch_outputs.ram_raf_evaluation.ram_ra",
        VerifierPhase::Stage2,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets the RAM RAF evaluation output claim",
    ),
    checked_standard(
        "stage2.claims.batch_outputs.ram_output_check",
        "claims.stage2.batch_outputs.ram_output_check.val_final",
        VerifierPhase::Stage2,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets the RAM output-check claim",
    ),
];

pub const STAGE3_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "stage3.batch.round_polynomial",
        "proof.stages.stage3_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage3,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "prover-fixture test mutates every compressed Stage 3 batch round polynomial",
    ),
    checked_standard(
        "stage3.batch.round_count.missing",
        "proof.stages.stage3_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage3,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "prover-fixture test removes a Stage 3 batch round",
    ),
    checked_standard(
        "stage3.batch.round_count.extra",
        "proof.stages.stage3_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage3,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "prover-fixture test appends a Stage 3 batch round",
    ),
    checked_standard(
        "stage3.claims.shift",
        "claims.stage3.shift.*",
        VerifierPhase::Stage3,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets each Spartan shift output claim",
    ),
    checked_standard(
        "stage3.claims.instruction_input",
        "claims.stage3.instruction_input.*",
        VerifierPhase::Stage3,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets each instruction-input output claim",
    ),
    checked_standard(
        "stage3.claims.registers_claim_reduction",
        "claims.stage3.registers_claim_reduction.*",
        VerifierPhase::Stage3,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets each register claim-reduction output claim",
    ),
];

pub const STAGE4_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "stage4.batch.round_polynomial",
        "proof.stages.stage4_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage4,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "prover-fixture test mutates every compressed Stage 4 batch round polynomial",
    ),
    checked_standard(
        "stage4.batch.round_count.missing",
        "proof.stages.stage4_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage4,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "prover-fixture test removes a Stage 4 batch round",
    ),
    checked_standard(
        "stage4.batch.round_count.extra",
        "proof.stages.stage4_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage4,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "prover-fixture test appends a Stage 4 batch round",
    ),
    checked_standard(
        "stage4.claims.registers_read_write",
        "claims.stage4.registers_read_write.*",
        VerifierPhase::Stage4,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets each register read-write output claim",
    ),
    checked_standard(
        "stage4.claims.ram_val_check",
        "claims.stage4.ram_val_check.*",
        VerifierPhase::Stage4,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets each RAM value-check output claim",
    ),
    checked_standard(
        "stage4.claims.advice.untrusted",
        "claims.stage4.ram_val_check.untrusted_advice",
        VerifierPhase::Stage4,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "advice fixture test offsets the untrusted advice opening consumed by RAM value check",
    ),
    checked_standard(
        "stage4.claims.advice.trusted",
        "claims.stage4.ram_val_check.trusted_advice",
        VerifierPhase::Stage4,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "advice fixture test offsets the trusted advice opening consumed by RAM value check",
    ),
    checked_standard(
        "stage4.claims.program_image_contribution",
        "claims.stage4.ram_val_check.program_image",
        VerifierPhase::Stage4,
        MutationStrategy::OffsetScalar,
        TamperCoverage::IgnoredUntilFixture,
        "committed fixture test offsets the staged program-image init contribution",
    ),
];

pub const STAGE5_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "stage5.batch.round_polynomial",
        "proof.stages.stage5_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage5,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "prover-fixture test mutates every compressed Stage 5 batch round polynomial",
    ),
    checked_standard(
        "stage5.batch.round_count.missing",
        "proof.stages.stage5_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage5,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "prover-fixture test removes a Stage 5 batch round",
    ),
    checked_standard(
        "stage5.batch.round_count.extra",
        "proof.stages.stage5_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage5,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "prover-fixture test appends a Stage 5 batch round",
    ),
    checked_standard(
        "stage5.claims.instruction_read_raf.lookup_table_flags",
        "claims.stage5.instruction_read_raf.lookup_table_flags",
        VerifierPhase::Stage5,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets each instruction read-RAF lookup-table flag claim",
    ),
    checked_standard(
        "stage5.claims.instruction_read_raf.instruction_ra",
        "claims.stage5.instruction_read_raf.instruction_ra",
        VerifierPhase::Stage5,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets each instruction read-RAF virtual RA claim",
    ),
    checked_standard(
        "stage5.claims.instruction_read_raf.instruction_raf_flag",
        "claims.stage5.instruction_read_raf.instruction_raf_flag",
        VerifierPhase::Stage5,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets the instruction read-RAF flag claim",
    ),
    checked_standard(
        "stage5.claims.ram_ra_claim_reduction",
        "claims.stage5.ram_ra_claim_reduction.ram_ra",
        VerifierPhase::Stage5,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets the RAM RA claim-reduction output claim",
    ),
    checked_standard(
        "stage5.claims.registers_val_evaluation",
        "claims.stage5.registers_val_evaluation.*",
        VerifierPhase::Stage5,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets each register value-evaluation output claim",
    ),
];

pub const STAGE6_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "stage6.address_phase.round_polynomial",
        "proof.stages.stage6a_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage6,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "prover-fixture test mutates every compressed Stage 6 address-phase round polynomial",
    ),
    checked_standard(
        "stage6.address_phase.round_count.missing",
        "proof.stages.stage6a_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage6,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "prover-fixture test removes a Stage 6 address-phase round",
    ),
    checked_standard(
        "stage6.address_phase.round_count.extra",
        "proof.stages.stage6a_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage6,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "prover-fixture test appends a Stage 6 address-phase round",
    ),
    checked_standard(
        "stage6.cycle_phase.round_polynomial",
        "proof.stages.stage6b_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage6,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "prover-fixture test mutates every compressed Stage 6 cycle-phase round polynomial",
    ),
    checked_standard(
        "stage6.cycle_phase.round_count.missing",
        "proof.stages.stage6b_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage6,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "prover-fixture test removes a Stage 6 cycle-phase round",
    ),
    checked_standard(
        "stage6.cycle_phase.round_count.extra",
        "proof.stages.stage6b_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage6,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "prover-fixture test appends a Stage 6 cycle-phase round",
    ),
    checked_standard(
        "stage6.claims.address_phase.bytecode_read_raf",
        "claims.stage6a.bytecode_read_raf.intermediate",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets the Stage 6 bytecode read-RAF address-phase output claim",
    ),
    checked_standard(
        "stage6.claims.address_phase.booleanity",
        "claims.stage6a.booleanity.intermediate",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets the Stage 6 Booleanity address-phase output claim",
    ),
    checked_standard(
        "stage6.claims.bytecode_read_raf.bytecode_ra",
        "claims.stage6b.bytecode_read_raf.bytecode_ra",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets every bytecode read-RAF output claim",
    ),
    checked_standard(
        "stage6.claims.booleanity.instruction_ra",
        "claims.stage6b.booleanity.instruction_ra",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets every Booleanity instruction RA output claim",
    ),
    checked_standard(
        "stage6.claims.booleanity.bytecode_ra",
        "claims.stage6b.booleanity.bytecode_ra",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets every Booleanity bytecode RA output claim",
    ),
    checked_standard(
        "stage6.claims.booleanity.ram_ra",
        "claims.stage6b.booleanity.ram_ra",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets every Booleanity RAM RA output claim",
    ),
    checked_standard(
        "stage6.claims.ram_hamming_booleanity.ram_hamming_weight",
        "claims.stage6b.ram_hamming_booleanity.ram_hamming_weight",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets the RAM hamming Booleanity output claim",
    ),
    checked_standard(
        "stage6.claims.ram_ra_virtualization.ram_ra",
        "claims.stage6b.ram_ra_virtualization.ram_ra",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets every RAM RA virtualization output claim",
    ),
    checked_standard(
        "stage6.claims.instruction_ra_virtualization.committed_instruction_ra",
        "claims.stage6b.instruction_ra_virtualization.committed_instruction_ra",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets every instruction RA virtualization output claim",
    ),
    checked_standard(
        "stage6.claims.inc_claim_reduction.ram_inc",
        "claims.stage6b.inc_claim_reduction.ram_inc",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets the RAM increment reduction output claim",
    ),
    checked_standard(
        "stage6.claims.inc_claim_reduction.rd_inc",
        "claims.stage6b.inc_claim_reduction.rd_inc",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets the register increment reduction output claim",
    ),
    checked_standard(
        "stage6.claims.trusted_advice.trusted",
        "claims.stage6b.trusted_advice.trusted",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "advice fixture test offsets the trusted advice cycle-phase output claim",
    ),
    checked_standard(
        "stage6.claims.untrusted_advice.untrusted",
        "claims.stage6b.untrusted_advice.untrusted",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "advice fixture test offsets the untrusted advice cycle-phase output claim",
    ),
    checked_standard(
        "stage6.claims.address_phase.bytecode_val_stages",
        "claims.stage6a.bytecode_read_raf.val_stages",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::IgnoredUntilFixture,
        "committed fixture test offsets each staged bytecode Val-stage claim",
    ),
    checked_standard(
        "stage6.claims.bytecode_reduction.intermediate",
        "claims.stage6b.bytecode_reduction.intermediate",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::IgnoredUntilFixture,
        "committed fixture test offsets the bytecode reduction cycle-phase intermediate output claim",
    ),
    checked_standard(
        "stage6.claims.bytecode_reduction.chunks",
        "claims.stage6b.bytecode_reduction.chunks",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::IgnoredUntilFixture,
        "committed fixture test offsets the bytecode reduction cycle-phase per-chunk output claims",
    ),
    checked_standard(
        "stage6.claims.program_image_reduction.program_image",
        "claims.stage6b.program_image_reduction.program_image",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::IgnoredUntilFixture,
        "committed fixture test offsets the program-image reduction cycle-phase output claim",
    ),
];

pub const STAGE7_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "stage7.batch.round_polynomial",
        "proof.stages.stage7_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage7,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "prover-fixture test mutates every compressed Stage 7 batch round polynomial",
    ),
    checked_standard(
        "stage7.batch.round_count.missing",
        "proof.stages.stage7_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage7,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "prover-fixture test removes a Stage 7 batch round",
    ),
    checked_standard(
        "stage7.batch.round_count.extra",
        "proof.stages.stage7_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage7,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "prover-fixture test appends a Stage 7 batch round",
    ),
    checked_standard(
        "stage7.claims.hamming_weight_claim_reduction.instruction_ra",
        "claims.stage7.hamming_weight_claim_reduction.instruction_ra",
        VerifierPhase::Stage7,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets every HammingWeight instruction RA output claim",
    ),
    checked_standard(
        "stage7.claims.hamming_weight_claim_reduction.bytecode_ra",
        "claims.stage7.hamming_weight_claim_reduction.bytecode_ra",
        VerifierPhase::Stage7,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets every HammingWeight bytecode RA output claim",
    ),
    checked_standard(
        "stage7.claims.hamming_weight_claim_reduction.ram_ra",
        "claims.stage7.hamming_weight_claim_reduction.ram_ra",
        VerifierPhase::Stage7,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "prover-fixture test offsets every HammingWeight RAM RA output claim",
    ),
    checked_standard(
        "stage7.claims.trusted_advice.trusted",
        "claims.stage7.trusted_advice.trusted",
        VerifierPhase::Stage7,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "advice fixture test offsets the trusted advice address-phase output claim",
    ),
    checked_standard(
        "stage7.claims.untrusted_advice.untrusted",
        "claims.stage7.untrusted_advice.untrusted",
        VerifierPhase::Stage7,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "advice fixture test offsets the untrusted advice address-phase output claim",
    ),
    checked_standard(
        "stage7.claims.bytecode_address_phase.chunks",
        "claims.stage7.bytecode_address_phase.chunks",
        VerifierPhase::Stage7,
        MutationStrategy::OffsetScalar,
        TamperCoverage::IgnoredUntilFixture,
        "committed fixture test offsets each final bytecode chunk claim",
    ),
    checked_standard(
        "stage7.claims.program_image_address_phase",
        "claims.stage7.program_image_address_phase.program_image",
        VerifierPhase::Stage7,
        MutationStrategy::OffsetScalar,
        TamperCoverage::IgnoredUntilFixture,
        "committed fixture test offsets the final program-image claim",
    ),
];

pub const FUTURE_STAGE_TARGETS: &[TamperTarget] = &[
    final_opening_standard(
        "stage8.opening_claim_values",
        "stage8.opening_claim_values",
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "real verifier fixture offsets final opening claim fields ",
    ),
    final_opening_standard(
        "stage8.opening_claim_points",
        "stage8.opening_claim_points",
        MutationStrategy::OffsetScalar,
        TamperCoverage::Deferred,
        "final opening points are transcript-derived; stage sumcheck payload tampering covers point changes",
    ),
    final_opening_zk(
        "zk.joint_opening_proof.eval_commitment",
        "proof.joint_opening_proof.y_com",
        MutationStrategy::RemoveItem,
        TamperCoverage::Active,
        "Stage 8 ZK opening verification rejects a missing Dory evaluation commitment",
    ),
    zk_target(
        "zk.blindfold_proof",
        "proof.claims.Zk.blindfold_proof",
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "verifier-native ZK fixture mutates the imported BlindFold proof payload ",
    ),
    zk_target(
        "zk.vector_commitment_setup",
        "preprocessing.vc_setup",
        MutationStrategy::ReplaceSetup,
        TamperCoverage::Active,
        "validate_inputs rejects missing VC setup in ZK mode",
    ),
];

/// The Akita-path claim cells: the read-raf fused-inc opening, lattice
/// Booleanity, the fused Stage-7 Hamming reduction, and the stage-8
/// reconstruction leaves. All active: the fixture-driven sweep in
/// `soundness/tampering/akita.rs` (`every_clear_claim_wire_rejects_offset`)
/// offsets every clear-claim scalar of the real packed-prover fixtures and
/// asserts each offset rejects.
#[cfg(feature = "akita")]
pub const AKITA_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "stage6.claims.bytecode_read_raf.fused_inc",
        "claims.stage6b.bytecode_read_raf.fused_inc",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "the lattice read-raf cycle output fold rejects an offset fused-inc opening",
    ),
    checked_standard(
        "stage6.claims.booleanity.unsigned_inc_chunks",
        "claims.stage6b.booleanity.unsigned_inc_chunks",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "the lattice booleanity output fold covers every chunk cell",
    ),
    checked_standard(
        "stage6.claims.booleanity.unsigned_inc_msb",
        "claims.stage6b.booleanity.unsigned_inc_msb",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "the lattice booleanity output fold covers the msb cell",
    ),
    checked_standard(
        "stage7.claims.hamming_weight_claim_reduction.unsigned_inc_chunks",
        "claims.stage7.hamming_weight_claim_reduction.unsigned_inc_chunks",
        VerifierPhase::Stage7,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "the hamming-weight reduction final-claim fold covers every increment chunk",
    ),
    checked_standard(
        "stage7.claims.hamming_weight_claim_reduction.unsigned_inc_msb",
        "claims.stage7.hamming_weight_claim_reduction.unsigned_inc_msb",
        VerifierPhase::Stage7,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "the hamming-weight reduction final-claim fold covers the increment MSB",
    ),
    checked_standard(
        "reconstruction.claims.untrusted_advice",
        "claims.reconstruction.untrusted_advice",
        VerifierPhase::Stage8Openings,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "the reconstruction final-claim fold covers the untrusted advice leaf",
    ),
    checked_standard(
        "reconstruction.claims.trusted_advice",
        "claims.reconstruction.trusted_advice",
        VerifierPhase::Stage8Openings,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "the reconstruction final-claim fold covers the trusted advice leaf",
    ),
    checked_standard(
        "reconstruction.claims.bytecode",
        "claims.reconstruction.bytecode",
        VerifierPhase::Stage8Openings,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "the reconstruction final-claim fold covers every bytecode lane leaf",
    ),
    checked_standard(
        "reconstruction.claims.program_image",
        "claims.reconstruction.program_image",
        VerifierPhase::Stage8Openings,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "the reconstruction final-claim fold covers the program image leaf",
    ),
];

pub fn all_targets() -> Vec<TamperTarget> {
    let targets = PREAMBLE_TARGETS
        .iter()
        .chain(COMMITMENT_TARGETS)
        .chain(PROOF_SHAPE_TARGETS)
        .chain(STAGE1_TARGETS)
        .chain(STAGE2_TARGETS)
        .chain(STAGE3_TARGETS)
        .chain(STAGE4_TARGETS)
        .chain(STAGE5_TARGETS)
        .chain(STAGE6_TARGETS)
        .chain(STAGE7_TARGETS)
        .chain(FUTURE_STAGE_TARGETS);
    #[cfg(feature = "akita")]
    let targets = targets.chain(AKITA_TARGETS);
    targets.copied().collect()
}

#[expect(
    clippy::panic,
    reason = "tamper tests should fail loudly when they reference a missing manifest entry"
)]
pub fn required_target(name: &str) -> TamperTarget {
    all_targets()
        .into_iter()
        .find(|target| target.name == name)
        .unwrap_or_else(|| panic!("missing tamper manifest target {name}"))
}

pub fn target_names_are_unique() -> bool {
    let mut names = BTreeSet::new();
    all_targets()
        .into_iter()
        .all(|target| names.insert(target.name))
}

pub fn manifest_paths() -> BTreeSet<&'static str> {
    all_targets()
        .into_iter()
        .flat_map(expand_manifest_path)
        .collect()
}

#[expect(
    clippy::expect_used,
    reason = "manifest structural tests should fail loudly if claim serialization breaks"
)]
pub fn clear_claim_leaf_paths() -> BTreeSet<String> {
    let claims = clear_claims::<Fr>(true);
    let value = serde_json::to_value(claims).expect("clear claims should serialize");
    let mut paths = BTreeSet::new();
    collect_leaf_paths("claims", &value, &mut paths);
    paths
}

pub fn proof_field_paths() -> &'static [&'static str] {
    &[
        "proof.commitments[*]",
        "proof.joint_opening_proof",
        "proof.untrusted_advice_commitment",
        "proof.claims",
        "proof.trace_length",
        "proof.ram_K",
        "proof.rw_config",
        "proof.one_hot_config",
        "proof.trace_polynomial_order",
        "proof.stages.stage1_uni_skip_first_round_proof.round_polynomials[*]",
        "proof.stages.stage1_sumcheck_proof.round_polynomials[*]",
        "proof.stages.stage2_uni_skip_first_round_proof.round_polynomials[*]",
        "proof.stages.stage2_sumcheck_proof.round_polynomials[*]",
        "proof.stages.stage3_sumcheck_proof.round_polynomials[*]",
        "proof.stages.stage4_sumcheck_proof.round_polynomials[*]",
        "proof.stages.stage5_sumcheck_proof.round_polynomials[*]",
        "proof.stages.stage6a_sumcheck_proof.round_polynomials[*]",
        "proof.stages.stage6b_sumcheck_proof.round_polynomials[*]",
        "proof.stages.stage7_sumcheck_proof.round_polynomials[*]",
    ]
}

pub fn verifier_owned_targets_without_active_coverage() -> Vec<TamperTarget> {
    all_targets()
        .into_iter()
        .filter(|target| {
            target.disposition == TamperDisposition::CheckedAtStage
                && target.coverage != TamperCoverage::Active
        })
        .collect()
}

pub fn assert_manifest_target_is_active(target: TamperTarget) {
    assert_eq!(
        target.coverage,
        TamperCoverage::Active,
        "tamper target is not active: {target:?}"
    );
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
pub fn assert_zk_target_active(name: &str) {
    let target = required_target(name);
    assert_manifest_target_is_active(target);
    assert!(
        target.mode.includes(true),
        "tamper target mode does not include ZK: {target:?}"
    );
}

/// The phase where a target's rejection is documented to fire, derived from
/// its disposition.
pub fn expected_rejection_phase(target: TamperTarget) -> VerifierPhase {
    match target.disposition {
        TamperDisposition::CheckedAtStage => target.checked_at,
        TamperDisposition::CheckedByLaterStage(phase) => phase,
        TamperDisposition::CheckedByFinalOpenings => VerifierPhase::Stage8Openings,
        TamperDisposition::CheckedByZk => VerifierPhase::Zk,
    }
}

/// Which stage's batched sumcheck verifies each relation. Folds 6a/6b into
/// `Stage6`; the stage-7 reductions are listed at stage 7 even though stage 8
/// re-reads their claims for reconstruction.
fn relation_phase(id: JoltRelationId) -> VerifierPhase {
    match id {
        JoltRelationId::SpartanOuter => VerifierPhase::Stage1,
        JoltRelationId::SpartanProductVirtualization
        | JoltRelationId::RamReadWriteChecking
        | JoltRelationId::RamRafEvaluation
        | JoltRelationId::RamOutputCheck
        | JoltRelationId::InstructionClaimReduction => VerifierPhase::Stage2,
        JoltRelationId::SpartanShift
        | JoltRelationId::InstructionInputVirtualization
        | JoltRelationId::RegistersClaimReduction => VerifierPhase::Stage3,
        JoltRelationId::RamValCheck | JoltRelationId::RegistersReadWriteChecking => {
            VerifierPhase::Stage4
        }
        JoltRelationId::InstructionReadRaf
        | JoltRelationId::RamRaClaimReduction
        | JoltRelationId::RegistersValEvaluation => VerifierPhase::Stage5,
        JoltRelationId::BytecodeReadRaf
        | JoltRelationId::Booleanity
        | JoltRelationId::RamHammingBooleanity
        | JoltRelationId::InstructionRaVirtualization
        | JoltRelationId::RamRaVirtualization
        | JoltRelationId::IncClaimReduction
        | JoltRelationId::AdviceClaimReductionCyclePhase
        | JoltRelationId::BytecodeClaimReductionCyclePhase
        | JoltRelationId::ProgramImageClaimReductionCyclePhase => VerifierPhase::Stage6,
        JoltRelationId::AdviceClaimReduction
        | JoltRelationId::BytecodeClaimReduction
        | JoltRelationId::HammingWeightClaimReduction
        | JoltRelationId::ProgramImageClaimReduction => VerifierPhase::Stage7,
        JoltRelationId::UntrustedAdviceReconstruction
        | JoltRelationId::TrustedAdviceReconstruction
        | JoltRelationId::ProgramImageReconstruction
        | JoltRelationId::BytecodeChunkReconstruction => VerifierPhase::Stage8Openings,
    }
}

/// Stage-error strings are `format!("{:?}", relation_id)`; recover the id.
fn relation_from_stage_string(stage: &str) -> Option<JoltRelationId> {
    [
        JoltRelationId::SpartanOuter,
        JoltRelationId::SpartanProductVirtualization,
        JoltRelationId::SpartanShift,
        JoltRelationId::InstructionClaimReduction,
        JoltRelationId::InstructionInputVirtualization,
        JoltRelationId::InstructionReadRaf,
        JoltRelationId::InstructionRaVirtualization,
        JoltRelationId::RamReadWriteChecking,
        JoltRelationId::RamRafEvaluation,
        JoltRelationId::RamOutputCheck,
        JoltRelationId::RamValCheck,
        JoltRelationId::RamRaClaimReduction,
        JoltRelationId::RamHammingBooleanity,
        JoltRelationId::RamRaVirtualization,
        JoltRelationId::RegistersClaimReduction,
        JoltRelationId::RegistersReadWriteChecking,
        JoltRelationId::RegistersValEvaluation,
        JoltRelationId::BytecodeReadRaf,
        JoltRelationId::Booleanity,
        JoltRelationId::AdviceClaimReductionCyclePhase,
        JoltRelationId::AdviceClaimReduction,
        JoltRelationId::BytecodeClaimReductionCyclePhase,
        JoltRelationId::BytecodeClaimReduction,
        JoltRelationId::ProgramImageClaimReductionCyclePhase,
        JoltRelationId::ProgramImageClaimReduction,
        JoltRelationId::IncClaimReduction,
        JoltRelationId::HammingWeightClaimReduction,
        JoltRelationId::UntrustedAdviceReconstruction,
        JoltRelationId::TrustedAdviceReconstruction,
        JoltRelationId::ProgramImageReconstruction,
        JoltRelationId::BytecodeChunkReconstruction,
    ]
    .into_iter()
    .find(|id| format!("{id:?}") == stage)
}

/// Maps a rejection to the verifier phase that raised it, where the error
/// variant carries enough information. `None` means the variant is
/// phase-agnostic (e.g. a missing claim id observed wherever it is first
/// consumed) and the caller should skip phase attribution.
pub fn observed_rejection_phase(error: &VerifierError) -> Option<VerifierPhase> {
    match error {
        VerifierError::ProtocolConfigMismatch { .. }
        | VerifierError::ExpectedClearProof { .. }
        | VerifierError::ExpectedCommittedProof { .. }
        | VerifierError::UnexpectedBlindFoldProof
        | VerifierError::MissingBlindFoldProof
        | VerifierError::UnexpectedOpeningClaims
        | VerifierError::MissingVectorCommitmentSetup
        | VerifierError::InvalidVectorCommitmentCapacity { .. }
        | VerifierError::MemoryLayoutMismatch
        | VerifierError::InputTooLarge { .. }
        | VerifierError::OutputTooLarge { .. }
        | VerifierError::InvalidTraceLength { .. }
        | VerifierError::InvalidRamK { .. }
        | VerifierError::InvalidMemoryLayout { .. }
        | VerifierError::InvalidPrecommittedSchedule { .. }
        | VerifierError::InvalidCommittedProgram { .. } => Some(VerifierPhase::Preamble),
        VerifierError::StageClaimSumcheckFailed { stage, .. }
        | VerifierError::StageClaimOpeningMismatch { stage, .. } => {
            relation_from_stage_string(stage).map(relation_phase)
        }
        VerifierError::StageClaimPublicInputFailed { stage, .. } => Some(relation_phase(*stage)),
        VerifierError::StageClaimOutputMismatch { stage } => match stage {
            1 => Some(VerifierPhase::Stage1),
            2 => Some(VerifierPhase::Stage2),
            3 => Some(VerifierPhase::Stage3),
            4 => Some(VerifierPhase::Stage4),
            5 => Some(VerifierPhase::Stage5),
            6 => Some(VerifierPhase::Stage6),
            7 => Some(VerifierPhase::Stage7),
            _ => None,
        },
        VerifierError::InvalidCommitmentCount { .. }
        | VerifierError::MissingFinalOpeningCommitment { .. }
        | VerifierError::FinalOpeningBatchFailed { .. }
        | VerifierError::FinalOpeningVerificationFailed { .. } => {
            Some(VerifierPhase::Stage8Openings)
        }
        VerifierError::BlindFoldConstructionFailed { .. }
        | VerifierError::BlindFoldVerificationFailed { .. } => Some(VerifierPhase::Zk),
        VerifierError::MissingOpeningClaim { .. }
        | VerifierError::UnexpectedOpeningClaim { .. }
        | VerifierError::MissingStageClaimChallenge { .. }
        | VerifierError::MissingStageClaimDerived { .. }
        | VerifierError::ChallengeDraw(_) => None,
    }
}

#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "zk"),
    not(feature = "akita")
))]
pub fn assert_verifier_fixture_tamper_rejects(
    target: TamperTarget,
    base: &crate::support::verifier_fixtures::VerifierFixtureCase,
    mutate: impl FnOnce(&mut crate::support::verifier_fixtures::VerifierFixtureCase),
) {
    assert_manifest_target_is_active(target);
    let mut case = base.clone();
    mutate(&mut case);
    let result = case.verify();
    assert!(
        result.is_err(),
        "tampered standard proof was accepted for target {}",
        target.name
    );
    // The manifest documents the LAST line of defense: rejection may fire
    // earlier (transcript-bound values diverge challenges at the first
    // post-absorption check) but never later than the documented phase.
    if let Err(error) = result {
        if let Some(observed) = observed_rejection_phase(&error) {
            let expected = expected_rejection_phase(target);
            assert!(
                observed <= expected,
                "target {} was rejected in {observed:?}, later than the manifest's \
                 documented {expected:?} (disposition {:?}); error: {error}",
                target.name,
                target.disposition
            );
        }
    }
}

fn expand_manifest_path(target: TamperTarget) -> Vec<&'static str> {
    match target.path {
        "claims.stage1.outer.*" => vec![
            "claims.stage1.outer.outer_remainder.left_instruction_input",
            "claims.stage1.outer.outer_remainder.right_instruction_input",
            "claims.stage1.outer.outer_remainder.product",
            "claims.stage1.outer.outer_remainder.should_branch",
            "claims.stage1.outer.outer_remainder.pc",
            "claims.stage1.outer.outer_remainder.unexpanded_pc",
            "claims.stage1.outer.outer_remainder.imm",
            "claims.stage1.outer.outer_remainder.ram_address",
            "claims.stage1.outer.outer_remainder.rs1_value",
            "claims.stage1.outer.outer_remainder.rs2_value",
            "claims.stage1.outer.outer_remainder.rd_write_value",
            "claims.stage1.outer.outer_remainder.ram_read_value",
            "claims.stage1.outer.outer_remainder.ram_write_value",
            "claims.stage1.outer.outer_remainder.left_lookup_operand",
            "claims.stage1.outer.outer_remainder.right_lookup_operand",
            "claims.stage1.outer.outer_remainder.next_unexpanded_pc",
            "claims.stage1.outer.outer_remainder.next_pc",
            "claims.stage1.outer.outer_remainder.next_is_virtual",
            "claims.stage1.outer.outer_remainder.next_is_first_in_sequence",
            "claims.stage1.outer.outer_remainder.lookup_output",
            "claims.stage1.outer.outer_remainder.should_jump",
            "claims.stage1.outer.outer_remainder.add_operands",
            "claims.stage1.outer.outer_remainder.subtract_operands",
            "claims.stage1.outer.outer_remainder.multiply_operands",
            "claims.stage1.outer.outer_remainder.load",
            "claims.stage1.outer.outer_remainder.store",
            "claims.stage1.outer.outer_remainder.jump",
            "claims.stage1.outer.outer_remainder.write_lookup_output_to_rd",
            "claims.stage1.outer.outer_remainder.virtual_instruction",
            "claims.stage1.outer.outer_remainder.assert",
            "claims.stage1.outer.outer_remainder.do_not_update_unexpanded_pc",
            "claims.stage1.outer.outer_remainder.advice",
            "claims.stage1.outer.outer_remainder.is_compressed",
            "claims.stage1.outer.outer_remainder.is_first_in_sequence",
            "claims.stage1.outer.outer_remainder.is_last_in_sequence",
        ],
        "claims.stage2.batch_outputs.ram_read_write.*" => vec![
            "claims.stage2.batch_outputs.ram_read_write.val",
            "claims.stage2.batch_outputs.ram_read_write.ra",
            "claims.stage2.batch_outputs.ram_read_write.inc",
        ],
        "claims.stage2.batch_outputs.product_remainder.{left_instruction_input,right_instruction_input,jump_flag,lookup_output,branch_flag,next_is_noop}" => vec![
            "claims.stage2.batch_outputs.product_remainder.left_instruction_input",
            "claims.stage2.batch_outputs.product_remainder.right_instruction_input",
            "claims.stage2.batch_outputs.product_remainder.jump_flag",
            "claims.stage2.batch_outputs.product_remainder.lookup_output",
            "claims.stage2.batch_outputs.product_remainder.branch_flag",
            "claims.stage2.batch_outputs.product_remainder.next_is_noop",
        ],
        "claims.stage2.batch_outputs.instruction_claim_reduction.*" => vec![
            "claims.stage2.batch_outputs.instruction_claim_reduction.lookup_output",
            "claims.stage2.batch_outputs.instruction_claim_reduction.left_lookup_operand",
            "claims.stage2.batch_outputs.instruction_claim_reduction.right_lookup_operand",
            "claims.stage2.batch_outputs.instruction_claim_reduction.left_instruction_input",
            "claims.stage2.batch_outputs.instruction_claim_reduction.right_instruction_input",
        ],
        "claims.stage3.shift.*" => vec![
            "claims.stage3.shift.unexpanded_pc",
            "claims.stage3.shift.pc",
            "claims.stage3.shift.is_virtual",
            "claims.stage3.shift.is_first_in_sequence",
            "claims.stage3.shift.is_noop",
        ],
        "claims.stage3.instruction_input.*" => vec![
            "claims.stage3.instruction_input.left_operand_is_rs1",
            "claims.stage3.instruction_input.rs1_value",
            "claims.stage3.instruction_input.left_operand_is_pc",
            "claims.stage3.instruction_input.unexpanded_pc",
            "claims.stage3.instruction_input.right_operand_is_rs2",
            "claims.stage3.instruction_input.rs2_value",
            "claims.stage3.instruction_input.right_operand_is_imm",
            "claims.stage3.instruction_input.imm",
        ],
        "claims.stage3.registers_claim_reduction.*" => vec![
            "claims.stage3.registers_claim_reduction.rd_write_value",
            "claims.stage3.registers_claim_reduction.rs1_value",
            "claims.stage3.registers_claim_reduction.rs2_value",
        ],
        "claims.stage4.registers_read_write.*" => vec![
            "claims.stage4.registers_read_write.registers_val",
            "claims.stage4.registers_read_write.rs1_ra",
            "claims.stage4.registers_read_write.rs2_ra",
            "claims.stage4.registers_read_write.rd_wa",
            "claims.stage4.registers_read_write.rd_inc",
        ],
        "claims.stage4.ram_val_check.*" => vec![
            "claims.stage4.ram_val_check.ram_ra",
            "claims.stage4.ram_val_check.ram_inc",
        ],
        "claims.stage5.registers_val_evaluation.*" => vec![
            "claims.stage5.registers_val_evaluation.rd_inc",
            "claims.stage5.registers_val_evaluation.rd_wa",
        ],
        path => vec![path],
    }
}

fn collect_leaf_paths(prefix: &str, value: &Value, paths: &mut BTreeSet<String>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                collect_leaf_paths(&format!("{prefix}.{key}"), value, paths);
            }
        }
        _ => {
            let _ = paths.insert(prefix.to_string());
        }
    }
}

pub fn clear_claims<F: Field>(fill_optionals: bool) -> ClearProofClaims<F> {
    let zero = F::zero();
    let optional = fill_optionals.then_some(zero);

    ClearProofClaims {
        #[cfg(feature = "akita")]
        reconstruction: jolt_verifier::stages::stage8::reconstruction::ReconstructionOutputClaims {
            untrusted_advice: None,
            trusted_advice: None,
            bytecode: None,
            program_image: None,
        },
        stage1: stage1::outputs::Stage1OutputClaims {
            uniskip_output_claim: zero,
            outer: stage1::outputs::Stage1BatchOutputClaims {
                outer_remainder: stage1::OuterRemainderOutputClaims {
                    left_instruction_input: zero,
                    right_instruction_input: zero,
                    product: zero,
                    should_branch: zero,
                    pc: zero,
                    unexpanded_pc: zero,
                    imm: zero,
                    ram_address: zero,
                    rs1_value: zero,
                    rs2_value: zero,
                    rd_write_value: zero,
                    ram_read_value: zero,
                    ram_write_value: zero,
                    left_lookup_operand: zero,
                    right_lookup_operand: zero,
                    next_unexpanded_pc: zero,
                    next_pc: zero,
                    next_is_virtual: zero,
                    next_is_first_in_sequence: zero,
                    lookup_output: zero,
                    should_jump: zero,
                    add_operands: zero,
                    subtract_operands: zero,
                    multiply_operands: zero,
                    load: zero,
                    store: zero,
                    jump: zero,
                    write_lookup_output_to_rd: zero,
                    virtual_instruction: zero,
                    assert: zero,
                    do_not_update_unexpanded_pc: zero,
                    advice: zero,
                    is_compressed: zero,
                    is_first_in_sequence: zero,
                    is_last_in_sequence: zero,
                },
            },
        },
        stage2: stage2::outputs::Stage2OutputClaims {
            product_uniskip_output_claim: zero,
            batch_outputs: stage2::outputs::Stage2BatchOutputClaims {
                ram_read_write: stage2::outputs::RamReadWriteOutputClaims {
                    val: zero,
                    ra: zero,
                    inc: zero,
                },
                product_remainder: stage2::outputs::ProductRemainderOutputClaims {
                    left_instruction_input: zero,
                    right_instruction_input: zero,
                    jump_flag: zero,
                    write_lookup_output_to_rd: zero,
                    lookup_output: zero,
                    branch_flag: zero,
                    next_is_noop: zero,
                    virtual_instruction: zero,
                },
                instruction_claim_reduction:
                    stage2::outputs::InstructionClaimReductionOutputClaims {
                        lookup_output: zero,
                        left_lookup_operand: zero,
                        right_lookup_operand: zero,
                        left_instruction_input: zero,
                        right_instruction_input: zero,
                    },
                ram_raf_evaluation: stage2::outputs::RamRafEvaluationOutputClaims { ram_ra: zero },
                ram_output_check: stage2::outputs::RamOutputCheckOutputClaims { val_final: zero },
            },
        },
        stage3: stage3::outputs::Stage3OutputClaims {
            shift: stage3::outputs::SpartanShiftOutputClaims {
                unexpanded_pc: zero,
                pc: zero,
                is_virtual: zero,
                is_first_in_sequence: zero,
                is_noop: zero,
            },
            instruction_input: stage3::outputs::InstructionInputOutputClaims {
                left_operand_is_rs1: zero,
                rs1_value: zero,
                left_operand_is_pc: zero,
                unexpanded_pc: zero,
                right_operand_is_rs2: zero,
                rs2_value: zero,
                right_operand_is_imm: zero,
                imm: zero,
            },
            registers_claim_reduction: stage3::outputs::RegistersClaimReductionOutputClaims {
                rd_write_value: zero,
                rs1_value: zero,
                rs2_value: zero,
            },
        },
        stage4: stage4::outputs::Stage4OutputClaims {
            registers_read_write: stage4::RegistersReadWriteOutputClaims {
                registers_val: zero,
                rs1_ra: zero,
                rs2_ra: zero,
                rd_wa: zero,
                rd_inc: zero,
            },
            ram_val_check: stage4::RamValCheckOutputClaims {
                untrusted_advice: optional,
                trusted_advice: optional,
                program_image: None,
                ram_ra: zero,
                ram_inc: zero,
            },
        },
        stage5: stage5::outputs::Stage5OutputClaims {
            instruction_read_raf: stage5::InstructionReadRafOutputClaims {
                lookup_table_flags: vec![zero],
                instruction_ra: vec![zero],
                instruction_raf_flag: zero,
            },
            ram_ra_claim_reduction: stage5::RamRaClaimReductionOutputClaims { ram_ra: zero },
            registers_val_evaluation: stage5::RegistersValEvaluationOutputClaims {
                rd_inc: zero,
                rd_wa: zero,
            },
        },
        stage6a: stage6a::outputs::Stage6aOutputClaims {
            bytecode_read_raf: stage6a::outputs::BytecodeReadRafAddressPhaseOutputClaims {
                intermediate: zero,
                val_stages: Vec::new(),
            },
            booleanity: stage6a::outputs::BooleanityAddressPhaseOutputClaims {
                intermediate: zero,
            },
        },
        stage6b: stage6b::outputs::Stage6bOutputClaims {
            #[cfg(not(feature = "akita"))]
            bytecode_read_raf: stage6b::outputs::BytecodeReadRafOutputClaims {
                bytecode_ra: vec![zero],
            },
            #[cfg(feature = "akita")]
            bytecode_read_raf:
                stage6b::bytecode_read_raf::LatticeBytecodeReadRafOutputClaims {
                    bytecode_ra: vec![zero],
                    fused_inc: zero,
                },
            #[cfg(not(feature = "akita"))]
            booleanity: stage6b::outputs::BooleanityOutputClaims {
                instruction_ra: vec![zero],
                bytecode_ra: vec![zero],
                ram_ra: vec![zero],
            },
            #[cfg(feature = "akita")]
            booleanity:
                jolt_claims::protocols::jolt::lattice::relations::booleanity::LatticeBooleanityOutputClaims {
                    instruction_ra: vec![zero],
                    bytecode_ra: vec![zero],
                    ram_ra: vec![zero],
                    unsigned_inc_chunks: vec![zero],
                    unsigned_inc_msb: zero,
                },
            ram_hamming_booleanity: stage6b::outputs::RamHammingBooleanityOutputClaims {
                ram_hamming_weight: zero,
            },
            ram_ra_virtualization: stage6b::outputs::RamRaVirtualizationOutputClaims {
                ram_ra: vec![zero],
            },
            instruction_ra_virtualization:
                stage6b::outputs::InstructionRaVirtualizationOutputClaims {
                    committed_instruction_ra: vec![zero],
                },
            #[cfg(not(feature = "akita"))]
            inc_claim_reduction: stage6b::outputs::IncClaimReductionOutputClaims {
                ram_inc: zero,
                rd_inc: zero,
            },
            trusted_advice: fill_optionals.then_some(
                stage6b::outputs::TrustedAdviceCyclePhaseOutputClaims { trusted: zero },
            ),
            untrusted_advice: fill_optionals.then_some(
                stage6b::outputs::UntrustedAdviceCyclePhaseOutputClaims { untrusted: zero },
            ),
            bytecode_reduction: fill_optionals.then_some(
                stage6b::outputs::BytecodeReductionCyclePhaseOutputClaims {
                    intermediate: Some(zero),
                    chunks: Vec::new(),
                },
            ),
            program_image_reduction: fill_optionals.then_some(
                stage6b::outputs::ProgramImageReductionCyclePhaseOutputClaims {
                    program_image: zero,
                },
            ),
        },
        stage7: stage7::outputs::Stage7OutputClaims {
            hamming_weight_claim_reduction:
                stage7::hamming_weight_claim_reduction::HammingWeightClaimReductionOutputClaims {
                    instruction_ra: vec![zero],
                    bytecode_ra: vec![zero],
                    ram_ra: vec![zero],
                    #[cfg(feature = "akita")]
                    unsigned_inc_chunks: vec![zero],
                    #[cfg(feature = "akita")]
                    unsigned_inc_msb: zero,
                },
            trusted_advice: fill_optionals.then_some(
                stage7::advice_address_phase::TrustedAdviceAddressPhaseOutputClaims {
                    trusted: zero,
                },
            ),
            untrusted_advice: fill_optionals.then_some(
                stage7::advice_address_phase::UntrustedAdviceAddressPhaseOutputClaims {
                    untrusted: zero,
                },
            ),
            bytecode_address_phase: fill_optionals.then_some(
                stage7::committed_reduction_address_phase::BytecodeReductionAddressPhaseOutputClaims {
                    chunks: vec![zero],
                },
            ),
            program_image_address_phase: fill_optionals.then_some(
                stage7::committed_reduction_address_phase::ProgramImageReductionAddressPhaseOutputClaims {
                    program_image: zero,
                },
            ),
        },
    }
}
