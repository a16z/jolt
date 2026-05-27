use std::collections::BTreeSet;
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use std::panic::{catch_unwind, AssertUnwindSafe};

use jolt_field::{Fr, FromPrimitiveInt};
use jolt_verifier::{
    proof::ClearProofClaims,
    stages::{stage1, stage2, stage3, stage4, stage5, stage6, stage7},
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
pub enum TamperDisposition {
    CheckedAtStage,
    CheckedByLaterStage(VerifierPhase),
    CheckedByFinalOpenings,
    CheckedByZk,
    NotVerifierOwned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutationStrategy {
    OffsetScalar,
    FlipBool,
    ChangeEnumVariant,
    RemoveItem,
    AddItem,
    DuplicateItem,
    SwapOrder,
    TruncateVector,
    ExtendVector,
    ReplaceProofPayload,
    ReplaceModePayload,
    ReplaceSetup,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TamperCoverage {
    Active,
    IgnoredUntilFixture,
    Deferred,
    NotApplicable,
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
        TamperCoverage::IgnoredUntilFixture,
        "value-level output mutation should be run on real core fixtures and rejects at RAM output check",
    ),
    checked_standard(
        "public_io.panic",
        "public_io.panic",
        VerifierPhase::Stage2,
        MutationStrategy::FlipBool,
        TamperCoverage::IgnoredUntilFixture,
        "panic materializes as public memory for the RAM output check",
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
        TamperCoverage::Deferred,
        "one-hot configuration is transcript-bound now but substantively checked by later RA virtualization stages",
    ),
    checked_standard(
        "proof.trace_polynomial_order",
        "proof.trace_polynomial_order",
        VerifierPhase::Stage1,
        MutationStrategy::ChangeEnumVariant,
        TamperCoverage::IgnoredUntilFixture,
        "layout order is transcript-bound; add a real fixture mutation before relying on it",
    ),
    checked_standard(
        "preprocessing.preprocessing_digest",
        "preprocessing.preprocessing_digest",
        VerifierPhase::Stage1,
        MutationStrategy::OffsetScalar,
        TamperCoverage::IgnoredUntilFixture,
        "digest is transcript-bound; real fixture mutation should reject at the first stage",
    ),
    checked_standard(
        "preprocessing.program.bytecode.entry_address",
        "preprocessing.program.bytecode.entry_address",
        VerifierPhase::Stage1,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Deferred,
        "entry address is transcript-bound but needs a real bytecode fixture mutation",
    ),
];

pub const COMMITMENT_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "proof.commitments.value",
        "proof.commitments[*]",
        VerifierPhase::Stage1,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "real core fixture replaces one commitment with another and rejects before acceptance",
    ),
    checked_standard(
        "proof.commitments.missing",
        "proof.commitments",
        VerifierPhase::Stage1,
        MutationStrategy::RemoveItem,
        TamperCoverage::Active,
        "real core fixture removes one commitment and rejects before acceptance",
    ),
    checked_standard(
        "proof.commitments.extra",
        "proof.commitments",
        VerifierPhase::Stage1,
        MutationStrategy::DuplicateItem,
        TamperCoverage::Active,
        "real core fixture duplicates one commitment and rejects before acceptance",
    ),
    checked_standard(
        "proof.commitments.order",
        "proof.commitments",
        VerifierPhase::Stage1,
        MutationStrategy::SwapOrder,
        TamperCoverage::Active,
        "real core fixture swaps commitments and rejects before acceptance",
    ),
    checked_standard(
        "proof.untrusted_advice_commitment",
        "proof.untrusted_advice_commitment",
        VerifierPhase::Stage1,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Deferred,
        "advice commitment coverage belongs with advice fixtures",
    ),
    checked_standard(
        "trusted_advice_commitment",
        "trusted_advice_commitment",
        VerifierPhase::Stage1,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Deferred,
        "trusted advice commitment coverage belongs with advice fixtures",
    ),
    final_opening_standard(
        "proof.joint_opening_proof",
        "proof.joint_opening_proof",
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "real core fixture replaces the joint opening proof with another valid proof payload",
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
        "core-fixture test mutates every full uni-skip round polynomial",
    ),
    checked_standard(
        "stage1.uni_skip.round_count.missing",
        "proof.stages.stage1_uni_skip_first_round_proof.round_polynomials",
        VerifierPhase::Stage1,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "core-fixture test removes a uni-skip round",
    ),
    checked_standard(
        "stage1.uni_skip.round_count.extra",
        "proof.stages.stage1_uni_skip_first_round_proof.round_polynomials",
        VerifierPhase::Stage1,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "core-fixture test appends a uni-skip round",
    ),
    checked_standard(
        "stage1.remainder.round_polynomial",
        "proof.stages.stage1_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage1,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "core-fixture test mutates every compressed remainder round polynomial",
    ),
    checked_standard(
        "stage1.remainder.round_count.missing",
        "proof.stages.stage1_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage1,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "core-fixture test removes a remainder round",
    ),
    checked_standard(
        "stage1.remainder.round_count.extra",
        "proof.stages.stage1_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage1,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "core-fixture test appends a remainder round",
    ),
    checked_standard(
        "stage1.claims.uniskip_output_claim",
        "claims.stage1.uniskip_output_claim",
        VerifierPhase::Stage1,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets the Spartan outer uni-skip opening claim",
    ),
    checked_standard(
        "stage1.claims.outer",
        "claims.stage1.outer.*",
        VerifierPhase::Stage1,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets every Spartan outer variable opening claim",
    ),
];

pub const STAGE2_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "stage2.product_uniskip.round_polynomial",
        "proof.stages.stage2_uni_skip_first_round_proof.round_polynomials[*]",
        VerifierPhase::Stage2,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "core-fixture test mutates every product uni-skip round polynomial",
    ),
    checked_standard(
        "stage2.product_uniskip.round_count.missing",
        "proof.stages.stage2_uni_skip_first_round_proof.round_polynomials",
        VerifierPhase::Stage2,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "core-fixture test removes a product uni-skip round",
    ),
    checked_standard(
        "stage2.product_uniskip.round_count.extra",
        "proof.stages.stage2_uni_skip_first_round_proof.round_polynomials",
        VerifierPhase::Stage2,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "core-fixture test appends a product uni-skip round",
    ),
    checked_standard(
        "stage2.batch.round_polynomial",
        "proof.stages.stage2_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage2,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "core-fixture test mutates every compressed Stage 2 batch round polynomial",
    ),
    checked_standard(
        "stage2.batch.round_count.missing",
        "proof.stages.stage2_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage2,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "core-fixture test removes a Stage 2 batch round",
    ),
    checked_standard(
        "stage2.batch.round_count.extra",
        "proof.stages.stage2_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage2,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "core-fixture test appends a Stage 2 batch round",
    ),
    checked_standard(
        "stage2.claims.product_uniskip_output_claim",
        "claims.stage2.product_uniskip_output_claim",
        VerifierPhase::Stage2,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets the product uni-skip opening claim",
    ),
    checked_standard(
        "stage2.claims.batch_outputs.ram_read_write",
        "claims.stage2.batch_outputs.ram_read_write.*",
        VerifierPhase::Stage2,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets each RAM read-write output claim",
    ),
    checked_standard(
        "stage2.claims.batch_outputs.product_remainder.checked",
        "claims.stage2.batch_outputs.product_remainder.{left_instruction_input,right_instruction_input,jump_flag,lookup_output,branch_flag,next_is_noop}",
        VerifierPhase::Stage2,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets every product remainder output used by the Stage 2 formula",
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
        "core-fixture test offsets each instruction claim-reduction output or its product alias",
    ),
    checked_standard(
        "stage2.claims.batch_outputs.ram_raf_evaluation",
        "claims.stage2.batch_outputs.ram_raf_evaluation",
        VerifierPhase::Stage2,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets the RAM RAF evaluation output claim",
    ),
    checked_standard(
        "stage2.claims.batch_outputs.ram_output_check",
        "claims.stage2.batch_outputs.ram_output_check",
        VerifierPhase::Stage2,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets the RAM output-check claim",
    ),
];

pub const STAGE3_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "stage3.batch.round_polynomial",
        "proof.stages.stage3_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage3,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "core-fixture test mutates every compressed Stage 3 batch round polynomial",
    ),
    checked_standard(
        "stage3.batch.round_count.missing",
        "proof.stages.stage3_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage3,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "core-fixture test removes a Stage 3 batch round",
    ),
    checked_standard(
        "stage3.batch.round_count.extra",
        "proof.stages.stage3_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage3,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "core-fixture test appends a Stage 3 batch round",
    ),
    checked_standard(
        "stage3.claims.shift",
        "claims.stage3.shift.*",
        VerifierPhase::Stage3,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets each Spartan shift output claim",
    ),
    checked_standard(
        "stage3.claims.instruction_input",
        "claims.stage3.instruction_input.*",
        VerifierPhase::Stage3,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets each instruction-input output claim",
    ),
    checked_standard(
        "stage3.claims.registers_claim_reduction",
        "claims.stage3.registers_claim_reduction.*",
        VerifierPhase::Stage3,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets each register claim-reduction output claim",
    ),
];

pub const STAGE4_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "stage4.batch.round_polynomial",
        "proof.stages.stage4_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage4,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "core-fixture test mutates every compressed Stage 4 batch round polynomial",
    ),
    checked_standard(
        "stage4.batch.round_count.missing",
        "proof.stages.stage4_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage4,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "core-fixture test removes a Stage 4 batch round",
    ),
    checked_standard(
        "stage4.batch.round_count.extra",
        "proof.stages.stage4_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage4,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "core-fixture test appends a Stage 4 batch round",
    ),
    checked_standard(
        "stage4.claims.registers_read_write",
        "claims.stage4.registers_read_write.*",
        VerifierPhase::Stage4,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets each register read-write output claim",
    ),
    checked_standard(
        "stage4.claims.ram_val_check",
        "claims.stage4.ram_val_check.*",
        VerifierPhase::Stage4,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets each RAM value-check output claim",
    ),
    checked_standard(
        "stage4.claims.advice.untrusted",
        "claims.stage4.advice.untrusted",
        VerifierPhase::Stage4,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "advice fixture test offsets the untrusted advice opening consumed by RAM value check",
    ),
    checked_standard(
        "stage4.claims.advice.trusted",
        "claims.stage4.advice.trusted",
        VerifierPhase::Stage4,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "advice fixture test offsets the trusted advice opening consumed by RAM value check",
    ),
];

pub const STAGE5_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "stage5.batch.round_polynomial",
        "proof.stages.stage5_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage5,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "core-fixture test mutates every compressed Stage 5 batch round polynomial",
    ),
    checked_standard(
        "stage5.batch.round_count.missing",
        "proof.stages.stage5_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage5,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "core-fixture test removes a Stage 5 batch round",
    ),
    checked_standard(
        "stage5.batch.round_count.extra",
        "proof.stages.stage5_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage5,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "core-fixture test appends a Stage 5 batch round",
    ),
    checked_standard(
        "stage5.claims.instruction_read_raf.lookup_table_flags",
        "claims.stage5.instruction_read_raf.lookup_table_flags",
        VerifierPhase::Stage5,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets each instruction read-RAF lookup-table flag claim",
    ),
    checked_standard(
        "stage5.claims.instruction_read_raf.instruction_ra",
        "claims.stage5.instruction_read_raf.instruction_ra",
        VerifierPhase::Stage5,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets each instruction read-RAF virtual RA claim",
    ),
    checked_standard(
        "stage5.claims.instruction_read_raf.instruction_raf_flag",
        "claims.stage5.instruction_read_raf.instruction_raf_flag",
        VerifierPhase::Stage5,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets the instruction read-RAF flag claim",
    ),
    checked_standard(
        "stage5.claims.ram_ra_claim_reduction",
        "claims.stage5.ram_ra_claim_reduction.ram_ra",
        VerifierPhase::Stage5,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets the RAM RA claim-reduction output claim",
    ),
    checked_standard(
        "stage5.claims.registers_val_evaluation",
        "claims.stage5.registers_val_evaluation.*",
        VerifierPhase::Stage5,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets each register value-evaluation output claim",
    ),
];

pub const STAGE6_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "stage6.address_phase.round_polynomial",
        "proof.stages.stage6a_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage6,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "core-fixture test mutates every compressed Stage 6 address-phase round polynomial",
    ),
    checked_standard(
        "stage6.address_phase.round_count.missing",
        "proof.stages.stage6a_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage6,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "core-fixture test removes a Stage 6 address-phase round",
    ),
    checked_standard(
        "stage6.address_phase.round_count.extra",
        "proof.stages.stage6a_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage6,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "core-fixture test appends a Stage 6 address-phase round",
    ),
    checked_standard(
        "stage6.cycle_phase.round_polynomial",
        "proof.stages.stage6b_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage6,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "core-fixture test mutates every compressed Stage 6 cycle-phase round polynomial",
    ),
    checked_standard(
        "stage6.cycle_phase.round_count.missing",
        "proof.stages.stage6b_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage6,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "core-fixture test removes a Stage 6 cycle-phase round",
    ),
    checked_standard(
        "stage6.cycle_phase.round_count.extra",
        "proof.stages.stage6b_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage6,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "core-fixture test appends a Stage 6 cycle-phase round",
    ),
    checked_standard(
        "stage6.claims.address_phase.bytecode_read_raf",
        "claims.stage6.address_phase.bytecode_read_raf",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets the Stage 6 bytecode read-RAF address-phase output claim",
    ),
    checked_standard(
        "stage6.claims.address_phase.booleanity",
        "claims.stage6.address_phase.booleanity",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets the Stage 6 Booleanity address-phase output claim",
    ),
    checked_standard(
        "stage6.claims.bytecode_read_raf.bytecode_ra",
        "claims.stage6.bytecode_read_raf.bytecode_ra",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets every bytecode read-RAF output claim",
    ),
    checked_standard(
        "stage6.claims.booleanity.instruction_ra",
        "claims.stage6.booleanity.instruction_ra",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets every Booleanity instruction RA output claim",
    ),
    checked_standard(
        "stage6.claims.booleanity.bytecode_ra",
        "claims.stage6.booleanity.bytecode_ra",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets every Booleanity bytecode RA output claim",
    ),
    checked_standard(
        "stage6.claims.booleanity.ram_ra",
        "claims.stage6.booleanity.ram_ra",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets every Booleanity RAM RA output claim",
    ),
    checked_standard(
        "stage6.claims.ram_hamming_booleanity.ram_hamming_weight",
        "claims.stage6.ram_hamming_booleanity.ram_hamming_weight",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets the RAM hamming Booleanity output claim",
    ),
    checked_standard(
        "stage6.claims.ram_ra_virtualization.ram_ra",
        "claims.stage6.ram_ra_virtualization.ram_ra",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets every RAM RA virtualization output claim",
    ),
    checked_standard(
        "stage6.claims.instruction_ra_virtualization.committed_instruction_ra",
        "claims.stage6.instruction_ra_virtualization.committed_instruction_ra",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets every instruction RA virtualization output claim",
    ),
    checked_standard(
        "stage6.claims.inc_claim_reduction.ram_inc",
        "claims.stage6.inc_claim_reduction.ram_inc",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets the RAM increment reduction output claim",
    ),
    checked_standard(
        "stage6.claims.inc_claim_reduction.rd_inc",
        "claims.stage6.inc_claim_reduction.rd_inc",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets the register increment reduction output claim",
    ),
    checked_standard(
        "stage6.claims.advice_cycle_phase.trusted.opening_claim",
        "claims.stage6.advice_cycle_phase.trusted.opening_claim",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "advice fixture test offsets the trusted advice cycle-phase output claim",
    ),
    checked_standard(
        "stage6.claims.advice_cycle_phase.untrusted.opening_claim",
        "claims.stage6.advice_cycle_phase.untrusted.opening_claim",
        VerifierPhase::Stage6,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "advice fixture test offsets the untrusted advice cycle-phase output claim",
    ),
];

pub const STAGE7_TARGETS: &[TamperTarget] = &[
    checked_standard(
        "stage7.batch.round_polynomial",
        "proof.stages.stage7_sumcheck_proof.round_polynomials[*]",
        VerifierPhase::Stage7,
        MutationStrategy::ReplaceProofPayload,
        TamperCoverage::Active,
        "core-fixture test mutates every compressed Stage 7 batch round polynomial",
    ),
    checked_standard(
        "stage7.batch.round_count.missing",
        "proof.stages.stage7_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage7,
        MutationStrategy::TruncateVector,
        TamperCoverage::Active,
        "core-fixture test removes a Stage 7 batch round",
    ),
    checked_standard(
        "stage7.batch.round_count.extra",
        "proof.stages.stage7_sumcheck_proof.round_polynomials",
        VerifierPhase::Stage7,
        MutationStrategy::ExtendVector,
        TamperCoverage::Active,
        "core-fixture test appends a Stage 7 batch round",
    ),
    checked_standard(
        "stage7.claims.hamming_weight_claim_reduction.instruction_ra",
        "claims.stage7.hamming_weight_claim_reduction.instruction_ra",
        VerifierPhase::Stage7,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets every HammingWeight instruction RA output claim",
    ),
    checked_standard(
        "stage7.claims.hamming_weight_claim_reduction.bytecode_ra",
        "claims.stage7.hamming_weight_claim_reduction.bytecode_ra",
        VerifierPhase::Stage7,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets every HammingWeight bytecode RA output claim",
    ),
    checked_standard(
        "stage7.claims.hamming_weight_claim_reduction.ram_ra",
        "claims.stage7.hamming_weight_claim_reduction.ram_ra",
        VerifierPhase::Stage7,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "core-fixture test offsets every HammingWeight RAM RA output claim",
    ),
    checked_standard(
        "stage7.claims.advice_address_phase.trusted.opening_claim",
        "claims.stage7.advice_address_phase.trusted.opening_claim",
        VerifierPhase::Stage7,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "advice fixture test offsets the trusted advice address-phase output claim",
    ),
    checked_standard(
        "stage7.claims.advice_address_phase.untrusted.opening_claim",
        "claims.stage7.advice_address_phase.untrusted.opening_claim",
        VerifierPhase::Stage7,
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "advice fixture test offsets the untrusted advice address-phase output claim",
    ),
];

pub const FUTURE_STAGE_TARGETS: &[TamperTarget] = &[
    final_opening_standard(
        "stage8.opening_claim_values",
        "stage8.opening_claim_values",
        MutationStrategy::OffsetScalar,
        TamperCoverage::Active,
        "real core fixture offsets final opening claim fields after compat conversion",
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
        "real core ZK fixture mutates the imported BlindFold proof payload after compat conversion",
    ),
    zk_target(
        "zk.vector_commitment_setup",
        "preprocessing.vc_setup",
        MutationStrategy::ReplaceSetup,
        TamperCoverage::Active,
        "validate_inputs rejects missing VC setup in ZK mode",
    ),
];

pub fn all_targets() -> Vec<TamperTarget> {
    PREAMBLE_TARGETS
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
        .chain(FUTURE_STAGE_TARGETS)
        .copied()
        .collect()
}

pub fn target(name: &str) -> Option<TamperTarget> {
    all_targets().into_iter().find(|target| target.name == name)
}

#[expect(
    clippy::panic,
    reason = "tamper tests should fail loudly when they reference a missing manifest entry"
)]
pub fn required_target(name: &str) -> TamperTarget {
    match target(name) {
        Some(target) => target,
        None => panic!("missing tamper manifest target {name}"),
    }
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
    let claims = zero_clear_claims();
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

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
pub fn assert_core_tamper_rejects(
    target: TamperTarget,
    base: &crate::support::core_fixtures::CoreVerifierCase,
    mutate: impl FnOnce(&mut crate::support::core_fixtures::CoreVerifierCase),
) {
    assert_manifest_target_is_active(target);
    let mut case = base.clone();
    mutate(&mut case);
    crate::support::assert_rejects(case.verify());
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
pub fn assert_precompat_core_tamper_rejects(
    target: TamperTarget,
    base: &crate::support::core_fixtures::CorePrecompatVerifierCase,
    mutate: impl FnOnce(&mut crate::support::core_fixtures::CorePrecompatVerifierCase),
) {
    assert_manifest_target_is_active(target);
    let mut case = base.clone();
    mutate(&mut case);

    let core_result = catch_unwind(AssertUnwindSafe(|| case.verify_core()));
    let core_rejected = match core_result {
        Ok(result) => result.is_err(),
        Err(_) => true,
    };
    assert!(
        core_rejected,
        "core verifier accepted pre-compat tampered target {target:?}"
    );
    crate::support::assert_rejects(case.verify_after_compat());
}

fn expand_manifest_path(target: TamperTarget) -> Vec<&'static str> {
    match target.path {
        "claims.stage1.outer.*" => vec![
            "claims.stage1.outer.left_instruction_input",
            "claims.stage1.outer.right_instruction_input",
            "claims.stage1.outer.product",
            "claims.stage1.outer.should_branch",
            "claims.stage1.outer.pc",
            "claims.stage1.outer.unexpanded_pc",
            "claims.stage1.outer.imm",
            "claims.stage1.outer.ram_address",
            "claims.stage1.outer.rs1_value",
            "claims.stage1.outer.rs2_value",
            "claims.stage1.outer.rd_write_value",
            "claims.stage1.outer.ram_read_value",
            "claims.stage1.outer.ram_write_value",
            "claims.stage1.outer.left_lookup_operand",
            "claims.stage1.outer.right_lookup_operand",
            "claims.stage1.outer.next_unexpanded_pc",
            "claims.stage1.outer.next_pc",
            "claims.stage1.outer.next_is_virtual",
            "claims.stage1.outer.next_is_first_in_sequence",
            "claims.stage1.outer.lookup_output",
            "claims.stage1.outer.should_jump",
            "claims.stage1.outer.flags.add_operands",
            "claims.stage1.outer.flags.subtract_operands",
            "claims.stage1.outer.flags.multiply_operands",
            "claims.stage1.outer.flags.load",
            "claims.stage1.outer.flags.store",
            "claims.stage1.outer.flags.jump",
            "claims.stage1.outer.flags.write_lookup_output_to_rd",
            "claims.stage1.outer.flags.virtual_instruction",
            "claims.stage1.outer.flags.assert",
            "claims.stage1.outer.flags.do_not_update_unexpanded_pc",
            "claims.stage1.outer.flags.advice",
            "claims.stage1.outer.flags.is_compressed",
            "claims.stage1.outer.flags.is_first_in_sequence",
            "claims.stage1.outer.flags.is_last_in_sequence",
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

fn zero_clear_claims() -> ClearProofClaims<Fr> {
    let zero = Fr::from_u64(0);

    ClearProofClaims {
        stage1: stage1::inputs::Stage1Claims {
            uniskip_output_claim: zero,
            outer: stage1::inputs::SpartanOuterClaims {
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
                flags: stage1::inputs::SpartanOuterFlagClaims {
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
        stage2: stage2::inputs::Stage2Claims {
            product_uniskip_output_claim: zero,
            batch_outputs: stage2::inputs::Stage2BatchOutputOpeningClaims {
                ram_read_write: stage2::inputs::RamReadWriteOutputOpeningClaims {
                    val: zero,
                    ra: zero,
                    inc: zero,
                },
                product_remainder: stage2::inputs::ProductRemainderOutputOpeningClaims {
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
                    stage2::inputs::InstructionClaimReductionOutputOpeningClaims {
                        lookup_output: Some(zero),
                        left_lookup_operand: zero,
                        right_lookup_operand: zero,
                        left_instruction_input: Some(zero),
                        right_instruction_input: Some(zero),
                    },
                ram_raf_evaluation: zero,
                ram_output_check: zero,
            },
        },
        stage3: stage3::inputs::Stage3Claims {
            shift: stage3::inputs::SpartanShiftOutputOpeningClaims {
                unexpanded_pc: zero,
                pc: zero,
                is_virtual: zero,
                is_first_in_sequence: zero,
                is_noop: zero,
            },
            instruction_input: stage3::inputs::InstructionInputOutputOpeningClaims {
                left_operand_is_rs1: zero,
                rs1_value: zero,
                left_operand_is_pc: zero,
                unexpanded_pc: zero,
                right_operand_is_rs2: zero,
                rs2_value: zero,
                right_operand_is_imm: zero,
                imm: zero,
            },
            registers_claim_reduction: stage3::inputs::RegistersClaimReductionOutputOpeningClaims {
                rd_write_value: zero,
                rs1_value: zero,
                rs2_value: zero,
            },
        },
        stage4: stage4::inputs::Stage4Claims {
            advice: stage4::inputs::RamValCheckAdviceOpeningClaims {
                untrusted: Some(zero),
                trusted: Some(zero),
            },
            registers_read_write: stage4::inputs::RegistersReadWriteOutputOpeningClaims {
                registers_val: zero,
                rs1_ra: zero,
                rs2_ra: zero,
                rd_wa: zero,
                rd_inc: zero,
            },
            ram_val_check: stage4::inputs::RamValCheckOutputOpeningClaims {
                ram_ra: zero,
                ram_inc: zero,
            },
        },
        stage5: stage5::inputs::Stage5Claims {
            instruction_read_raf: stage5::inputs::InstructionReadRafOutputOpeningClaims {
                lookup_table_flags: vec![zero],
                instruction_ra: vec![zero],
                instruction_raf_flag: zero,
            },
            ram_ra_claim_reduction: stage5::inputs::RamRaClaimReductionOutputOpeningClaims {
                ram_ra: zero,
            },
            registers_val_evaluation: stage5::inputs::RegistersValEvaluationOutputOpeningClaims {
                rd_inc: zero,
                rd_wa: zero,
            },
        },
        stage6: stage6::inputs::Stage6Claims {
            address_phase: stage6::inputs::Stage6AddressPhaseClaims {
                bytecode_read_raf: zero,
                booleanity: zero,
            },
            bytecode_read_raf: stage6::inputs::BytecodeReadRafOutputOpeningClaims {
                bytecode_ra: vec![zero],
            },
            booleanity: stage6::inputs::BooleanityOutputOpeningClaims {
                instruction_ra: vec![zero],
                bytecode_ra: vec![zero],
                ram_ra: vec![zero],
            },
            ram_hamming_booleanity: stage6::inputs::RamHammingBooleanityOutputOpeningClaims {
                ram_hamming_weight: zero,
            },
            ram_ra_virtualization: stage6::inputs::RamRaVirtualizationOutputOpeningClaims {
                ram_ra: vec![zero],
            },
            instruction_ra_virtualization:
                stage6::inputs::InstructionRaVirtualizationOutputOpeningClaims {
                    committed_instruction_ra: vec![zero],
                },
            inc_claim_reduction: stage6::inputs::IncClaimReductionOutputOpeningClaims {
                ram_inc: zero,
                rd_inc: zero,
            },
            advice_cycle_phase: stage6::inputs::Stage6AdviceCyclePhaseClaims {
                trusted: Some(stage6::inputs::AdviceCyclePhaseOutputClaim {
                    opening_claim: zero,
                }),
                untrusted: Some(stage6::inputs::AdviceCyclePhaseOutputClaim {
                    opening_claim: zero,
                }),
            },
        },
        stage7: stage7::inputs::Stage7Claims {
            hamming_weight_claim_reduction:
                stage7::inputs::HammingWeightClaimReductionOutputOpeningClaims {
                    instruction_ra: vec![zero],
                    bytecode_ra: vec![zero],
                    ram_ra: vec![zero],
                },
            advice_address_phase: stage7::inputs::Stage7AdviceAddressPhaseClaims {
                trusted: Some(stage7::inputs::AdviceAddressPhaseOutputClaim {
                    opening_claim: zero,
                }),
                untrusted: Some(stage7::inputs::AdviceAddressPhaseOutputClaim {
                    opening_claim: zero,
                }),
            },
        },
    }
}
