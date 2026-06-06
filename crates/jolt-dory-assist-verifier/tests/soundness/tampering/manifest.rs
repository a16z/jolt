use crate::{
    soundness::tampering,
    support::{assert_unique_tamper_target_names, TamperTarget},
};

pub const TARGETS: &[TamperTarget] = &[
    TamperTarget {
        name: tampering::CLEAR_INPUT_EVAL.name,
        fixture: tampering::CLEAR_INPUT_EVAL.fixture,
        checked_at: tampering::CLEAR_INPUT_EVAL.checked_at,
        coverage_note: "active: checked by the continued transcript digest bound into Stage 1 public claims",
    },
    TamperTarget {
        name: tampering::CLEAR_INPUT_POINT.name,
        fixture: tampering::CLEAR_INPUT_POINT.fixture,
        checked_at: tampering::CLEAR_INPUT_POINT.checked_at,
        coverage_note: "active: checked by the continued transcript digest bound into Stage 1 public claims",
    },
    TamperTarget {
        name: tampering::ZK_INPUT_POINT.name,
        fixture: tampering::ZK_INPUT_POINT.fixture,
        checked_at: tampering::ZK_INPUT_POINT.checked_at,
        coverage_note: "active: checked by the continued transcript digest bound into Stage 1 public claims",
    },
    TamperTarget {
        name: tampering::CHECKED_INPUT_DIGEST.name,
        fixture: tampering::CHECKED_INPUT_DIGEST.fixture,
        checked_at: tampering::CHECKED_INPUT_DIGEST.checked_at,
        coverage_note: "active: checked by the continued transcript digest bound into Stage 1 public claims",
    },
    TamperTarget {
        name: tampering::VERIFIER_SETUP_DIGEST.name,
        fixture: tampering::VERIFIER_SETUP_DIGEST.fixture,
        checked_at: tampering::VERIFIER_SETUP_DIGEST.checked_at,
        coverage_note: "active: checked by the verifier setup digest public claim",
    },
    TamperTarget {
        name: tampering::VERIFIER_SETUP_ARTIFACT.name,
        fixture: tampering::VERIFIER_SETUP_ARTIFACT.fixture,
        checked_at: tampering::VERIFIER_SETUP_ARTIFACT.checked_at,
        coverage_note: "active: checked by concrete verifier setup artifact public claims",
    },
    TamperTarget {
        name: tampering::DORY_PROOF_ARTIFACT.name,
        fixture: tampering::DORY_PROOF_ARTIFACT.fixture,
        checked_at: tampering::DORY_PROOF_ARTIFACT.checked_at,
        coverage_note: "active: checked by the Dory proof artifact digest public claim",
    },
    TamperTarget {
        name: tampering::DORY_VMV_C_ARTIFACT.name,
        fixture: tampering::DORY_VMV_C_ARTIFACT.fixture,
        checked_at: tampering::DORY_VMV_C_ARTIFACT.checked_at,
        coverage_note: "active: checked by the concrete Dory VMV C GT artifact public claim",
    },
    TamperTarget {
        name: tampering::DORY_VMV_E1_ARTIFACT.name,
        fixture: tampering::DORY_VMV_E1_ARTIFACT.fixture,
        checked_at: tampering::DORY_VMV_E1_ARTIFACT.checked_at,
        coverage_note: "active: checked by the concrete Dory VMV E1 G1 artifact public claim",
    },
    TamperTarget {
        name: tampering::DORY_ZK_ARTIFACT.name,
        fixture: tampering::DORY_ZK_ARTIFACT.fixture,
        checked_at: tampering::DORY_ZK_ARTIFACT.checked_at,
        coverage_note: "active: checked by the concrete Dory ZK artifact public claim",
    },
    TamperTarget {
        name: tampering::ZK_MULTIROUND_DORY_E2_ARTIFACT.name,
        fixture: tampering::ZK_MULTIROUND_DORY_E2_ARTIFACT.fixture,
        checked_at: tampering::ZK_MULTIROUND_DORY_E2_ARTIFACT.checked_at,
        coverage_note: "active: checked by the concrete ZK Dory e2 artifact public claim on the multiround verifier path",
    },
    TamperTarget {
        name: tampering::ZK_MULTIROUND_DORY_Y_COM_ARTIFACT.name,
        fixture: tampering::ZK_MULTIROUND_DORY_Y_COM_ARTIFACT.fixture,
        checked_at: tampering::ZK_MULTIROUND_DORY_Y_COM_ARTIFACT.checked_at,
        coverage_note: "active: checked by the concrete ZK Dory y_com artifact public claim on the multiround verifier path",
    },
    TamperTarget {
        name: tampering::ZK_MULTIROUND_DORY_SCALAR_PRODUCT_ARTIFACT.name,
        fixture: tampering::ZK_MULTIROUND_DORY_SCALAR_PRODUCT_ARTIFACT.fixture,
        checked_at: tampering::ZK_MULTIROUND_DORY_SCALAR_PRODUCT_ARTIFACT.checked_at,
        coverage_note: "active: checked by the concrete ZK scalar-product artifact public claim on the multiround verifier path",
    },
    TamperTarget {
        name: tampering::DORY_REDUCE_ROUND_ARTIFACT.name,
        fixture: tampering::DORY_REDUCE_ROUND_ARTIFACT.fixture,
        checked_at: tampering::DORY_REDUCE_ROUND_ARTIFACT.checked_at,
        coverage_note: "active: checked by concrete Dory reduce-round artifact public claims",
    },
    TamperTarget {
        name: tampering::ZK_MULTIROUND_DORY_REDUCE_ROUND_ARTIFACT.name,
        fixture: tampering::ZK_MULTIROUND_DORY_REDUCE_ROUND_ARTIFACT.fixture,
        checked_at: tampering::ZK_MULTIROUND_DORY_REDUCE_ROUND_ARTIFACT.checked_at,
        coverage_note: "active: checked by concrete Dory reduce-round artifact public claims on the ZK multiround verifier path",
    },
    TamperTarget {
        name: tampering::DORY_REDUCE_DIMENSIONS.name,
        fixture: tampering::DORY_REDUCE_DIMENSIONS.fixture,
        checked_at: tampering::DORY_REDUCE_DIMENSIONS.checked_at,
        coverage_note: "active: checked by binding proof Dory-reduce dimensions to the PCS proof point length and reduce-round count",
    },
    TamperTarget {
        name: tampering::ZK_MULTIROUND_DORY_REDUCE_DIMENSIONS.name,
        fixture: tampering::ZK_MULTIROUND_DORY_REDUCE_DIMENSIONS.fixture,
        checked_at: tampering::ZK_MULTIROUND_DORY_REDUCE_DIMENSIONS.checked_at,
        coverage_note: "active: checked by binding ZK multiround proof Dory-reduce dimensions to the PCS proof point length and reduce-round count",
    },
    TamperTarget {
        name: tampering::GT_DIMENSIONS.name,
        fixture: tampering::GT_DIMENSIONS.fixture,
        checked_at: tampering::GT_DIMENSIONS.checked_at,
        coverage_note: "active: checked by rejecting non-canonical active protocol dimensions before deriving stage catalogs",
    },
    TamperTarget {
        name: tampering::PACKING_DIMENSIONS.name,
        fixture: tampering::PACKING_DIMENSIONS.fixture,
        checked_at: tampering::PACKING_DIMENSIONS.checked_at,
        coverage_note: "active: checked by requiring prefix-packing dimensions to equal catalog-derived minimal dimensions",
    },
    TamperTarget {
        name: tampering::DORY_FINAL_ARTIFACT.name,
        fixture: tampering::DORY_FINAL_ARTIFACT.fixture,
        checked_at: tampering::DORY_FINAL_ARTIFACT.checked_at,
        coverage_note: "active: checked by concrete Dory final scalar-product artifact public claims",
    },
    TamperTarget {
        name: tampering::JOLT_COMMITMENT_CLAIM.name,
        fixture: tampering::JOLT_COMMITMENT_CLAIM.fixture,
        checked_at: tampering::JOLT_COMMITMENT_CLAIM.checked_at,
        coverage_note: "active: checked by the Jolt commitment digest public claim",
    },
    TamperTarget {
        name: tampering::JOLT_COMMITMENT_GT_CLAIM.name,
        fixture: tampering::JOLT_COMMITMENT_GT_CLAIM.fixture,
        checked_at: tampering::JOLT_COMMITMENT_GT_CLAIM.checked_at,
        coverage_note: "active: checked by the concrete joint commitment GT artifact public claim",
    },
    TamperTarget {
        name: tampering::JOLT_EVALUATION_CLAIM.name,
        fixture: tampering::JOLT_EVALUATION_CLAIM.fixture,
        checked_at: tampering::JOLT_EVALUATION_CLAIM.checked_at,
        coverage_note: "active: checked by the clear Jolt evaluation public claim",
    },
    TamperTarget {
        name: tampering::TRANSCRIPT_SCALAR_CLAIM.name,
        fixture: tampering::TRANSCRIPT_SCALAR_CLAIM.fixture,
        checked_at: tampering::TRANSCRIPT_SCALAR_CLAIM.checked_at,
        coverage_note: "active: checked by the Fr-to-Fq transcript scalar public claim",
    },
    TamperTarget {
        name: tampering::ZK_MULTIROUND_SIGMA_C_TRANSCRIPT_SCALAR_CLAIM.name,
        fixture: tampering::ZK_MULTIROUND_SIGMA_C_TRANSCRIPT_SCALAR_CLAIM.fixture,
        checked_at: tampering::ZK_MULTIROUND_SIGMA_C_TRANSCRIPT_SCALAR_CLAIM.checked_at,
        coverage_note: "active: checked by the ZK scalar-product sigma_c transcript scalar public claim on the multiround verifier path",
    },
    TamperTarget {
        name: tampering::STAGE1_PAYLOAD.name,
        fixture: tampering::STAGE1_PAYLOAD.fixture,
        checked_at: tampering::STAGE1_PAYLOAD.checked_at,
        coverage_note: "active: checked by stage 1 canonical relation catalog binding",
    },
    TamperTarget {
        name: tampering::STAGE1_SUMCHECK_ROUNDS.name,
        fixture: tampering::STAGE1_SUMCHECK_ROUNDS.fixture,
        checked_at: tampering::STAGE1_SUMCHECK_ROUNDS.checked_at,
        coverage_note: "active: checked by stage 1 compressed sumcheck transcript verification",
    },
    TamperTarget {
        name: tampering::STAGE1_RELATION_OUTPUT.name,
        fixture: tampering::STAGE1_RELATION_OUTPUT.fixture,
        checked_at: tampering::STAGE1_RELATION_OUTPUT.checked_at,
        coverage_note: "active: checked by evaluating the stage 1 jolt-claims output expression",
    },
    TamperTarget {
        name: tampering::STAGE1_DIGIT_SELECTOR_OUTPUT.name,
        fixture: tampering::STAGE1_DIGIT_SELECTOR_OUTPUT.fixture,
        checked_at: tampering::STAGE1_DIGIT_SELECTOR_OUTPUT.checked_at,
        coverage_note: "active: checked by the GT base-4 digit-selector relation",
    },
    TamperTarget {
        name: tampering::STAGE1_SHIFT_OUTPUT.name,
        fixture: tampering::STAGE1_SHIFT_OUTPUT.fixture,
        checked_at: tampering::STAGE1_SHIFT_OUTPUT.checked_at,
        coverage_note: "active: checked by the GT exponentiation shift relation",
    },
    TamperTarget {
        name: tampering::STAGE1_SHIFT_PUBLIC.name,
        fixture: tampering::STAGE1_SHIFT_PUBLIC.fixture,
        checked_at: tampering::STAGE1_SHIFT_PUBLIC.checked_at,
        coverage_note: "active: checked by the typed GT shift-kernel public claim on a nonzero shift accumulator",
    },
    TamperTarget {
        name: tampering::STAGE1_BOUNDARY_OUTPUT.name,
        fixture: tampering::STAGE1_BOUNDARY_OUTPUT.fixture,
        checked_at: tampering::STAGE1_BOUNDARY_OUTPUT.checked_at,
        coverage_note: "active: checked by the GT exponentiation boundary relation",
    },
    TamperTarget {
        name: tampering::STAGE1_BOUNDARY_PUBLIC.name,
        fixture: tampering::STAGE1_BOUNDARY_PUBLIC.fixture,
        checked_at: tampering::STAGE1_BOUNDARY_PUBLIC.checked_at,
        coverage_note: "active: checked by the typed GT boundary public claim",
    },
    TamperTarget {
        name: tampering::STAGE1_MULTIPLICATION_OUTPUT.name,
        fixture: tampering::STAGE1_MULTIPLICATION_OUTPUT.fixture,
        checked_at: tampering::STAGE1_MULTIPLICATION_OUTPUT.checked_at,
        coverage_note: "active: checked by the GT multiplication quotient relation",
    },
    TamperTarget {
        name: tampering::STAGE1_G1_SCALAR_MULTIPLICATION_OUTPUT.name,
        fixture: tampering::STAGE1_G1_SCALAR_MULTIPLICATION_OUTPUT.fixture,
        checked_at: tampering::STAGE1_G1_SCALAR_MULTIPLICATION_OUTPUT.checked_at,
        coverage_note: "active: checked by the G1 scalar-multiplication relation",
    },
    TamperTarget {
        name: tampering::STAGE1_G1_SHIFT_OUTPUT.name,
        fixture: tampering::STAGE1_G1_SHIFT_OUTPUT.fixture,
        checked_at: tampering::STAGE1_G1_SHIFT_OUTPUT.checked_at,
        coverage_note: "active: checked by the G1 scalar-multiplication shift relation",
    },
    TamperTarget {
        name: tampering::STAGE1_G1_BOUNDARY_PUBLIC.name,
        fixture: tampering::STAGE1_G1_BOUNDARY_PUBLIC.fixture,
        checked_at: tampering::STAGE1_G1_BOUNDARY_PUBLIC.checked_at,
        coverage_note: "active: checked by the G1 scalar-multiplication boundary public claim",
    },
    TamperTarget {
        name: tampering::STAGE1_G1_ADDITION_OUTPUT.name,
        fixture: tampering::STAGE1_G1_ADDITION_OUTPUT.fixture,
        checked_at: tampering::STAGE1_G1_ADDITION_OUTPUT.checked_at,
        coverage_note: "active: checked by the G1 addition relation",
    },
    TamperTarget {
        name: tampering::STAGE1_G2_SCALAR_MULTIPLICATION_OUTPUT.name,
        fixture: tampering::STAGE1_G2_SCALAR_MULTIPLICATION_OUTPUT.fixture,
        checked_at: tampering::STAGE1_G2_SCALAR_MULTIPLICATION_OUTPUT.checked_at,
        coverage_note: "active: checked by the G2 scalar-multiplication relation",
    },
    TamperTarget {
        name: tampering::STAGE1_G2_SHIFT_OUTPUT.name,
        fixture: tampering::STAGE1_G2_SHIFT_OUTPUT.fixture,
        checked_at: tampering::STAGE1_G2_SHIFT_OUTPUT.checked_at,
        coverage_note: "active: checked by the G2 scalar-multiplication shift relation",
    },
    TamperTarget {
        name: tampering::STAGE1_G2_BOUNDARY_PUBLIC.name,
        fixture: tampering::STAGE1_G2_BOUNDARY_PUBLIC.fixture,
        checked_at: tampering::STAGE1_G2_BOUNDARY_PUBLIC.checked_at,
        coverage_note: "active: checked by the G2 scalar-multiplication boundary public claim",
    },
    TamperTarget {
        name: tampering::STAGE1_G2_ADDITION_OUTPUT.name,
        fixture: tampering::STAGE1_G2_ADDITION_OUTPUT.fixture,
        checked_at: tampering::STAGE1_G2_ADDITION_OUTPUT.checked_at,
        coverage_note: "active: checked by the G2 addition relation",
    },
    TamperTarget {
        name: tampering::STAGE1_LINE_STEP_OUTPUT.name,
        fixture: tampering::STAGE1_LINE_STEP_OUTPUT.fixture,
        checked_at: tampering::STAGE1_LINE_STEP_OUTPUT.checked_at,
        coverage_note: "active: checked by the Miller-loop G2 line-step relation",
    },
    TamperTarget {
        name: tampering::STAGE1_LINE_EVALUATION_OUTPUT.name,
        fixture: tampering::STAGE1_LINE_EVALUATION_OUTPUT.fixture,
        checked_at: tampering::STAGE1_LINE_EVALUATION_OUTPUT.checked_at,
        coverage_note: "active: checked by the Miller-loop sparse line-evaluation relation",
    },
    TamperTarget {
        name: tampering::STAGE1_PAIR_PRODUCT_OUTPUT.name,
        fixture: tampering::STAGE1_PAIR_PRODUCT_OUTPUT.fixture,
        checked_at: tampering::STAGE1_PAIR_PRODUCT_OUTPUT.checked_at,
        coverage_note: "active: checked by the Miller-loop pair-product relation",
    },
    TamperTarget {
        name: tampering::STAGE1_ACCUMULATOR_OUTPUT.name,
        fixture: tampering::STAGE1_ACCUMULATOR_OUTPUT.fixture,
        checked_at: tampering::STAGE1_ACCUMULATOR_OUTPUT.checked_at,
        coverage_note: "active: checked by the Miller-loop accumulator relation",
    },
    TamperTarget {
        name: tampering::STAGE1_MILLER_BOUNDARY_OUTPUT.name,
        fixture: tampering::STAGE1_MILLER_BOUNDARY_OUTPUT.fixture,
        checked_at: tampering::STAGE1_MILLER_BOUNDARY_OUTPUT.checked_at,
        coverage_note: "active: checked by the Miller-loop boundary relation",
    },
    TamperTarget {
        name: tampering::STAGE1_DORY_REDUCE_GT_TRANSITION_OUTPUT.name,
        fixture: tampering::STAGE1_DORY_REDUCE_GT_TRANSITION_OUTPUT.fixture,
        checked_at: tampering::STAGE1_DORY_REDUCE_GT_TRANSITION_OUTPUT.checked_at,
        coverage_note: "active: checked by the Dory reduce GT transition relation",
    },
    TamperTarget {
        name: tampering::STAGE1_DORY_REDUCE_G1_TRANSITION_OUTPUT.name,
        fixture: tampering::STAGE1_DORY_REDUCE_G1_TRANSITION_OUTPUT.fixture,
        checked_at: tampering::STAGE1_DORY_REDUCE_G1_TRANSITION_OUTPUT.checked_at,
        coverage_note: "active: checked by the Dory reduce G1 transition relation",
    },
    TamperTarget {
        name: tampering::STAGE1_DORY_REDUCE_G2_TRANSITION_OUTPUT.name,
        fixture: tampering::STAGE1_DORY_REDUCE_G2_TRANSITION_OUTPUT.fixture,
        checked_at: tampering::STAGE1_DORY_REDUCE_G2_TRANSITION_OUTPUT.checked_at,
        coverage_note: "active: checked by the Dory reduce G2 transition relation",
    },
    TamperTarget {
        name: tampering::STAGE1_DORY_REDUCE_SCALAR_FOLD_OUTPUT.name,
        fixture: tampering::STAGE1_DORY_REDUCE_SCALAR_FOLD_OUTPUT.fixture,
        checked_at: tampering::STAGE1_DORY_REDUCE_SCALAR_FOLD_OUTPUT.checked_at,
        coverage_note: "active: checked by the Dory reduce scalar accumulator-fold relation",
    },
    TamperTarget {
        name: tampering::STAGE1_DORY_REDUCE_STATE_CHAIN_OUTPUT.name,
        fixture: tampering::STAGE1_DORY_REDUCE_STATE_CHAIN_OUTPUT.fixture,
        checked_at: tampering::STAGE1_DORY_REDUCE_STATE_CHAIN_OUTPUT.checked_at,
        coverage_note: "active: checked by the multi-round Dory reduce next-row/current-row state-chain relation",
    },
    TamperTarget {
        name: tampering::STAGE1_DORY_REDUCE_BOUNDARY_OUTPUT.name,
        fixture: tampering::STAGE1_DORY_REDUCE_BOUNDARY_OUTPUT.fixture,
        checked_at: tampering::STAGE1_DORY_REDUCE_BOUNDARY_OUTPUT.checked_at,
        coverage_note: "active: checked by the multi-round Dory reduce initial/final boundary relation",
    },
    TamperTarget {
        name: tampering::ZK_STAGE1_DORY_REDUCE_STATE_CHAIN_OUTPUT.name,
        fixture: tampering::ZK_STAGE1_DORY_REDUCE_STATE_CHAIN_OUTPUT.fixture,
        checked_at: tampering::ZK_STAGE1_DORY_REDUCE_STATE_CHAIN_OUTPUT.checked_at,
        coverage_note: "active: checks the ZK multi-round Dory reduce next-row/current-row state-chain relation",
    },
    TamperTarget {
        name: tampering::ZK_STAGE1_DORY_REDUCE_BOUNDARY_OUTPUT.name,
        fixture: tampering::ZK_STAGE1_DORY_REDUCE_BOUNDARY_OUTPUT.fixture,
        checked_at: tampering::ZK_STAGE1_DORY_REDUCE_BOUNDARY_OUTPUT.checked_at,
        coverage_note: "active: checks the ZK multi-round Dory reduce initial/final boundary relation",
    },
    TamperTarget {
        name: tampering::STAGE2_PAYLOAD.name,
        fixture: tampering::STAGE2_PAYLOAD.fixture,
        checked_at: tampering::STAGE2_PAYLOAD.checked_at,
        coverage_note: "active: checked by stage 2 canonical direct-copy catalog binding",
    },
    TamperTarget {
        name: tampering::STAGE2_COPY_VALUE.name,
        fixture: tampering::STAGE2_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_COPY_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 direct equality over resolved copy endpoints",
    },
    TamperTarget {
        name: tampering::STAGE2_PUBLIC_VMV_C_COPY_VALUE.name,
        fixture: tampering::STAGE2_PUBLIC_VMV_C_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_PUBLIC_VMV_C_COPY_VALUE.checked_at,
        coverage_note: "active: checks Dory VMV C public artifact is the GT exponentiation accumulator consumed by Stage 2",
    },
    TamperTarget {
        name: tampering::STAGE2_PUBLIC_VMV_E1_COPY_VALUE.name,
        fixture: tampering::STAGE2_PUBLIC_VMV_E1_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_PUBLIC_VMV_E1_COPY_VALUE.checked_at,
        coverage_note: "active: checks Dory VMV E1 public artifact is the Miller-loop G1 evaluation point consumed by Stage 2",
    },
    TamperTarget {
        name: tampering::STAGE2_LINE_COPY_VALUE.name,
        fixture: tampering::STAGE2_LINE_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_LINE_COPY_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 direct equality over Miller-loop line copy endpoints",
    },
    TamperTarget {
        name: tampering::STAGE2_PAIR_PRODUCT_COPY_VALUE.name,
        fixture: tampering::STAGE2_PAIR_PRODUCT_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_PAIR_PRODUCT_COPY_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 direct equality over Miller-loop pair-product copy endpoints",
    },
    TamperTarget {
        name: tampering::STAGE2_PAIR_PRODUCT_QUOTIENT_COPY_VALUE.name,
        fixture: tampering::STAGE2_PAIR_PRODUCT_QUOTIENT_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_PAIR_PRODUCT_QUOTIENT_COPY_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 direct equality over Miller-loop pair-product quotient copy endpoints",
    },
    TamperTarget {
        name: tampering::STAGE2_ACCUMULATOR_COPY_VALUE.name,
        fixture: tampering::STAGE2_ACCUMULATOR_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_ACCUMULATOR_COPY_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 direct equality over Miller-loop accumulator copy endpoints",
    },
    TamperTarget {
        name: tampering::STAGE2_ACCUMULATOR_QUOTIENT_COPY_VALUE.name,
        fixture: tampering::STAGE2_ACCUMULATOR_QUOTIENT_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_ACCUMULATOR_QUOTIENT_COPY_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 direct equality over Miller-loop accumulator quotient copy endpoints",
    },
    TamperTarget {
        name: tampering::STAGE2_BOUNDARY_COPY_VALUE.name,
        fixture: tampering::STAGE2_BOUNDARY_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_BOUNDARY_COPY_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 direct equality over Miller-loop boundary copy endpoints",
    },
    TamperTarget {
        name: tampering::STAGE2_G1_SHIFT_COPY_VALUE.name,
        fixture: tampering::STAGE2_G1_SHIFT_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_G1_SHIFT_COPY_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 direct equality over G1 scalar-mul shift endpoints",
    },
    TamperTarget {
        name: tampering::STAGE2_G1_BOUNDARY_COPY_VALUE.name,
        fixture: tampering::STAGE2_G1_BOUNDARY_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_G1_BOUNDARY_COPY_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 direct equality over G1 scalar-mul boundary endpoints",
    },
    TamperTarget {
        name: tampering::STAGE2_G2_SHIFT_COPY_VALUE.name,
        fixture: tampering::STAGE2_G2_SHIFT_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_G2_SHIFT_COPY_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 direct equality over G2 scalar-mul shift endpoints",
    },
    TamperTarget {
        name: tampering::STAGE2_G2_BOUNDARY_COPY_VALUE.name,
        fixture: tampering::STAGE2_G2_BOUNDARY_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_G2_BOUNDARY_COPY_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 direct equality over G2 scalar-mul boundary endpoints",
    },
    TamperTarget {
        name: tampering::STAGE2_DORY_REDUCE_SCALAR_FOLD_COPY_VALUE.name,
        fixture: tampering::STAGE2_DORY_REDUCE_SCALAR_FOLD_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_DORY_REDUCE_SCALAR_FOLD_COPY_VALUE.checked_at,
        coverage_note: "active: checks Dory reduce scalar fold factors are copied from the verifier transcript scalars",
    },
    TamperTarget {
        name: tampering::STAGE2_DORY_REDUCE_INITIAL_STATE_COPY_VALUE.name,
        fixture: tampering::STAGE2_DORY_REDUCE_INITIAL_STATE_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_DORY_REDUCE_INITIAL_STATE_COPY_VALUE.checked_at,
        coverage_note: "active: checks the Dory reduce initial state is copied from the public Dory proof, commitment, and scalar identity values before transition checks",
    },
    TamperTarget {
        name: tampering::STAGE2_DORY_REDUCE_PROOF_ARTIFACT_COPY_VALUE.name,
        fixture: tampering::STAGE2_DORY_REDUCE_PROOF_ARTIFACT_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_DORY_REDUCE_PROOF_ARTIFACT_COPY_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 direct equality from Dory reduce proof-message artifacts into transition openings",
    },
    TamperTarget {
        name: tampering::STAGE2_DORY_REDUCE_SETUP_ARTIFACT_COPY_VALUE.name,
        fixture: tampering::STAGE2_DORY_REDUCE_SETUP_ARTIFACT_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_DORY_REDUCE_SETUP_ARTIFACT_COPY_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 direct equality from verifier setup chi/delta artifacts into transition openings",
    },
    TamperTarget {
        name: tampering::STAGE2_DORY_REDUCE_TRANSCRIPT_SCALAR_COPY_VALUE.name,
        fixture: tampering::STAGE2_DORY_REDUCE_TRANSCRIPT_SCALAR_COPY_VALUE.fixture,
        checked_at: tampering::STAGE2_DORY_REDUCE_TRANSCRIPT_SCALAR_COPY_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 direct equality from Fr-derived Dory reduce transcript scalars into transition openings",
    },
    TamperTarget {
        name: tampering::STAGE2_DORY_REDUCE_PUBLIC_FOLD_VALUE.name,
        fixture: tampering::STAGE2_DORY_REDUCE_PUBLIC_FOLD_VALUE.fixture,
        checked_at: tampering::STAGE2_DORY_REDUCE_PUBLIC_FOLD_VALUE.checked_at,
        coverage_note: "active: checked by stage 2 Dory-reduce public folds for multi-round public vectors",
    },
    TamperTarget {
        name: tampering::ZK_STAGE2_DORY_REDUCE_PUBLIC_FOLD_VALUE.name,
        fixture: tampering::ZK_STAGE2_DORY_REDUCE_PUBLIC_FOLD_VALUE.fixture,
        checked_at: tampering::ZK_STAGE2_DORY_REDUCE_PUBLIC_FOLD_VALUE.checked_at,
        coverage_note: "active: checks ZK stage 2 Dory-reduce public folds for multi-round public vectors with ZK transcript staging",
    },
    TamperTarget {
        name: tampering::STAGE3_PAYLOAD.name,
        fixture: tampering::STAGE3_PAYLOAD.fixture,
        checked_at: tampering::STAGE3_PAYLOAD.checked_at,
        coverage_note: "active: checked by stage 3 prefix-weighted packed-eval binding",
    },
    TamperTarget {
        name: tampering::STAGE3_REDUCED_OPENINGS.name,
        fixture: tampering::STAGE3_REDUCED_OPENINGS.fixture,
        checked_at: tampering::STAGE3_REDUCED_OPENINGS.checked_at,
        coverage_note: "active: checked by stage 3 canonical reduced-opening order binding",
    },
    TamperTarget {
        name: tampering::OPENING_CLAIM_POINT.name,
        fixture: tampering::OPENING_CLAIM_POINT.fixture,
        checked_at: tampering::OPENING_CLAIM_POINT.checked_at,
        coverage_note: "active: checked by packed Hyrax opening verification",
    },
    TamperTarget {
        name: tampering::OPENING_CLAIM_EVAL.name,
        fixture: tampering::OPENING_CLAIM_EVAL.fixture,
        checked_at: tampering::OPENING_CLAIM_EVAL.checked_at,
        coverage_note: "active: checked by packed-eval binding",
    },
    TamperTarget {
        name: tampering::HYRAX_OPENING_ROW.name,
        fixture: tampering::HYRAX_OPENING_ROW.fixture,
        checked_at: tampering::HYRAX_OPENING_ROW.checked_at,
        coverage_note: "active: checked by Hyrax opening proof verification",
    },
    TamperTarget {
        name: tampering::HYRAX_OPENING_SCALAR.name,
        fixture: tampering::HYRAX_OPENING_SCALAR.fixture,
        checked_at: tampering::HYRAX_OPENING_SCALAR.checked_at,
        coverage_note: "active: checked by Hyrax opening proof verification",
    },
    TamperTarget {
        name: tampering::DENSE_COMMITMENT.name,
        fixture: tampering::DENSE_COMMITMENT.fixture,
        checked_at: tampering::DENSE_COMMITMENT.checked_at,
        coverage_note: "active: checked by packed witness commitment binding",
    },
    TamperTarget {
        name: tampering::PUBLIC_OUTPUT.name,
        fixture: tampering::PUBLIC_OUTPUT.fixture,
        checked_at: tampering::PUBLIC_OUTPUT.checked_at,
        coverage_note: "active: checked by native pre-final-exponentiation output binding",
    },
    TamperTarget {
        name: tampering::ZK_PUBLIC_OUTPUT.name,
        fixture: tampering::ZK_PUBLIC_OUTPUT.fixture,
        checked_at: tampering::ZK_PUBLIC_OUTPUT.checked_at,
        coverage_note: "active: checked by ZK native pre-final-exponentiation output binding",
    },
    TamperTarget {
        name: tampering::NATIVE_FINAL_INPUT.name,
        fixture: tampering::NATIVE_FINAL_INPUT.fixture,
        checked_at: tampering::NATIVE_FINAL_INPUT.checked_at,
        coverage_note: "active: checked by native-final reducer-state public input binding",
    },
    TamperTarget {
        name: tampering::ZK_NATIVE_FINAL_INPUT.name,
        fixture: tampering::ZK_NATIVE_FINAL_INPUT.fixture,
        checked_at: tampering::ZK_NATIVE_FINAL_INPUT.checked_at,
        coverage_note: "active: checked by ZK native-final reducer-state public input binding",
    },
];

#[test]
fn tamper_manifest_target_names_are_unique() {
    assert_unique_tamper_target_names(TARGETS);
}

#[test]
fn tamper_manifest_covers_registered_cases() {
    let missing: Vec<_> = tampering::ALL
        .iter()
        .filter(|case| !TARGETS.iter().any(|target| target.name == case.name))
        .map(|case| case.name)
        .collect();

    assert!(
        missing.is_empty(),
        "tamper cases missing from manifest: {missing:?}",
    );
}

#[test]
fn tamper_targets_are_documented() {
    let undocumented: Vec<_> = TARGETS
        .iter()
        .filter(|target| target.coverage_note.is_empty())
        .map(|target| target.name)
        .collect();

    assert!(
        undocumented.is_empty(),
        "tamper targets need a coverage note: {undocumented:?}",
    );
}
