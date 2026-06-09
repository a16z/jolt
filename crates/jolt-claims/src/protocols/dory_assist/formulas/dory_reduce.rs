use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::{challenge, constant, opening, public};

use super::super::{
    DoryAssistBoundaryEndpoint, DoryAssistChallengeId, DoryAssistCopyConstraint, DoryAssistExpr,
    DoryAssistOpeningId, DoryAssistPublicId, DoryAssistRelationClaims, DoryAssistRelationId,
    DoryAssistValueRef, DoryAssistValueType, DoryAssistVirtualPolynomial, DoryReduceChallenge,
    DoryReducePolynomial,
};
use super::{
    artifacts,
    dimensions::{DoryAssistSumcheckSpec, DoryReduceDimensions},
    setup_artifacts, transcript_scalars,
};

const LOCAL_ROW: usize = 0;
const NEXT_ROW: usize = 1;
const JOLT_COMMITMENT_GT_START: usize = 1;

pub const DORY_REDUCE_NATIVE_FINAL_GT_C_START: usize = 0;
pub const DORY_REDUCE_NATIVE_FINAL_D1_START: usize =
    DORY_REDUCE_NATIVE_FINAL_GT_C_START + artifacts::GT_ARTIFACT_COEFFS;
pub const DORY_REDUCE_NATIVE_FINAL_D2_START: usize =
    DORY_REDUCE_NATIVE_FINAL_D1_START + artifacts::GT_ARTIFACT_COEFFS;
pub const DORY_REDUCE_NATIVE_FINAL_E1_START: usize =
    DORY_REDUCE_NATIVE_FINAL_D2_START + artifacts::GT_ARTIFACT_COEFFS;
pub const DORY_REDUCE_NATIVE_FINAL_E2_START: usize =
    DORY_REDUCE_NATIVE_FINAL_E1_START + artifacts::G1_ARTIFACT_COORDS;
pub const DORY_REDUCE_NATIVE_FINAL_E1_INIT_START: usize =
    DORY_REDUCE_NATIVE_FINAL_E2_START + artifacts::G2_ARTIFACT_COORDS;
pub const DORY_REDUCE_NATIVE_FINAL_D2_INIT_START: usize =
    DORY_REDUCE_NATIVE_FINAL_E1_INIT_START + artifacts::G1_ARTIFACT_COORDS;
pub const DORY_REDUCE_NATIVE_FINAL_S1_ACC_INDEX: usize =
    DORY_REDUCE_NATIVE_FINAL_D2_INIT_START + artifacts::GT_ARTIFACT_COEFFS;
pub const DORY_REDUCE_NATIVE_FINAL_S2_ACC_INDEX: usize = DORY_REDUCE_NATIVE_FINAL_S1_ACC_INDEX + 1;
pub const DORY_REDUCE_NATIVE_FINAL_INPUT_LEN: usize = DORY_REDUCE_NATIVE_FINAL_S2_ACC_INDEX + 1;

pub const fn verifier_setup_round(reduce_rounds: usize, round: usize) -> usize {
    reduce_rounds - round
}

pub const fn scalar_fold_sumcheck(dimensions: DoryReduceDimensions) -> DoryAssistSumcheckSpec {
    dimensions.scalar_fold_sumcheck()
}

pub const fn state_chain_sumcheck(dimensions: DoryReduceDimensions) -> DoryAssistSumcheckSpec {
    dimensions.state_chain_sumcheck()
}

pub const fn boundary_sumcheck(dimensions: DoryReduceDimensions) -> DoryAssistSumcheckSpec {
    dimensions.boundary_sumcheck()
}

pub const fn transition_sumcheck(dimensions: DoryReduceDimensions) -> DoryAssistSumcheckSpec {
    DoryAssistSumcheckSpec::boolean(dimensions.reduce_round_vars(), 2)
}

pub fn gt_transition<F>(dimensions: DoryReduceDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let updates = (0..artifacts::GT_ARTIFACT_COEFFS)
        .flat_map(gt_transition_updates)
        .collect::<Vec<_>>();
    transition_relation(
        DoryAssistRelationId::DoryReduceGtTransition,
        DoryReduceChallenge::GtTransitionBatch,
        dimensions,
        &updates,
    )
}

pub fn g1_transition<F>(dimensions: DoryReduceDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let updates = (0..artifacts::G1_ARTIFACT_COORDS)
        .map(g1_transition_update)
        .collect::<Vec<_>>();
    transition_relation(
        DoryAssistRelationId::DoryReduceG1Transition,
        DoryReduceChallenge::G1TransitionBatch,
        dimensions,
        &updates,
    )
}

pub fn g2_transition<F>(dimensions: DoryReduceDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let updates = (0..artifacts::G2_ARTIFACT_COORDS)
        .map(g2_transition_update)
        .collect::<Vec<_>>();
    transition_relation(
        DoryAssistRelationId::DoryReduceG2Transition,
        DoryReduceChallenge::G2TransitionBatch,
        dimensions,
        &updates,
    )
}

pub fn scalar_fold<F>(dimensions: DoryReduceDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let beta = dory_reduce_challenge(DoryReduceChallenge::ScalarFoldBatch);
    let input = opening(s1_next_accumulator_opening())
        + beta.clone() * opening(s2_next_accumulator_opening());
    let output = opening(s1_accumulator_opening()) * opening(s1_fold_factor_opening())
        + beta * opening(s2_accumulator_opening()) * opening(s2_fold_factor_opening());

    DoryAssistRelationClaims::new(
        DoryAssistRelationId::DoryReduceScalarFold,
        scalar_fold_sumcheck(dimensions),
        input,
        output,
    )
}

pub fn state_chain<F>(dimensions: DoryReduceDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let gamma = dory_reduce_challenge(DoryReduceChallenge::StateChainBatch);
    let pairs = state_chain_pairs();
    let input = combine_state_chain_openings(
        gamma.clone(),
        pairs
            .iter()
            .map(|(source, _)| dory_reduce_state_chain_opening(*source)),
    );
    let output = public(DoryAssistPublicId::DoryReduceShiftEqKernel)
        * combine_state_chain_openings(
            gamma,
            pairs
                .iter()
                .map(|(_, target)| dory_reduce_state_chain_opening(*target)),
        );

    DoryAssistRelationClaims::new(
        DoryAssistRelationId::DoryReduceStateChain,
        state_chain_sumcheck(dimensions),
        input,
        output,
    )
}

pub fn boundary<F>(dimensions: DoryReduceDimensions) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let relation = DoryAssistRelationId::DoryReduceBoundary;
    let gamma = dory_reduce_challenge(DoryReduceChallenge::BoundaryPoint);
    let initial_terms = initial_boundary_terms();
    let final_terms = final_boundary_terms();
    let initial = boundary_selector(relation, DoryAssistBoundaryEndpoint::Initial)
        * combine_boundary_differences(gamma.clone(), initial_terms.iter().copied());
    let final_value = boundary_selector(relation, DoryAssistBoundaryEndpoint::Final)
        * combine_boundary_differences(gamma.clone(), final_terms.iter().copied());

    DoryAssistRelationClaims::new(
        relation,
        boundary_sumcheck(dimensions),
        constant(F::zero()),
        initial + gamma.pow(initial_terms.len()) * final_value,
    )
}

pub fn gt_transition_openings() -> Vec<DoryAssistOpeningId> {
    let updates = (0..artifacts::GT_ARTIFACT_COEFFS).flat_map(gt_transition_updates);
    transition_openings(DoryAssistRelationId::DoryReduceGtTransition, updates)
}

pub fn g1_transition_openings() -> Vec<DoryAssistOpeningId> {
    let updates = (0..artifacts::G1_ARTIFACT_COORDS).map(g1_transition_update);
    transition_openings(DoryAssistRelationId::DoryReduceG1Transition, updates)
}

pub fn g2_transition_openings() -> Vec<DoryAssistOpeningId> {
    let updates = (0..artifacts::G2_ARTIFACT_COORDS).map(g2_transition_update);
    transition_openings(DoryAssistRelationId::DoryReduceG2Transition, updates)
}

pub fn scalar_fold_input_openings() -> [DoryAssistOpeningId; 2] {
    [s1_next_accumulator_opening(), s2_next_accumulator_opening()]
}

pub fn scalar_fold_output_openings() -> [DoryAssistOpeningId; 4] {
    [
        s1_accumulator_opening(),
        s1_fold_factor_opening(),
        s2_accumulator_opening(),
        s2_fold_factor_opening(),
    ]
}

pub fn state_chain_input_openings() -> Vec<DoryAssistOpeningId> {
    state_chain_pairs()
        .into_iter()
        .map(|(source, _)| dory_reduce_state_chain_opening(source))
        .collect()
}

pub fn state_chain_output_openings() -> Vec<DoryAssistOpeningId> {
    state_chain_pairs()
        .into_iter()
        .map(|(_, target)| dory_reduce_state_chain_opening(target))
        .collect()
}

pub fn boundary_output_openings() -> Vec<DoryAssistOpeningId> {
    initial_boundary_terms()
        .into_iter()
        .chain(final_boundary_terms())
        .map(|term| term.opening)
        .collect()
}

pub fn s1_accumulator_opening() -> DoryAssistOpeningId {
    dory_reduce_opening(DoryReducePolynomial::S1Accumulator)
}

pub fn s1_next_accumulator_opening() -> DoryAssistOpeningId {
    dory_reduce_opening(DoryReducePolynomial::S1NextAccumulator)
}

pub fn s1_fold_factor_opening() -> DoryAssistOpeningId {
    dory_reduce_opening(DoryReducePolynomial::S1FoldFactor)
}

pub fn s2_accumulator_opening() -> DoryAssistOpeningId {
    dory_reduce_opening(DoryReducePolynomial::S2Accumulator)
}

pub fn s2_next_accumulator_opening() -> DoryAssistOpeningId {
    dory_reduce_opening(DoryReducePolynomial::S2NextAccumulator)
}

pub fn s2_fold_factor_opening() -> DoryAssistOpeningId {
    dory_reduce_opening(DoryReducePolynomial::S2FoldFactor)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DoryReduceScalarTerm {
    One,
    Beta,
    BetaInverse,
    Alpha,
    AlphaInverse,
    AlphaBeta,
    AlphaInverseBetaInverse,
    S1FoldFactor,
    S2FoldFactor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryReduceTerm {
    pub scalar: DoryReduceScalarTerm,
    pub value: DoryReducePolynomial,
}

impl DoryReduceTerm {
    pub const fn new(scalar: DoryReduceScalarTerm, value: DoryReducePolynomial) -> Self {
        Self { scalar, value }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryReduceUpdate {
    pub target: DoryReducePolynomial,
    pub terms: Vec<DoryReduceTerm>,
}

impl DoryReduceUpdate {
    pub fn new<I>(target: DoryReducePolynomial, terms: I) -> Self
    where
        I: IntoIterator<Item = DoryReduceTerm>,
    {
        Self {
            target,
            terms: terms.into_iter().collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoryReducePublicFoldConstraint {
    pub value_type: DoryAssistValueType,
    pub sources: Vec<DoryAssistPublicId>,
    pub target: DoryAssistValueRef,
}

impl DoryReducePublicFoldConstraint {
    pub fn new(
        value_type: DoryAssistValueType,
        sources: Vec<DoryAssistPublicId>,
        target: DoryAssistValueRef,
    ) -> Self {
        Self {
            value_type,
            sources,
            target,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoryReduceBoundaryTerm {
    pub opening: DoryAssistOpeningId,
    pub value: DoryReduceBoundaryValue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoryReduceBoundaryValue {
    ConstantOne,
    Public(DoryAssistPublicId),
}

pub fn gt_transition_updates(component: usize) -> [DoryReduceUpdate; 3] {
    [
        DoryReduceUpdate::new(
            DoryReducePolynomial::NextC(component),
            [
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::One,
                    DoryReducePolynomial::CurrentC(component),
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::One,
                    DoryReducePolynomial::SetupChi(component),
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::Beta,
                    DoryReducePolynomial::CurrentD2(component),
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::BetaInverse,
                    DoryReducePolynomial::CurrentD1(component),
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::Alpha,
                    DoryReducePolynomial::MessageCPlus(component),
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::AlphaInverse,
                    DoryReducePolynomial::MessageCMinus(component),
                ),
            ],
        ),
        DoryReduceUpdate::new(
            DoryReducePolynomial::NextD1(component),
            [
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::Alpha,
                    DoryReducePolynomial::MessageD1Left(component),
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::One,
                    DoryReducePolynomial::MessageD1Right(component),
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::AlphaBeta,
                    DoryReducePolynomial::SetupDelta1L(component),
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::Beta,
                    DoryReducePolynomial::SetupDelta1R(component),
                ),
            ],
        ),
        DoryReduceUpdate::new(
            DoryReducePolynomial::NextD2(component),
            [
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::AlphaInverse,
                    DoryReducePolynomial::MessageD2Left(component),
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::One,
                    DoryReducePolynomial::MessageD2Right(component),
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::AlphaInverseBetaInverse,
                    DoryReducePolynomial::SetupDelta2L(component),
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::BetaInverse,
                    DoryReducePolynomial::SetupDelta2R(component),
                ),
            ],
        ),
    ]
}

pub fn g1_transition_update(component: usize) -> DoryReduceUpdate {
    DoryReduceUpdate::new(
        g1_next_polynomial(component),
        [
            DoryReduceTerm::new(DoryReduceScalarTerm::One, g1_current_polynomial(component)),
            DoryReduceTerm::new(DoryReduceScalarTerm::Beta, g1_beta_polynomial(component)),
            DoryReduceTerm::new(DoryReduceScalarTerm::Alpha, g1_plus_polynomial(component)),
            DoryReduceTerm::new(
                DoryReduceScalarTerm::AlphaInverse,
                g1_minus_polynomial(component),
            ),
        ],
    )
}

pub fn g2_transition_update(component: usize) -> DoryReduceUpdate {
    DoryReduceUpdate::new(
        g2_next_polynomial(component),
        [
            DoryReduceTerm::new(DoryReduceScalarTerm::One, g2_current_polynomial(component)),
            DoryReduceTerm::new(
                DoryReduceScalarTerm::BetaInverse,
                g2_beta_polynomial(component),
            ),
            DoryReduceTerm::new(DoryReduceScalarTerm::Alpha, g2_plus_polynomial(component)),
            DoryReduceTerm::new(
                DoryReduceScalarTerm::AlphaInverse,
                g2_minus_polynomial(component),
            ),
        ],
    )
}

pub fn scalar_fold_updates() -> [DoryReduceUpdate; 2] {
    [
        DoryReduceUpdate::new(
            DoryReducePolynomial::S1NextAccumulator,
            [DoryReduceTerm::new(
                DoryReduceScalarTerm::S1FoldFactor,
                DoryReducePolynomial::S1Accumulator,
            )],
        ),
        DoryReduceUpdate::new(
            DoryReducePolynomial::S2NextAccumulator,
            [DoryReduceTerm::new(
                DoryReduceScalarTerm::S2FoldFactor,
                DoryReducePolynomial::S2Accumulator,
            )],
        ),
    ]
}

pub fn state_chain_pairs() -> Vec<(DoryReducePolynomial, DoryReducePolynomial)> {
    let mut pairs = Vec::new();
    for component in 0..artifacts::GT_ARTIFACT_COEFFS {
        pairs.push((
            DoryReducePolynomial::NextC(component),
            DoryReducePolynomial::CurrentC(component),
        ));
        pairs.push((
            DoryReducePolynomial::NextD1(component),
            DoryReducePolynomial::CurrentD1(component),
        ));
        pairs.push((
            DoryReducePolynomial::NextD2(component),
            DoryReducePolynomial::CurrentD2(component),
        ));
    }
    pairs.extend([
        (
            DoryReducePolynomial::NextE1X,
            DoryReducePolynomial::CurrentE1X,
        ),
        (
            DoryReducePolynomial::NextE1Y,
            DoryReducePolynomial::CurrentE1Y,
        ),
        (
            DoryReducePolynomial::NextE1Infinity,
            DoryReducePolynomial::CurrentE1Infinity,
        ),
        (
            DoryReducePolynomial::NextE2X0,
            DoryReducePolynomial::CurrentE2X0,
        ),
        (
            DoryReducePolynomial::NextE2X1,
            DoryReducePolynomial::CurrentE2X1,
        ),
        (
            DoryReducePolynomial::NextE2Y0,
            DoryReducePolynomial::CurrentE2Y0,
        ),
        (
            DoryReducePolynomial::NextE2Y1,
            DoryReducePolynomial::CurrentE2Y1,
        ),
        (
            DoryReducePolynomial::NextE2Infinity,
            DoryReducePolynomial::CurrentE2Infinity,
        ),
        (
            DoryReducePolynomial::S1NextAccumulator,
            DoryReducePolynomial::S1Accumulator,
        ),
        (
            DoryReducePolynomial::S2NextAccumulator,
            DoryReducePolynomial::S2Accumulator,
        ),
    ]);
    pairs
}

pub fn initial_boundary_terms() -> Vec<DoryReduceBoundaryTerm> {
    let mut terms = Vec::new();
    extend_gt_boundary_terms(
        &mut terms,
        DoryReducePolynomial::CurrentC,
        DoryAssistPublicId::DoryProofArtifact,
        artifacts::DORY_VMV_C_START,
    );
    extend_gt_boundary_terms(
        &mut terms,
        DoryReducePolynomial::CurrentD1,
        DoryAssistPublicId::JoltCommitment,
        JOLT_COMMITMENT_GT_START,
    );
    extend_gt_boundary_terms(
        &mut terms,
        DoryReducePolynomial::CurrentD2,
        DoryAssistPublicId::DoryProofArtifact,
        artifacts::DORY_VMV_D2_START,
    );
    extend_g1_boundary_terms(
        &mut terms,
        [
            DoryReducePolynomial::CurrentE1X,
            DoryReducePolynomial::CurrentE1Y,
            DoryReducePolynomial::CurrentE1Infinity,
        ],
        DoryAssistPublicId::DoryProofArtifact,
        artifacts::DORY_VMV_E1_START,
    );
    extend_g2_boundary_terms(
        &mut terms,
        [
            DoryReducePolynomial::CurrentE2X0,
            DoryReducePolynomial::CurrentE2X1,
            DoryReducePolynomial::CurrentE2Y0,
            DoryReducePolynomial::CurrentE2Y1,
            DoryReducePolynomial::CurrentE2Infinity,
        ],
        DoryAssistPublicId::DoryReduceInitialE2,
        0,
    );
    terms.extend([
        DoryReduceBoundaryTerm {
            opening: dory_reduce_boundary_opening(DoryReducePolynomial::S1Accumulator),
            value: DoryReduceBoundaryValue::ConstantOne,
        },
        DoryReduceBoundaryTerm {
            opening: dory_reduce_boundary_opening(DoryReducePolynomial::S2Accumulator),
            value: DoryReduceBoundaryValue::ConstantOne,
        },
    ]);
    terms
}

pub fn final_boundary_terms() -> Vec<DoryReduceBoundaryTerm> {
    let mut terms = Vec::new();
    extend_gt_boundary_terms(
        &mut terms,
        DoryReducePolynomial::NextC,
        DoryAssistPublicId::NativeFinalCheckInput,
        DORY_REDUCE_NATIVE_FINAL_GT_C_START,
    );
    extend_gt_boundary_terms(
        &mut terms,
        DoryReducePolynomial::NextD1,
        DoryAssistPublicId::NativeFinalCheckInput,
        DORY_REDUCE_NATIVE_FINAL_D1_START,
    );
    extend_gt_boundary_terms(
        &mut terms,
        DoryReducePolynomial::NextD2,
        DoryAssistPublicId::NativeFinalCheckInput,
        DORY_REDUCE_NATIVE_FINAL_D2_START,
    );
    extend_g1_boundary_terms(
        &mut terms,
        [
            DoryReducePolynomial::NextE1X,
            DoryReducePolynomial::NextE1Y,
            DoryReducePolynomial::NextE1Infinity,
        ],
        DoryAssistPublicId::NativeFinalCheckInput,
        DORY_REDUCE_NATIVE_FINAL_E1_START,
    );
    extend_g2_boundary_terms(
        &mut terms,
        [
            DoryReducePolynomial::NextE2X0,
            DoryReducePolynomial::NextE2X1,
            DoryReducePolynomial::NextE2Y0,
            DoryReducePolynomial::NextE2Y1,
            DoryReducePolynomial::NextE2Infinity,
        ],
        DoryAssistPublicId::NativeFinalCheckInput,
        DORY_REDUCE_NATIVE_FINAL_E2_START,
    );
    terms.extend([
        DoryReduceBoundaryTerm {
            opening: dory_reduce_boundary_opening(DoryReducePolynomial::S1NextAccumulator),
            value: DoryReduceBoundaryValue::Public(DoryAssistPublicId::NativeFinalCheckInput(
                DORY_REDUCE_NATIVE_FINAL_S1_ACC_INDEX,
            )),
        },
        DoryReduceBoundaryTerm {
            opening: dory_reduce_boundary_opening(DoryReducePolynomial::S2NextAccumulator),
            value: DoryReduceBoundaryValue::Public(DoryAssistPublicId::NativeFinalCheckInput(
                DORY_REDUCE_NATIVE_FINAL_S2_ACC_INDEX,
            )),
        },
    ]);
    terms
}

pub fn proof_artifact_copy_constraints(round: usize) -> Vec<DoryAssistCopyConstraint> {
    let mut constraints = Vec::new();
    extend_gt_artifact_constraints(
        &mut constraints,
        artifacts::reduce_first_d1_left_start(round),
        DoryReducePolynomial::MessageD1Left,
    );
    extend_gt_artifact_constraints(
        &mut constraints,
        artifacts::reduce_first_d1_right_start(round),
        DoryReducePolynomial::MessageD1Right,
    );
    extend_gt_artifact_constraints(
        &mut constraints,
        artifacts::reduce_first_d2_left_start(round),
        DoryReducePolynomial::MessageD2Left,
    );
    extend_gt_artifact_constraints(
        &mut constraints,
        artifacts::reduce_first_d2_right_start(round),
        DoryReducePolynomial::MessageD2Right,
    );
    extend_g1_artifact_constraints(
        &mut constraints,
        artifacts::reduce_first_e1_beta_start(round),
        [
            DoryReducePolynomial::MessageE1BetaX,
            DoryReducePolynomial::MessageE1BetaY,
            DoryReducePolynomial::MessageE1BetaInfinity,
        ],
    );
    extend_g2_artifact_constraints(
        &mut constraints,
        artifacts::reduce_first_e2_beta_start(round),
        [
            DoryReducePolynomial::MessageE2BetaX0,
            DoryReducePolynomial::MessageE2BetaX1,
            DoryReducePolynomial::MessageE2BetaY0,
            DoryReducePolynomial::MessageE2BetaY1,
            DoryReducePolynomial::MessageE2BetaInfinity,
        ],
    );
    extend_gt_artifact_constraints(
        &mut constraints,
        artifacts::reduce_second_c_plus_start(round),
        DoryReducePolynomial::MessageCPlus,
    );
    extend_gt_artifact_constraints(
        &mut constraints,
        artifacts::reduce_second_c_minus_start(round),
        DoryReducePolynomial::MessageCMinus,
    );
    extend_g1_artifact_constraints(
        &mut constraints,
        artifacts::reduce_second_e1_plus_start(round),
        [
            DoryReducePolynomial::MessageE1PlusX,
            DoryReducePolynomial::MessageE1PlusY,
            DoryReducePolynomial::MessageE1PlusInfinity,
        ],
    );
    extend_g1_artifact_constraints(
        &mut constraints,
        artifacts::reduce_second_e1_minus_start(round),
        [
            DoryReducePolynomial::MessageE1MinusX,
            DoryReducePolynomial::MessageE1MinusY,
            DoryReducePolynomial::MessageE1MinusInfinity,
        ],
    );
    extend_g2_artifact_constraints(
        &mut constraints,
        artifacts::reduce_second_e2_plus_start(round),
        [
            DoryReducePolynomial::MessageE2PlusX0,
            DoryReducePolynomial::MessageE2PlusX1,
            DoryReducePolynomial::MessageE2PlusY0,
            DoryReducePolynomial::MessageE2PlusY1,
            DoryReducePolynomial::MessageE2PlusInfinity,
        ],
    );
    extend_g2_artifact_constraints(
        &mut constraints,
        artifacts::reduce_second_e2_minus_start(round),
        [
            DoryReducePolynomial::MessageE2MinusX0,
            DoryReducePolynomial::MessageE2MinusX1,
            DoryReducePolynomial::MessageE2MinusY0,
            DoryReducePolynomial::MessageE2MinusY1,
            DoryReducePolynomial::MessageE2MinusInfinity,
        ],
    );
    constraints
}

pub fn setup_artifact_copy_constraints(
    max_rounds: usize,
    setup_round: usize,
) -> Vec<DoryAssistCopyConstraint> {
    let mut constraints = Vec::new();
    extend_setup_gt_constraints(
        &mut constraints,
        setup_artifacts::dory_setup_chi_start(setup_round),
        DoryReducePolynomial::SetupChi,
    );
    extend_setup_gt_constraints(
        &mut constraints,
        setup_artifacts::dory_setup_delta_1l_start(max_rounds, setup_round),
        DoryReducePolynomial::SetupDelta1L,
    );
    extend_setup_gt_constraints(
        &mut constraints,
        setup_artifacts::dory_setup_delta_1r_start(max_rounds, setup_round),
        DoryReducePolynomial::SetupDelta1R,
    );
    extend_setup_gt_constraints(
        &mut constraints,
        setup_artifacts::dory_setup_delta_2l_start(max_rounds, setup_round),
        DoryReducePolynomial::SetupDelta2L,
    );
    extend_setup_gt_constraints(
        &mut constraints,
        setup_artifacts::dory_setup_delta_2r_start(max_rounds, setup_round),
        DoryReducePolynomial::SetupDelta2R,
    );
    constraints
}

pub fn round_setup_artifact_copy_constraints(
    reduce_rounds: usize,
    round: usize,
) -> Vec<DoryAssistCopyConstraint> {
    setup_artifact_copy_constraints(reduce_rounds, verifier_setup_round(reduce_rounds, round))
}

pub fn initial_state_copy_constraints() -> Vec<DoryAssistCopyConstraint> {
    let mut constraints = Vec::new();
    extend_public_gt_constraints(
        &mut constraints,
        DoryAssistPublicId::DoryProofArtifact,
        artifacts::DORY_VMV_C_START,
        DoryReducePolynomial::CurrentC,
    );
    extend_public_gt_constraints(
        &mut constraints,
        DoryAssistPublicId::JoltCommitment,
        JOLT_COMMITMENT_GT_START,
        DoryReducePolynomial::CurrentD1,
    );
    extend_public_gt_constraints(
        &mut constraints,
        DoryAssistPublicId::DoryProofArtifact,
        artifacts::DORY_VMV_D2_START,
        DoryReducePolynomial::CurrentD2,
    );
    extend_public_g1_constraints(
        &mut constraints,
        artifacts::DORY_VMV_E1_START,
        [
            DoryReducePolynomial::CurrentE1X,
            DoryReducePolynomial::CurrentE1Y,
            DoryReducePolynomial::CurrentE1Infinity,
        ],
    );
    constraints.push(copy_constraint(
        DoryAssistValueType::Scalar,
        DoryAssistValueRef::Constant(1),
        reduce_ref(
            DoryAssistRelationId::DoryReduceScalarFold,
            DoryReducePolynomial::S1Accumulator,
            0,
        ),
    ));
    constraints.push(copy_constraint(
        DoryAssistValueType::Scalar,
        DoryAssistValueRef::Constant(1),
        reduce_ref(
            DoryAssistRelationId::DoryReduceScalarFold,
            DoryReducePolynomial::S2Accumulator,
            0,
        ),
    ));
    constraints
}

pub fn state_chaining_copy_constraints(reduce_rounds: usize) -> Vec<DoryAssistCopyConstraint> {
    if reduce_rounds <= 1 {
        return Vec::new();
    }

    let mut constraints = Vec::new();
    extend_gt_state_chaining_constraints(
        &mut constraints,
        DoryReducePolynomial::NextC,
        DoryReducePolynomial::CurrentC,
    );
    extend_gt_state_chaining_constraints(
        &mut constraints,
        DoryReducePolynomial::NextD1,
        DoryReducePolynomial::CurrentD1,
    );
    extend_gt_state_chaining_constraints(
        &mut constraints,
        DoryReducePolynomial::NextD2,
        DoryReducePolynomial::CurrentD2,
    );
    extend_g1_state_chaining_constraints(
        &mut constraints,
        [
            DoryReducePolynomial::NextE1X,
            DoryReducePolynomial::NextE1Y,
            DoryReducePolynomial::NextE1Infinity,
        ],
        [
            DoryReducePolynomial::CurrentE1X,
            DoryReducePolynomial::CurrentE1Y,
            DoryReducePolynomial::CurrentE1Infinity,
        ],
    );
    extend_g2_state_chaining_constraints(
        &mut constraints,
        [
            DoryReducePolynomial::NextE2X0,
            DoryReducePolynomial::NextE2X1,
            DoryReducePolynomial::NextE2Y0,
            DoryReducePolynomial::NextE2Y1,
            DoryReducePolynomial::NextE2Infinity,
        ],
        [
            DoryReducePolynomial::CurrentE2X0,
            DoryReducePolynomial::CurrentE2X1,
            DoryReducePolynomial::CurrentE2Y0,
            DoryReducePolynomial::CurrentE2Y1,
            DoryReducePolynomial::CurrentE2Infinity,
        ],
    );
    constraints.push(state_chaining_constraint(
        DoryAssistValueType::Scalar,
        DoryAssistRelationId::DoryReduceScalarFold,
        DoryReducePolynomial::S1NextAccumulator,
        DoryReducePolynomial::S1Accumulator,
        0,
    ));
    constraints.push(state_chaining_constraint(
        DoryAssistValueType::Scalar,
        DoryAssistRelationId::DoryReduceScalarFold,
        DoryReducePolynomial::S2NextAccumulator,
        DoryReducePolynomial::S2Accumulator,
        0,
    ));
    constraints
}

pub fn public_fold_constraints(
    dimensions: DoryReduceDimensions,
) -> Vec<DoryReducePublicFoldConstraint> {
    proof_artifact_fold_constraints(dimensions.reduce_rounds())
        .into_iter()
        .chain(round_setup_artifact_fold_constraints(
            dimensions.reduce_rounds(),
        ))
        .chain(transition_transcript_scalar_fold_constraints(
            dimensions.point_len(),
            dimensions.reduce_rounds(),
        ))
        .chain(scalar_fold_transcript_scalar_fold_constraints(
            dimensions.point_len(),
            dimensions.reduce_rounds(),
        ))
        .collect()
}

pub fn proof_artifact_fold_constraints(
    reduce_rounds: usize,
) -> Vec<DoryReducePublicFoldConstraint> {
    let mut constraints = Vec::new();
    extend_gt_artifact_fold_constraints(
        &mut constraints,
        reduce_rounds,
        artifacts::reduce_first_d1_left_start,
        DoryReducePolynomial::MessageD1Left,
    );
    extend_gt_artifact_fold_constraints(
        &mut constraints,
        reduce_rounds,
        artifacts::reduce_first_d1_right_start,
        DoryReducePolynomial::MessageD1Right,
    );
    extend_gt_artifact_fold_constraints(
        &mut constraints,
        reduce_rounds,
        artifacts::reduce_first_d2_left_start,
        DoryReducePolynomial::MessageD2Left,
    );
    extend_gt_artifact_fold_constraints(
        &mut constraints,
        reduce_rounds,
        artifacts::reduce_first_d2_right_start,
        DoryReducePolynomial::MessageD2Right,
    );
    extend_g1_artifact_fold_constraints(
        &mut constraints,
        reduce_rounds,
        artifacts::reduce_first_e1_beta_start,
        [
            DoryReducePolynomial::MessageE1BetaX,
            DoryReducePolynomial::MessageE1BetaY,
            DoryReducePolynomial::MessageE1BetaInfinity,
        ],
    );
    extend_g2_artifact_fold_constraints(
        &mut constraints,
        reduce_rounds,
        artifacts::reduce_first_e2_beta_start,
        [
            DoryReducePolynomial::MessageE2BetaX0,
            DoryReducePolynomial::MessageE2BetaX1,
            DoryReducePolynomial::MessageE2BetaY0,
            DoryReducePolynomial::MessageE2BetaY1,
            DoryReducePolynomial::MessageE2BetaInfinity,
        ],
    );
    extend_gt_artifact_fold_constraints(
        &mut constraints,
        reduce_rounds,
        artifacts::reduce_second_c_plus_start,
        DoryReducePolynomial::MessageCPlus,
    );
    extend_gt_artifact_fold_constraints(
        &mut constraints,
        reduce_rounds,
        artifacts::reduce_second_c_minus_start,
        DoryReducePolynomial::MessageCMinus,
    );
    extend_g1_artifact_fold_constraints(
        &mut constraints,
        reduce_rounds,
        artifacts::reduce_second_e1_plus_start,
        [
            DoryReducePolynomial::MessageE1PlusX,
            DoryReducePolynomial::MessageE1PlusY,
            DoryReducePolynomial::MessageE1PlusInfinity,
        ],
    );
    extend_g1_artifact_fold_constraints(
        &mut constraints,
        reduce_rounds,
        artifacts::reduce_second_e1_minus_start,
        [
            DoryReducePolynomial::MessageE1MinusX,
            DoryReducePolynomial::MessageE1MinusY,
            DoryReducePolynomial::MessageE1MinusInfinity,
        ],
    );
    extend_g2_artifact_fold_constraints(
        &mut constraints,
        reduce_rounds,
        artifacts::reduce_second_e2_plus_start,
        [
            DoryReducePolynomial::MessageE2PlusX0,
            DoryReducePolynomial::MessageE2PlusX1,
            DoryReducePolynomial::MessageE2PlusY0,
            DoryReducePolynomial::MessageE2PlusY1,
            DoryReducePolynomial::MessageE2PlusInfinity,
        ],
    );
    extend_g2_artifact_fold_constraints(
        &mut constraints,
        reduce_rounds,
        artifacts::reduce_second_e2_minus_start,
        [
            DoryReducePolynomial::MessageE2MinusX0,
            DoryReducePolynomial::MessageE2MinusX1,
            DoryReducePolynomial::MessageE2MinusY0,
            DoryReducePolynomial::MessageE2MinusY1,
            DoryReducePolynomial::MessageE2MinusInfinity,
        ],
    );
    constraints
}

pub fn round_setup_artifact_fold_constraints(
    reduce_rounds: usize,
) -> Vec<DoryReducePublicFoldConstraint> {
    let mut constraints = Vec::new();
    extend_setup_gt_fold_constraints(
        &mut constraints,
        reduce_rounds,
        |max_rounds, round| {
            setup_artifacts::dory_setup_chi_start(verifier_setup_round(max_rounds, round))
        },
        DoryReducePolynomial::SetupChi,
    );
    extend_setup_gt_fold_constraints(
        &mut constraints,
        reduce_rounds,
        |max_rounds, round| {
            setup_artifacts::dory_setup_delta_1l_start(
                max_rounds,
                verifier_setup_round(max_rounds, round),
            )
        },
        DoryReducePolynomial::SetupDelta1L,
    );
    extend_setup_gt_fold_constraints(
        &mut constraints,
        reduce_rounds,
        |max_rounds, round| {
            setup_artifacts::dory_setup_delta_1r_start(
                max_rounds,
                verifier_setup_round(max_rounds, round),
            )
        },
        DoryReducePolynomial::SetupDelta1R,
    );
    extend_setup_gt_fold_constraints(
        &mut constraints,
        reduce_rounds,
        |max_rounds, round| {
            setup_artifacts::dory_setup_delta_2l_start(
                max_rounds,
                verifier_setup_round(max_rounds, round),
            )
        },
        DoryReducePolynomial::SetupDelta2L,
    );
    extend_setup_gt_fold_constraints(
        &mut constraints,
        reduce_rounds,
        |max_rounds, round| {
            setup_artifacts::dory_setup_delta_2r_start(
                max_rounds,
                verifier_setup_round(max_rounds, round),
            )
        },
        DoryReducePolynomial::SetupDelta2R,
    );
    constraints
}

pub fn transition_transcript_scalar_fold_constraints(
    point_len: usize,
    reduce_rounds: usize,
) -> Vec<DoryReducePublicFoldConstraint> {
    let gt_scalars = [
        (
            transcript_scalars::dory_reduce_beta as fn(usize, usize) -> usize,
            DoryReducePolynomial::Beta,
        ),
        (
            transcript_scalars::dory_reduce_beta_inverse as fn(usize, usize) -> usize,
            DoryReducePolynomial::BetaInverse,
        ),
        (
            transcript_scalars::dory_reduce_alpha as fn(usize, usize) -> usize,
            DoryReducePolynomial::Alpha,
        ),
        (
            transcript_scalars::dory_reduce_alpha_inverse as fn(usize, usize) -> usize,
            DoryReducePolynomial::AlphaInverse,
        ),
        (
            transcript_scalars::dory_reduce_alpha_beta as fn(usize, usize) -> usize,
            DoryReducePolynomial::AlphaBeta,
        ),
        (
            transcript_scalars::dory_reduce_alpha_inverse_beta_inverse as fn(usize, usize) -> usize,
            DoryReducePolynomial::AlphaInverseBetaInverse,
        ),
    ];
    let g1_scalars = [
        (
            transcript_scalars::dory_reduce_beta as fn(usize, usize) -> usize,
            DoryReducePolynomial::Beta,
        ),
        (
            transcript_scalars::dory_reduce_alpha as fn(usize, usize) -> usize,
            DoryReducePolynomial::Alpha,
        ),
        (
            transcript_scalars::dory_reduce_alpha_inverse as fn(usize, usize) -> usize,
            DoryReducePolynomial::AlphaInverse,
        ),
    ];
    let g2_scalars = [
        (
            transcript_scalars::dory_reduce_beta_inverse as fn(usize, usize) -> usize,
            DoryReducePolynomial::BetaInverse,
        ),
        (
            transcript_scalars::dory_reduce_alpha as fn(usize, usize) -> usize,
            DoryReducePolynomial::Alpha,
        ),
        (
            transcript_scalars::dory_reduce_alpha_inverse as fn(usize, usize) -> usize,
            DoryReducePolynomial::AlphaInverse,
        ),
    ];

    gt_scalars
        .into_iter()
        .map(|(source, polynomial)| {
            transcript_scalar_fold_constraint(
                point_len,
                reduce_rounds,
                source,
                DoryAssistRelationId::DoryReduceGtTransition,
                polynomial,
            )
        })
        .chain(g1_scalars.into_iter().map(|(source, polynomial)| {
            transcript_scalar_fold_constraint(
                point_len,
                reduce_rounds,
                source,
                DoryAssistRelationId::DoryReduceG1Transition,
                polynomial,
            )
        }))
        .chain(g2_scalars.into_iter().map(|(source, polynomial)| {
            transcript_scalar_fold_constraint(
                point_len,
                reduce_rounds,
                source,
                DoryAssistRelationId::DoryReduceG2Transition,
                polynomial,
            )
        }))
        .collect()
}

pub fn scalar_fold_transcript_scalar_fold_constraints(
    point_len: usize,
    reduce_rounds: usize,
) -> Vec<DoryReducePublicFoldConstraint> {
    [
        (
            transcript_scalars::dory_reduce_s1_fold_factor as fn(usize, usize) -> usize,
            DoryReducePolynomial::S1FoldFactor,
        ),
        (
            transcript_scalars::dory_reduce_s2_fold_factor as fn(usize, usize) -> usize,
            DoryReducePolynomial::S2FoldFactor,
        ),
    ]
    .into_iter()
    .map(|(source, polynomial)| {
        transcript_scalar_fold_constraint(
            point_len,
            reduce_rounds,
            source,
            DoryAssistRelationId::DoryReduceScalarFold,
            polynomial,
        )
    })
    .collect()
}

pub fn transcript_scalar_copy_constraints(
    point_len: usize,
    round: usize,
) -> Vec<DoryAssistCopyConstraint> {
    [
        (
            transcript_scalars::dory_reduce_beta(point_len, round),
            DoryReducePolynomial::Beta,
        ),
        (
            transcript_scalars::dory_reduce_beta_inverse(point_len, round),
            DoryReducePolynomial::BetaInverse,
        ),
        (
            transcript_scalars::dory_reduce_alpha(point_len, round),
            DoryReducePolynomial::Alpha,
        ),
        (
            transcript_scalars::dory_reduce_alpha_inverse(point_len, round),
            DoryReducePolynomial::AlphaInverse,
        ),
        (
            transcript_scalars::dory_reduce_alpha_beta(point_len, round),
            DoryReducePolynomial::AlphaBeta,
        ),
        (
            transcript_scalars::dory_reduce_alpha_inverse_beta_inverse(point_len, round),
            DoryReducePolynomial::AlphaInverseBetaInverse,
        ),
        (
            transcript_scalars::dory_reduce_s1_fold_factor(point_len, round),
            DoryReducePolynomial::S1FoldFactor,
        ),
        (
            transcript_scalars::dory_reduce_s2_fold_factor(point_len, round),
            DoryReducePolynomial::S2FoldFactor,
        ),
    ]
    .into_iter()
    .flat_map(|(public_index, polynomial)| {
        [
            scalar_copy_constraint(
                public_index,
                DoryAssistRelationId::DoryReduceGtTransition,
                polynomial,
            ),
            scalar_copy_constraint(
                public_index,
                DoryAssistRelationId::DoryReduceG1Transition,
                polynomial,
            ),
            scalar_copy_constraint(
                public_index,
                DoryAssistRelationId::DoryReduceG2Transition,
                polynomial,
            ),
            scalar_copy_constraint(
                public_index,
                DoryAssistRelationId::DoryReduceScalarFold,
                polynomial,
            ),
        ]
    })
    .collect()
}

pub fn transition_transcript_scalar_copy_constraints(
    point_len: usize,
    round: usize,
) -> Vec<DoryAssistCopyConstraint> {
    let gt_scalars = [
        (
            transcript_scalars::dory_reduce_beta(point_len, round),
            DoryReducePolynomial::Beta,
        ),
        (
            transcript_scalars::dory_reduce_beta_inverse(point_len, round),
            DoryReducePolynomial::BetaInverse,
        ),
        (
            transcript_scalars::dory_reduce_alpha(point_len, round),
            DoryReducePolynomial::Alpha,
        ),
        (
            transcript_scalars::dory_reduce_alpha_inverse(point_len, round),
            DoryReducePolynomial::AlphaInverse,
        ),
        (
            transcript_scalars::dory_reduce_alpha_beta(point_len, round),
            DoryReducePolynomial::AlphaBeta,
        ),
        (
            transcript_scalars::dory_reduce_alpha_inverse_beta_inverse(point_len, round),
            DoryReducePolynomial::AlphaInverseBetaInverse,
        ),
    ];
    let g1_scalars = [
        (
            transcript_scalars::dory_reduce_beta(point_len, round),
            DoryReducePolynomial::Beta,
        ),
        (
            transcript_scalars::dory_reduce_alpha(point_len, round),
            DoryReducePolynomial::Alpha,
        ),
        (
            transcript_scalars::dory_reduce_alpha_inverse(point_len, round),
            DoryReducePolynomial::AlphaInverse,
        ),
    ];
    let g2_scalars = [
        (
            transcript_scalars::dory_reduce_beta_inverse(point_len, round),
            DoryReducePolynomial::BetaInverse,
        ),
        (
            transcript_scalars::dory_reduce_alpha(point_len, round),
            DoryReducePolynomial::Alpha,
        ),
        (
            transcript_scalars::dory_reduce_alpha_inverse(point_len, round),
            DoryReducePolynomial::AlphaInverse,
        ),
    ];

    gt_scalars
        .into_iter()
        .map(|(public_index, polynomial)| {
            scalar_copy_constraint(
                public_index,
                DoryAssistRelationId::DoryReduceGtTransition,
                polynomial,
            )
        })
        .chain(g1_scalars.into_iter().map(|(public_index, polynomial)| {
            scalar_copy_constraint(
                public_index,
                DoryAssistRelationId::DoryReduceG1Transition,
                polynomial,
            )
        }))
        .chain(g2_scalars.into_iter().map(|(public_index, polynomial)| {
            scalar_copy_constraint(
                public_index,
                DoryAssistRelationId::DoryReduceG2Transition,
                polynomial,
            )
        }))
        .collect()
}

pub fn scalar_fold_transcript_scalar_copy_constraints(
    point_len: usize,
    round: usize,
) -> Vec<DoryAssistCopyConstraint> {
    [
        (
            transcript_scalars::dory_reduce_s1_fold_factor(point_len, round),
            DoryReducePolynomial::S1FoldFactor,
        ),
        (
            transcript_scalars::dory_reduce_s2_fold_factor(point_len, round),
            DoryReducePolynomial::S2FoldFactor,
        ),
    ]
    .into_iter()
    .map(|(public_index, polynomial)| {
        scalar_copy_constraint(
            public_index,
            DoryAssistRelationId::DoryReduceScalarFold,
            polynomial,
        )
    })
    .collect()
}

fn extend_gt_artifact_fold_constraints(
    constraints: &mut Vec<DoryReducePublicFoldConstraint>,
    reduce_rounds: usize,
    start: fn(usize) -> usize,
    polynomial: impl Fn(usize) -> DoryReducePolynomial,
) {
    for component in 0..artifacts::GT_ARTIFACT_COEFFS {
        constraints.push(public_fold_constraint(
            DoryAssistValueType::Gt,
            (0..reduce_rounds)
                .map(|round| DoryAssistPublicId::DoryProofArtifact(start(round) + component))
                .collect(),
            reduce_ref(
                DoryAssistRelationId::DoryReduceGtTransition,
                polynomial(component),
                component,
            ),
        ));
    }
}

fn extend_setup_gt_fold_constraints(
    constraints: &mut Vec<DoryReducePublicFoldConstraint>,
    reduce_rounds: usize,
    start: fn(usize, usize) -> usize,
    polynomial: impl Fn(usize) -> DoryReducePolynomial,
) {
    for component in 0..artifacts::GT_ARTIFACT_COEFFS {
        constraints.push(public_fold_constraint(
            DoryAssistValueType::Gt,
            (0..reduce_rounds)
                .map(|round| {
                    DoryAssistPublicId::VerifierSetupArtifact(
                        start(reduce_rounds, round) + component,
                    )
                })
                .collect(),
            reduce_ref(
                DoryAssistRelationId::DoryReduceGtTransition,
                polynomial(component),
                component,
            ),
        ));
    }
}

fn extend_g1_artifact_fold_constraints(
    constraints: &mut Vec<DoryReducePublicFoldConstraint>,
    reduce_rounds: usize,
    start: fn(usize) -> usize,
    polynomials: [DoryReducePolynomial; artifacts::G1_ARTIFACT_COORDS],
) {
    for (component, polynomial) in polynomials.into_iter().enumerate() {
        constraints.push(public_fold_constraint(
            DoryAssistValueType::G1,
            (0..reduce_rounds)
                .map(|round| DoryAssistPublicId::DoryProofArtifact(start(round) + component))
                .collect(),
            reduce_ref(
                DoryAssistRelationId::DoryReduceG1Transition,
                polynomial,
                component,
            ),
        ));
    }
}

fn extend_g2_artifact_fold_constraints(
    constraints: &mut Vec<DoryReducePublicFoldConstraint>,
    reduce_rounds: usize,
    start: fn(usize) -> usize,
    polynomials: [DoryReducePolynomial; artifacts::G2_ARTIFACT_COORDS],
) {
    for (component, polynomial) in polynomials.into_iter().enumerate() {
        constraints.push(public_fold_constraint(
            DoryAssistValueType::G2,
            (0..reduce_rounds)
                .map(|round| DoryAssistPublicId::DoryProofArtifact(start(round) + component))
                .collect(),
            reduce_ref(
                DoryAssistRelationId::DoryReduceG2Transition,
                polynomial,
                component,
            ),
        ));
    }
}

fn transcript_scalar_fold_constraint(
    point_len: usize,
    reduce_rounds: usize,
    source: fn(usize, usize) -> usize,
    relation: DoryAssistRelationId,
    polynomial: DoryReducePolynomial,
) -> DoryReducePublicFoldConstraint {
    public_fold_constraint(
        DoryAssistValueType::Scalar,
        (0..reduce_rounds)
            .map(|round| DoryAssistPublicId::TranscriptScalar(source(point_len, round)))
            .collect(),
        reduce_ref(relation, polynomial, 0),
    )
}

fn public_fold_constraint(
    value_type: DoryAssistValueType,
    sources: Vec<DoryAssistPublicId>,
    target: DoryAssistValueRef,
) -> DoryReducePublicFoldConstraint {
    DoryReducePublicFoldConstraint::new(value_type, sources, target)
}

fn extend_gt_boundary_terms(
    terms: &mut Vec<DoryReduceBoundaryTerm>,
    polynomial: impl Fn(usize) -> DoryReducePolynomial,
    public_id: impl Fn(usize) -> DoryAssistPublicId,
    start: usize,
) {
    for component in 0..artifacts::GT_ARTIFACT_COEFFS {
        terms.push(DoryReduceBoundaryTerm {
            opening: dory_reduce_boundary_opening(polynomial(component)),
            value: DoryReduceBoundaryValue::Public(public_id(start + component)),
        });
    }
}

fn extend_g1_boundary_terms(
    terms: &mut Vec<DoryReduceBoundaryTerm>,
    polynomials: [DoryReducePolynomial; artifacts::G1_ARTIFACT_COORDS],
    public_id: impl Fn(usize) -> DoryAssistPublicId,
    start: usize,
) {
    for (component, polynomial) in polynomials.into_iter().enumerate() {
        terms.push(DoryReduceBoundaryTerm {
            opening: dory_reduce_boundary_opening(polynomial),
            value: DoryReduceBoundaryValue::Public(public_id(start + component)),
        });
    }
}

fn extend_g2_boundary_terms(
    terms: &mut Vec<DoryReduceBoundaryTerm>,
    polynomials: [DoryReducePolynomial; artifacts::G2_ARTIFACT_COORDS],
    public_id: impl Fn(usize) -> DoryAssistPublicId,
    start: usize,
) {
    for (component, polynomial) in polynomials.into_iter().enumerate() {
        terms.push(DoryReduceBoundaryTerm {
            opening: dory_reduce_boundary_opening(polynomial),
            value: DoryReduceBoundaryValue::Public(public_id(start + component)),
        });
    }
}

fn extend_gt_artifact_constraints(
    constraints: &mut Vec<DoryAssistCopyConstraint>,
    start: usize,
    polynomial: impl Fn(usize) -> DoryReducePolynomial,
) {
    for component in 0..artifacts::GT_ARTIFACT_COEFFS {
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            DoryAssistValueRef::public(
                DoryAssistPublicId::DoryProofArtifact(start + component),
                component,
            ),
            reduce_ref(
                DoryAssistRelationId::DoryReduceGtTransition,
                polynomial(component),
                component,
            ),
        ));
    }
}

fn extend_public_gt_constraints(
    constraints: &mut Vec<DoryAssistCopyConstraint>,
    public_id: impl Fn(usize) -> DoryAssistPublicId,
    start: usize,
    polynomial: impl Fn(usize) -> DoryReducePolynomial,
) {
    for component in 0..artifacts::GT_ARTIFACT_COEFFS {
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            DoryAssistValueRef::public(public_id(start + component), component),
            reduce_ref(
                DoryAssistRelationId::DoryReduceGtTransition,
                polynomial(component),
                component,
            ),
        ));
    }
}

fn extend_setup_gt_constraints(
    constraints: &mut Vec<DoryAssistCopyConstraint>,
    start: usize,
    polynomial: impl Fn(usize) -> DoryReducePolynomial,
) {
    for component in 0..artifacts::GT_ARTIFACT_COEFFS {
        constraints.push(copy_constraint(
            DoryAssistValueType::Gt,
            DoryAssistValueRef::public(
                DoryAssistPublicId::VerifierSetupArtifact(start + component),
                component,
            ),
            reduce_ref(
                DoryAssistRelationId::DoryReduceGtTransition,
                polynomial(component),
                component,
            ),
        ));
    }
}

fn extend_public_g1_constraints(
    constraints: &mut Vec<DoryAssistCopyConstraint>,
    start: usize,
    polynomials: [DoryReducePolynomial; artifacts::G1_ARTIFACT_COORDS],
) {
    for (component, polynomial) in polynomials.into_iter().enumerate() {
        constraints.push(copy_constraint(
            DoryAssistValueType::G1,
            DoryAssistValueRef::public(
                DoryAssistPublicId::DoryProofArtifact(start + component),
                component,
            ),
            reduce_ref(
                DoryAssistRelationId::DoryReduceG1Transition,
                polynomial,
                component,
            ),
        ));
    }
}

fn extend_g1_artifact_constraints(
    constraints: &mut Vec<DoryAssistCopyConstraint>,
    start: usize,
    polynomials: [DoryReducePolynomial; artifacts::G1_ARTIFACT_COORDS],
) {
    for (component, polynomial) in polynomials.into_iter().enumerate() {
        constraints.push(copy_constraint(
            DoryAssistValueType::G1,
            DoryAssistValueRef::public(
                DoryAssistPublicId::DoryProofArtifact(start + component),
                component,
            ),
            reduce_ref(
                DoryAssistRelationId::DoryReduceG1Transition,
                polynomial,
                component,
            ),
        ));
    }
}

fn extend_g2_artifact_constraints(
    constraints: &mut Vec<DoryAssistCopyConstraint>,
    start: usize,
    polynomials: [DoryReducePolynomial; artifacts::G2_ARTIFACT_COORDS],
) {
    for (component, polynomial) in polynomials.into_iter().enumerate() {
        constraints.push(copy_constraint(
            DoryAssistValueType::G2,
            DoryAssistValueRef::public(
                DoryAssistPublicId::DoryProofArtifact(start + component),
                component,
            ),
            reduce_ref(
                DoryAssistRelationId::DoryReduceG2Transition,
                polynomial,
                component,
            ),
        ));
    }
}

fn extend_gt_state_chaining_constraints(
    constraints: &mut Vec<DoryAssistCopyConstraint>,
    source: impl Fn(usize) -> DoryReducePolynomial,
    target: impl Fn(usize) -> DoryReducePolynomial,
) {
    for component in 0..artifacts::GT_ARTIFACT_COEFFS {
        constraints.push(state_chaining_constraint(
            DoryAssistValueType::Gt,
            DoryAssistRelationId::DoryReduceGtTransition,
            source(component),
            target(component),
            component,
        ));
    }
}

fn extend_g1_state_chaining_constraints(
    constraints: &mut Vec<DoryAssistCopyConstraint>,
    sources: [DoryReducePolynomial; artifacts::G1_ARTIFACT_COORDS],
    targets: [DoryReducePolynomial; artifacts::G1_ARTIFACT_COORDS],
) {
    for (component, (source, target)) in sources.into_iter().zip(targets).enumerate() {
        constraints.push(state_chaining_constraint(
            DoryAssistValueType::G1,
            DoryAssistRelationId::DoryReduceG1Transition,
            source,
            target,
            component,
        ));
    }
}

fn extend_g2_state_chaining_constraints(
    constraints: &mut Vec<DoryAssistCopyConstraint>,
    sources: [DoryReducePolynomial; artifacts::G2_ARTIFACT_COORDS],
    targets: [DoryReducePolynomial; artifacts::G2_ARTIFACT_COORDS],
) {
    for (component, (source, target)) in sources.into_iter().zip(targets).enumerate() {
        constraints.push(state_chaining_constraint(
            DoryAssistValueType::G2,
            DoryAssistRelationId::DoryReduceG2Transition,
            source,
            target,
            component,
        ));
    }
}

fn state_chaining_constraint(
    value_type: DoryAssistValueType,
    relation: DoryAssistRelationId,
    source: DoryReducePolynomial,
    target: DoryReducePolynomial,
    component: usize,
) -> DoryAssistCopyConstraint {
    copy_constraint(
        value_type,
        reduce_ref_at_row(relation, source, LOCAL_ROW, component),
        reduce_ref_at_row(relation, target, NEXT_ROW, component),
    )
}

fn scalar_copy_constraint(
    public_index: usize,
    relation: DoryAssistRelationId,
    polynomial: DoryReducePolynomial,
) -> DoryAssistCopyConstraint {
    copy_constraint(
        DoryAssistValueType::Scalar,
        DoryAssistValueRef::public(DoryAssistPublicId::TranscriptScalar(public_index), 0),
        reduce_ref(relation, polynomial, 0),
    )
}

fn transition_relation<F>(
    relation: DoryAssistRelationId,
    batch_challenge: DoryReduceChallenge,
    dimensions: DoryReduceDimensions,
    updates: &[DoryReduceUpdate],
) -> DoryAssistRelationClaims<F>
where
    F: RingCore,
{
    let gamma = dory_reduce_challenge(batch_challenge);
    let (input, output) = updates.iter().enumerate().fold(
        (constant(F::zero()), constant(F::zero())),
        |(input, output), (index, update)| {
            let weight = gamma.clone().pow(index);
            (
                input + weight.clone() * opening(dory_reduce_opening_for(relation, update.target)),
                output + weight * update_expression(relation, update),
            )
        },
    );

    DoryAssistRelationClaims::new(relation, transition_sumcheck(dimensions), input, output)
}

fn update_expression<F>(
    relation: DoryAssistRelationId,
    update: &DoryReduceUpdate,
) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    update.terms.iter().fold(constant(F::zero()), |acc, term| {
        acc + scalar_expression(relation, term.scalar)
            * opening(dory_reduce_opening_for(relation, term.value))
    })
}

fn scalar_expression<F>(
    relation: DoryAssistRelationId,
    scalar: DoryReduceScalarTerm,
) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    match scalar_polynomial(scalar) {
        Some(polynomial) => opening(dory_reduce_opening_for(relation, polynomial)),
        None => constant(F::one()),
    }
}

fn transition_openings<I>(relation: DoryAssistRelationId, updates: I) -> Vec<DoryAssistOpeningId>
where
    I: IntoIterator<Item = DoryReduceUpdate>,
{
    let mut openings = Vec::new();
    for update in updates {
        openings.push(dory_reduce_opening_for(relation, update.target));
        for term in update.terms {
            if let Some(polynomial) = scalar_polynomial(term.scalar) {
                openings.push(dory_reduce_opening_for(relation, polynomial));
            }
            openings.push(dory_reduce_opening_for(relation, term.value));
        }
    }
    openings
}

fn combine_state_chain_openings<F>(
    gamma: DoryAssistExpr<F>,
    openings: impl IntoIterator<Item = DoryAssistOpeningId>,
) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    openings
        .into_iter()
        .enumerate()
        .fold(constant(F::zero()), |acc, (index, id)| {
            acc + gamma.clone().pow(index) * opening(id)
        })
}

fn combine_boundary_differences<F>(
    gamma: DoryAssistExpr<F>,
    terms: impl IntoIterator<Item = DoryReduceBoundaryTerm>,
) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    terms
        .into_iter()
        .enumerate()
        .fold(constant(F::zero()), |acc, (index, term)| {
            acc + gamma.clone().pow(index)
                * (opening(term.opening) - boundary_term_value(term.value))
        })
}

fn boundary_term_value<F>(value: DoryReduceBoundaryValue) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    match value {
        DoryReduceBoundaryValue::ConstantOne => constant(F::one()),
        DoryReduceBoundaryValue::Public(id) => public(id),
    }
}

fn boundary_selector<F>(
    relation: DoryAssistRelationId,
    endpoint: DoryAssistBoundaryEndpoint,
) -> DoryAssistExpr<F>
where
    F: RingCore,
{
    public(DoryAssistPublicId::BoundarySelector { relation, endpoint })
}

fn scalar_polynomial(scalar: DoryReduceScalarTerm) -> Option<DoryReducePolynomial> {
    match scalar {
        DoryReduceScalarTerm::One => None,
        DoryReduceScalarTerm::Beta => Some(DoryReducePolynomial::Beta),
        DoryReduceScalarTerm::BetaInverse => Some(DoryReducePolynomial::BetaInverse),
        DoryReduceScalarTerm::Alpha => Some(DoryReducePolynomial::Alpha),
        DoryReduceScalarTerm::AlphaInverse => Some(DoryReducePolynomial::AlphaInverse),
        DoryReduceScalarTerm::AlphaBeta => Some(DoryReducePolynomial::AlphaBeta),
        DoryReduceScalarTerm::AlphaInverseBetaInverse => {
            Some(DoryReducePolynomial::AlphaInverseBetaInverse)
        }
        DoryReduceScalarTerm::S1FoldFactor => Some(DoryReducePolynomial::S1FoldFactor),
        DoryReduceScalarTerm::S2FoldFactor => Some(DoryReducePolynomial::S2FoldFactor),
    }
}

fn copy_constraint(
    value_type: DoryAssistValueType,
    source: DoryAssistValueRef,
    target: DoryAssistValueRef,
) -> DoryAssistCopyConstraint {
    DoryAssistCopyConstraint::new(value_type, source, target)
}

fn reduce_ref(
    relation: DoryAssistRelationId,
    polynomial: DoryReducePolynomial,
    component: usize,
) -> DoryAssistValueRef {
    reduce_ref_at_row(relation, polynomial, LOCAL_ROW, component)
}

fn reduce_ref_at_row(
    relation: DoryAssistRelationId,
    polynomial: DoryReducePolynomial,
    row: usize,
    component: usize,
) -> DoryAssistValueRef {
    DoryAssistValueRef::witness(
        relation,
        DoryAssistVirtualPolynomial::DoryReduce(polynomial),
        row,
        component,
    )
}

fn dory_reduce_challenge<F>(
    challenge_id: DoryReduceChallenge,
) -> crate::Expr<F, DoryAssistOpeningId, DoryAssistPublicId, DoryAssistChallengeId>
where
    F: RingCore,
{
    challenge(DoryAssistChallengeId::from(challenge_id))
}

fn dory_reduce_opening(polynomial: DoryReducePolynomial) -> DoryAssistOpeningId {
    dory_reduce_opening_for(DoryAssistRelationId::DoryReduceScalarFold, polynomial)
}

fn dory_reduce_state_chain_opening(polynomial: DoryReducePolynomial) -> DoryAssistOpeningId {
    dory_reduce_opening_for(DoryAssistRelationId::DoryReduceStateChain, polynomial)
}

fn dory_reduce_boundary_opening(polynomial: DoryReducePolynomial) -> DoryAssistOpeningId {
    dory_reduce_opening_for(DoryAssistRelationId::DoryReduceBoundary, polynomial)
}

fn dory_reduce_opening_for(
    relation: DoryAssistRelationId,
    polynomial: DoryReducePolynomial,
) -> DoryAssistOpeningId {
    DoryAssistOpeningId::virtual_polynomial(
        DoryAssistVirtualPolynomial::DoryReduce(polynomial),
        relation,
    )
}

fn g1_current_polynomial(component: usize) -> DoryReducePolynomial {
    match component {
        0 => DoryReducePolynomial::CurrentE1X,
        1 => DoryReducePolynomial::CurrentE1Y,
        _ => DoryReducePolynomial::CurrentE1Infinity,
    }
}

fn g1_next_polynomial(component: usize) -> DoryReducePolynomial {
    match component {
        0 => DoryReducePolynomial::NextE1X,
        1 => DoryReducePolynomial::NextE1Y,
        _ => DoryReducePolynomial::NextE1Infinity,
    }
}

fn g1_beta_polynomial(component: usize) -> DoryReducePolynomial {
    match component {
        0 => DoryReducePolynomial::MessageE1BetaX,
        1 => DoryReducePolynomial::MessageE1BetaY,
        _ => DoryReducePolynomial::MessageE1BetaInfinity,
    }
}

fn g1_plus_polynomial(component: usize) -> DoryReducePolynomial {
    match component {
        0 => DoryReducePolynomial::MessageE1PlusX,
        1 => DoryReducePolynomial::MessageE1PlusY,
        _ => DoryReducePolynomial::MessageE1PlusInfinity,
    }
}

fn g1_minus_polynomial(component: usize) -> DoryReducePolynomial {
    match component {
        0 => DoryReducePolynomial::MessageE1MinusX,
        1 => DoryReducePolynomial::MessageE1MinusY,
        _ => DoryReducePolynomial::MessageE1MinusInfinity,
    }
}

fn g2_current_polynomial(component: usize) -> DoryReducePolynomial {
    match component {
        0 => DoryReducePolynomial::CurrentE2X0,
        1 => DoryReducePolynomial::CurrentE2X1,
        2 => DoryReducePolynomial::CurrentE2Y0,
        3 => DoryReducePolynomial::CurrentE2Y1,
        _ => DoryReducePolynomial::CurrentE2Infinity,
    }
}

fn g2_next_polynomial(component: usize) -> DoryReducePolynomial {
    match component {
        0 => DoryReducePolynomial::NextE2X0,
        1 => DoryReducePolynomial::NextE2X1,
        2 => DoryReducePolynomial::NextE2Y0,
        3 => DoryReducePolynomial::NextE2Y1,
        _ => DoryReducePolynomial::NextE2Infinity,
    }
}

fn g2_beta_polynomial(component: usize) -> DoryReducePolynomial {
    match component {
        0 => DoryReducePolynomial::MessageE2BetaX0,
        1 => DoryReducePolynomial::MessageE2BetaX1,
        2 => DoryReducePolynomial::MessageE2BetaY0,
        3 => DoryReducePolynomial::MessageE2BetaY1,
        _ => DoryReducePolynomial::MessageE2BetaInfinity,
    }
}

fn g2_plus_polynomial(component: usize) -> DoryReducePolynomial {
    match component {
        0 => DoryReducePolynomial::MessageE2PlusX0,
        1 => DoryReducePolynomial::MessageE2PlusX1,
        2 => DoryReducePolynomial::MessageE2PlusY0,
        3 => DoryReducePolynomial::MessageE2PlusY1,
        _ => DoryReducePolynomial::MessageE2PlusInfinity,
    }
}

fn g2_minus_polynomial(component: usize) -> DoryReducePolynomial {
    match component {
        0 => DoryReducePolynomial::MessageE2MinusX0,
        1 => DoryReducePolynomial::MessageE2MinusX1,
        2 => DoryReducePolynomial::MessageE2MinusY0,
        3 => DoryReducePolynomial::MessageE2MinusY1,
        _ => DoryReducePolynomial::MessageE2MinusInfinity,
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests fail loudly on missing semantic constraints"
    )]

    use jolt_field::{Fr, FromPrimitiveInt};

    use super::*;

    #[test]
    fn scalar_fold_relation_batches_s1_and_s2_accumulator_updates() {
        let dimensions = DoryReduceDimensions::new(2, 1);
        let relation = scalar_fold::<Fr>(dimensions);

        assert_eq!(relation.id, DoryAssistRelationId::DoryReduceScalarFold);
        assert_eq!(relation.sumcheck, dimensions.scalar_fold_sumcheck());
        assert_eq!(
            relation.input.required_openings,
            vec![s1_next_accumulator_opening(), s2_next_accumulator_opening()]
        );
        assert_eq!(
            relation.output.required_openings,
            vec![
                s1_accumulator_opening(),
                s1_fold_factor_opening(),
                s2_accumulator_opening(),
                s2_fold_factor_opening(),
            ]
        );
        assert_eq!(
            relation.required_challenges(),
            vec![DoryAssistChallengeId::from(
                DoryReduceChallenge::ScalarFoldBatch
            )]
        );
    }

    #[test]
    fn state_chain_relation_batches_next_to_following_current_state() {
        let dimensions = DoryReduceDimensions::new(4, 2);
        let relation = state_chain::<Fr>(dimensions);
        let input_openings = state_chain_input_openings();
        let output_openings = state_chain_output_openings();

        assert_eq!(relation.id, DoryAssistRelationId::DoryReduceStateChain);
        assert_eq!(relation.sumcheck, dimensions.state_chain_sumcheck());
        assert_eq!(
            relation.required_challenges(),
            vec![DoryAssistChallengeId::from(
                DoryReduceChallenge::StateChainBatch
            )]
        );
        assert_eq!(
            relation.required_publics(),
            vec![DoryAssistPublicId::DoryReduceShiftEqKernel]
        );
        assert_eq!(
            input_openings.len(),
            3 * artifacts::GT_ARTIFACT_COEFFS
                + artifacts::G1_ARTIFACT_COORDS
                + artifacts::G2_ARTIFACT_COORDS
                + 2
        );
        assert_eq!(input_openings.len(), output_openings.len());
        assert!(input_openings.contains(&dory_reduce_state_chain_opening(
            DoryReducePolynomial::NextC(0)
        )));
        assert!(output_openings.contains(&dory_reduce_state_chain_opening(
            DoryReducePolynomial::CurrentC(0)
        )));
        assert!(input_openings.contains(&dory_reduce_state_chain_opening(
            DoryReducePolynomial::S2NextAccumulator
        )));
        assert!(output_openings.contains(&dory_reduce_state_chain_opening(
            DoryReducePolynomial::S2Accumulator
        )));
    }

    #[test]
    fn boundary_relation_binds_initial_checked_inputs_and_final_native_inputs() {
        let dimensions = DoryReduceDimensions::new(4, 2);
        let relation = boundary::<Fr>(dimensions);
        let openings = boundary_output_openings();
        let required_publics = relation.required_publics();

        assert_eq!(relation.id, DoryAssistRelationId::DoryReduceBoundary);
        assert_eq!(relation.sumcheck, dimensions.boundary_sumcheck());
        assert!(relation.input.required_openings.is_empty());
        assert_eq!(relation.output.required_openings, openings);
        assert_eq!(
            relation.required_challenges(),
            vec![DoryAssistChallengeId::from(
                DoryReduceChallenge::BoundaryPoint
            )]
        );
        assert!(
            required_publics.contains(&DoryAssistPublicId::BoundarySelector {
                relation: DoryAssistRelationId::DoryReduceBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Initial,
            })
        );
        assert!(
            required_publics.contains(&DoryAssistPublicId::BoundarySelector {
                relation: DoryAssistRelationId::DoryReduceBoundary,
                endpoint: DoryAssistBoundaryEndpoint::Final,
            })
        );
        assert!(required_publics.contains(&DoryAssistPublicId::DoryReduceInitialE2(0)));
        assert!(
            required_publics.contains(&DoryAssistPublicId::NativeFinalCheckInput(
                DORY_REDUCE_NATIVE_FINAL_GT_C_START
            ))
        );
        assert!(openings.contains(&dory_reduce_boundary_opening(
            DoryReducePolynomial::CurrentE2X0
        )));
        assert!(
            openings.contains(&dory_reduce_boundary_opening(DoryReducePolynomial::NextC(
                0
            )))
        );
        assert!(openings.contains(&dory_reduce_boundary_opening(
            DoryReducePolynomial::S2NextAccumulator
        )));
    }

    #[test]
    fn boundary_relation_detects_mismatched_final_reducer_state() {
        let dimensions = DoryReduceDimensions::new(4, 2);
        let relation = boundary::<Fr>(dimensions);

        let output = relation.output.expression().evaluate(
            |opening| match *opening {
                id if id == dory_reduce_boundary_opening(DoryReducePolynomial::NextC(0)) => {
                    Fr::from_u64(7)
                }
                id if id == dory_reduce_boundary_opening(DoryReducePolynomial::S1Accumulator) => {
                    Fr::from_u64(1)
                }
                id if id == dory_reduce_boundary_opening(DoryReducePolynomial::S2Accumulator) => {
                    Fr::from_u64(1)
                }
                _ => Fr::from_u64(0),
            },
            |_| Fr::from_u64(3),
            |public_id| match *public_id {
                DoryAssistPublicId::BoundarySelector {
                    relation: DoryAssistRelationId::DoryReduceBoundary,
                    ..
                } => Fr::from_u64(1),
                DoryAssistPublicId::NativeFinalCheckInput(DORY_REDUCE_NATIVE_FINAL_GT_C_START) => {
                    Fr::from_u64(5)
                }
                DoryAssistPublicId::DoryProofArtifact(artifacts::DORY_VMV_C_START) => {
                    Fr::from_u64(0)
                }
                DoryAssistPublicId::JoltCommitment(JOLT_COMMITMENT_GT_START) => Fr::from_u64(0),
                DoryAssistPublicId::DoryReduceInitialE2(_) => Fr::from_u64(0),
                _ => Fr::from_u64(0),
            },
        );

        assert_ne!(output, Fr::from_u64(0));
    }

    #[test]
    fn gt_transition_relation_batches_dory_reduce_gt_updates() {
        let dimensions = DoryReduceDimensions::new(2, 1);
        let relation = gt_transition::<Fr>(dimensions);

        assert_eq!(relation.id, DoryAssistRelationId::DoryReduceGtTransition);
        assert_eq!(relation.sumcheck, transition_sumcheck(dimensions));
        assert_eq!(
            relation.required_challenges(),
            vec![DoryAssistChallengeId::from(
                DoryReduceChallenge::GtTransitionBatch
            )]
        );
        assert!(relation
            .required_openings()
            .contains(&dory_reduce_opening_for(
                DoryAssistRelationId::DoryReduceGtTransition,
                DoryReducePolynomial::NextC(0),
            )));
        assert!(relation
            .required_openings()
            .contains(&dory_reduce_opening_for(
                DoryAssistRelationId::DoryReduceGtTransition,
                DoryReducePolynomial::AlphaBeta,
            )));
        assert!(relation
            .required_openings()
            .contains(&dory_reduce_opening_for(
                DoryAssistRelationId::DoryReduceGtTransition,
                DoryReducePolynomial::SetupDelta1L(0),
            )));
    }

    #[test]
    fn g1_and_g2_transition_relations_batch_coordinate_updates() {
        let dimensions = DoryReduceDimensions::new(2, 1);
        let g1_relation = g1_transition::<Fr>(dimensions);
        let g2_relation = g2_transition::<Fr>(dimensions);

        assert_eq!(g1_relation.id, DoryAssistRelationId::DoryReduceG1Transition);
        assert_eq!(g2_relation.id, DoryAssistRelationId::DoryReduceG2Transition);
        assert_eq!(g1_relation.sumcheck, transition_sumcheck(dimensions));
        assert_eq!(g2_relation.sumcheck, transition_sumcheck(dimensions));
        assert_eq!(
            g1_relation.required_challenges(),
            vec![DoryAssistChallengeId::from(
                DoryReduceChallenge::G1TransitionBatch
            )]
        );
        assert_eq!(
            g2_relation.required_challenges(),
            vec![DoryAssistChallengeId::from(
                DoryReduceChallenge::G2TransitionBatch
            )]
        );
        assert!(g1_relation
            .required_openings()
            .contains(&dory_reduce_opening_for(
                DoryAssistRelationId::DoryReduceG1Transition,
                DoryReducePolynomial::NextE1X,
            )));
        assert!(g2_relation
            .required_openings()
            .contains(&dory_reduce_opening_for(
                DoryAssistRelationId::DoryReduceG2Transition,
                DoryReducePolynomial::NextE2X0,
            )));
    }

    #[test]
    fn gt_transition_terms_match_dory_verifier_state_updates() {
        let updates = gt_transition_updates(0);

        assert_eq!(updates[0].target, DoryReducePolynomial::NextC(0));
        assert_eq!(
            updates[0].terms,
            vec![
                DoryReduceTerm::new(DoryReduceScalarTerm::One, DoryReducePolynomial::CurrentC(0)),
                DoryReduceTerm::new(DoryReduceScalarTerm::One, DoryReducePolynomial::SetupChi(0)),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::Beta,
                    DoryReducePolynomial::CurrentD2(0)
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::BetaInverse,
                    DoryReducePolynomial::CurrentD1(0),
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::Alpha,
                    DoryReducePolynomial::MessageCPlus(0),
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::AlphaInverse,
                    DoryReducePolynomial::MessageCMinus(0),
                ),
            ]
        );
        assert_eq!(updates[1].target, DoryReducePolynomial::NextD1(0));
        assert_eq!(updates[2].target, DoryReducePolynomial::NextD2(0));
    }

    #[test]
    fn g1_and_g2_transition_terms_use_the_expected_challenge_powers() {
        assert_eq!(
            g1_transition_update(0).terms,
            vec![
                DoryReduceTerm::new(DoryReduceScalarTerm::One, DoryReducePolynomial::CurrentE1X),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::Beta,
                    DoryReducePolynomial::MessageE1BetaX,
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::Alpha,
                    DoryReducePolynomial::MessageE1PlusX,
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::AlphaInverse,
                    DoryReducePolynomial::MessageE1MinusX,
                ),
            ]
        );
        assert_eq!(
            g2_transition_update(0).terms,
            vec![
                DoryReduceTerm::new(DoryReduceScalarTerm::One, DoryReducePolynomial::CurrentE2X0),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::BetaInverse,
                    DoryReducePolynomial::MessageE2BetaX0,
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::Alpha,
                    DoryReducePolynomial::MessageE2PlusX0,
                ),
                DoryReduceTerm::new(
                    DoryReduceScalarTerm::AlphaInverse,
                    DoryReducePolynomial::MessageE2MinusX0,
                ),
            ]
        );
    }

    #[test]
    fn scalar_fold_terms_use_verifier_derived_fr_fold_factors() {
        let updates = scalar_fold_updates();

        assert_eq!(
            updates[0],
            DoryReduceUpdate::new(
                DoryReducePolynomial::S1NextAccumulator,
                [DoryReduceTerm::new(
                    DoryReduceScalarTerm::S1FoldFactor,
                    DoryReducePolynomial::S1Accumulator,
                )],
            )
        );
        assert_eq!(
            updates[1],
            DoryReduceUpdate::new(
                DoryReducePolynomial::S2NextAccumulator,
                [DoryReduceTerm::new(
                    DoryReduceScalarTerm::S2FoldFactor,
                    DoryReducePolynomial::S2Accumulator,
                )],
            )
        );
    }

    #[test]
    fn proof_artifact_copy_constraints_bind_all_reduce_messages() {
        let constraints = proof_artifact_copy_constraints(0);

        assert_eq!(constraints.len(), artifacts::REDUCE_ROUND_ARTIFACT_COORDS);
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Gt,
            DoryAssistValueRef::public(
                DoryAssistPublicId::DoryProofArtifact(artifacts::reduce_first_d1_left_start(0)),
                0,
            ),
            reduce_ref(
                DoryAssistRelationId::DoryReduceGtTransition,
                DoryReducePolynomial::MessageD1Left(0),
                0,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::G1,
            DoryAssistValueRef::public(
                DoryAssistPublicId::DoryProofArtifact(
                    artifacts::reduce_second_e1_minus_start(0) + 1
                ),
                1,
            ),
            reduce_ref(
                DoryAssistRelationId::DoryReduceG1Transition,
                DoryReducePolynomial::MessageE1MinusY,
                1,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::G2,
            DoryAssistValueRef::public(
                DoryAssistPublicId::DoryProofArtifact(
                    artifacts::reduce_second_e2_minus_start(0) + 4
                ),
                4,
            ),
            reduce_ref(
                DoryAssistRelationId::DoryReduceG2Transition,
                DoryReducePolynomial::MessageE2MinusInfinity,
                4,
            ),
        )));
    }

    #[test]
    fn setup_artifact_copy_constraints_bind_round_constants() {
        let constraints = setup_artifact_copy_constraints(2, 1);

        assert_eq!(constraints.len(), 5 * artifacts::GT_ARTIFACT_COEFFS);
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Gt,
            DoryAssistValueRef::public(
                DoryAssistPublicId::VerifierSetupArtifact(
                    setup_artifacts::dory_setup_delta_2r_start(2, 1),
                ),
                0,
            ),
            reduce_ref(
                DoryAssistRelationId::DoryReduceGtTransition,
                DoryReducePolynomial::SetupDelta2R(0),
                0,
            ),
        )));
    }

    #[test]
    fn round_setup_artifact_copy_constraints_use_dory_remaining_round_index() {
        assert_eq!(verifier_setup_round(2, 0), 2);
        assert_eq!(verifier_setup_round(2, 1), 1);

        let constraints = round_setup_artifact_copy_constraints(2, 0);

        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Gt,
            DoryAssistValueRef::public(
                DoryAssistPublicId::VerifierSetupArtifact(
                    setup_artifacts::dory_setup_chi_start(2,)
                ),
                0,
            ),
            reduce_ref(
                DoryAssistRelationId::DoryReduceGtTransition,
                DoryReducePolynomial::SetupChi(0),
                0,
            ),
        )));
    }

    #[test]
    fn initial_state_copy_constraints_bind_checked_reducer_inputs() {
        let constraints = initial_state_copy_constraints();

        assert_eq!(
            constraints.len(),
            3 * artifacts::GT_ARTIFACT_COEFFS + artifacts::G1_ARTIFACT_COORDS + 2
        );
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Gt,
            DoryAssistValueRef::public(
                DoryAssistPublicId::DoryProofArtifact(artifacts::DORY_VMV_C_START),
                0,
            ),
            reduce_ref(
                DoryAssistRelationId::DoryReduceGtTransition,
                DoryReducePolynomial::CurrentC(0),
                0,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Gt,
            DoryAssistValueRef::public(DoryAssistPublicId::JoltCommitment(1), 0),
            reduce_ref(
                DoryAssistRelationId::DoryReduceGtTransition,
                DoryReducePolynomial::CurrentD1(0),
                0,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::G1,
            DoryAssistValueRef::public(
                DoryAssistPublicId::DoryProofArtifact(artifacts::DORY_VMV_E1_START + 2),
                2,
            ),
            reduce_ref(
                DoryAssistRelationId::DoryReduceG1Transition,
                DoryReducePolynomial::CurrentE1Infinity,
                2,
            ),
        )));
        assert!(constraints.contains(&copy_constraint(
            DoryAssistValueType::Scalar,
            DoryAssistValueRef::Constant(1),
            reduce_ref(
                DoryAssistRelationId::DoryReduceScalarFold,
                DoryReducePolynomial::S1Accumulator,
                0,
            ),
        )));
    }

    #[test]
    fn state_chaining_copy_constraints_skip_singleton_reduce() {
        assert!(state_chaining_copy_constraints(1).is_empty());
    }

    #[test]
    fn state_chaining_copy_constraints_connect_next_state_to_following_current_state() {
        let constraints = state_chaining_copy_constraints(2);

        assert_eq!(
            constraints.len(),
            3 * artifacts::GT_ARTIFACT_COEFFS
                + artifacts::G1_ARTIFACT_COORDS
                + artifacts::G2_ARTIFACT_COORDS
                + 2
        );
        assert!(constraints.contains(&state_chaining_constraint(
            DoryAssistValueType::Gt,
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryReducePolynomial::NextC(0),
            DoryReducePolynomial::CurrentC(0),
            0,
        )));
        assert!(constraints.contains(&state_chaining_constraint(
            DoryAssistValueType::Gt,
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryReducePolynomial::NextD2(15),
            DoryReducePolynomial::CurrentD2(15),
            15,
        )));
        assert!(constraints.contains(&state_chaining_constraint(
            DoryAssistValueType::G1,
            DoryAssistRelationId::DoryReduceG1Transition,
            DoryReducePolynomial::NextE1Infinity,
            DoryReducePolynomial::CurrentE1Infinity,
            2,
        )));
        assert!(constraints.contains(&state_chaining_constraint(
            DoryAssistValueType::G2,
            DoryAssistRelationId::DoryReduceG2Transition,
            DoryReducePolynomial::NextE2Infinity,
            DoryReducePolynomial::CurrentE2Infinity,
            4,
        )));
        assert!(constraints.contains(&state_chaining_constraint(
            DoryAssistValueType::Scalar,
            DoryAssistRelationId::DoryReduceScalarFold,
            DoryReducePolynomial::S2NextAccumulator,
            DoryReducePolynomial::S2Accumulator,
            0,
        )));
    }

    #[test]
    fn proof_artifact_fold_constraints_collect_sources_by_reduce_round() {
        let constraints = proof_artifact_fold_constraints(2);
        let target = reduce_ref(
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryReducePolynomial::MessageD1Left(0),
            0,
        );
        let constraint = constraints
            .iter()
            .find(|constraint| constraint.target == target)
            .expect("fold constraint exists");

        assert_eq!(constraint.value_type, DoryAssistValueType::Gt);
        assert_eq!(
            constraint.sources,
            vec![
                DoryAssistPublicId::DoryProofArtifact(artifacts::reduce_first_d1_left_start(0)),
                DoryAssistPublicId::DoryProofArtifact(artifacts::reduce_first_d1_left_start(1)),
            ]
        );
    }

    #[test]
    fn setup_and_transcript_fold_constraints_track_dory_round_order() {
        let setup_constraints = round_setup_artifact_fold_constraints(2);
        let setup_target = reduce_ref(
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryReducePolynomial::SetupChi(0),
            0,
        );
        let setup_constraint = setup_constraints
            .iter()
            .find(|constraint| constraint.target == setup_target)
            .expect("setup fold constraint exists");

        assert_eq!(
            setup_constraint.sources,
            vec![
                DoryAssistPublicId::VerifierSetupArtifact(setup_artifacts::dory_setup_chi_start(2)),
                DoryAssistPublicId::VerifierSetupArtifact(setup_artifacts::dory_setup_chi_start(1)),
            ]
        );

        let scalar_constraints = transition_transcript_scalar_fold_constraints(3, 2);
        let scalar_target = reduce_ref(
            DoryAssistRelationId::DoryReduceG2Transition,
            DoryReducePolynomial::BetaInverse,
            0,
        );
        let scalar_constraint = scalar_constraints
            .iter()
            .find(|constraint| constraint.target == scalar_target)
            .expect("transcript scalar fold constraint exists");

        assert_eq!(
            scalar_constraint.sources,
            vec![
                DoryAssistPublicId::TranscriptScalar(transcript_scalars::dory_reduce_beta_inverse(
                    3, 0,
                )),
                DoryAssistPublicId::TranscriptScalar(transcript_scalars::dory_reduce_beta_inverse(
                    3, 1,
                )),
            ]
        );
    }

    #[test]
    fn transcript_scalar_copy_constraints_bind_derived_fr_scalars_to_each_reduce_relation() {
        let constraints = transcript_scalar_copy_constraints(2, 0);

        assert_eq!(
            constraints.len(),
            4 * transcript_scalars::DORY_REDUCE_ROUND_TRANSCRIPT_SCALARS
        );
        assert!(constraints.contains(&scalar_copy_constraint(
            transcript_scalars::dory_reduce_alpha_inverse_beta_inverse(2, 0),
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryReducePolynomial::AlphaInverseBetaInverse,
        )));
        assert!(constraints.contains(&scalar_copy_constraint(
            transcript_scalars::dory_reduce_s2_fold_factor(2, 0),
            DoryAssistRelationId::DoryReduceScalarFold,
            DoryReducePolynomial::S2FoldFactor,
        )));
    }

    #[test]
    fn scalar_fold_transcript_scalar_copy_constraints_only_bind_fold_factors() {
        let constraints = scalar_fold_transcript_scalar_copy_constraints(2, 0);

        assert_eq!(constraints.len(), 2);
        assert_eq!(
            constraints,
            vec![
                scalar_copy_constraint(
                    transcript_scalars::dory_reduce_s1_fold_factor(2, 0),
                    DoryAssistRelationId::DoryReduceScalarFold,
                    DoryReducePolynomial::S1FoldFactor,
                ),
                scalar_copy_constraint(
                    transcript_scalars::dory_reduce_s2_fold_factor(2, 0),
                    DoryAssistRelationId::DoryReduceScalarFold,
                    DoryReducePolynomial::S2FoldFactor,
                ),
            ]
        );
    }

    #[test]
    fn transition_transcript_scalar_copy_constraints_bind_only_transition_scalars() {
        let constraints = transition_transcript_scalar_copy_constraints(2, 0);

        assert_eq!(constraints.len(), 12);
        assert!(constraints.contains(&scalar_copy_constraint(
            transcript_scalars::dory_reduce_beta(2, 0),
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryReducePolynomial::Beta,
        )));
        assert!(constraints.contains(&scalar_copy_constraint(
            transcript_scalars::dory_reduce_alpha_inverse_beta_inverse(2, 0),
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryReducePolynomial::AlphaInverseBetaInverse,
        )));
        assert!(constraints.contains(&scalar_copy_constraint(
            transcript_scalars::dory_reduce_beta(2, 0),
            DoryAssistRelationId::DoryReduceG1Transition,
            DoryReducePolynomial::Beta,
        )));
        assert!(constraints.contains(&scalar_copy_constraint(
            transcript_scalars::dory_reduce_beta_inverse(2, 0),
            DoryAssistRelationId::DoryReduceG2Transition,
            DoryReducePolynomial::BetaInverse,
        )));
        assert!(!constraints.contains(&scalar_copy_constraint(
            transcript_scalars::dory_reduce_alpha_beta(2, 0),
            DoryAssistRelationId::DoryReduceG1Transition,
            DoryReducePolynomial::AlphaBeta,
        )));
        assert!(!constraints.contains(&scalar_copy_constraint(
            transcript_scalars::dory_reduce_s1_fold_factor(2, 0),
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryReducePolynomial::S1FoldFactor,
        )));
    }
}
