//! The stage 4 `RamValCheck` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover and the
//! verifier. It owns the RAM value-check point derivation and the
//! `LtCyclePlusGamma` public-value computation; the decomposition of
//! `Val_init(r_address)` into a public evaluation plus committed
//! advice/program-image contributions lives in its `jolt-claims` formula
//! (`ram::val_check`), so the clear path, the prover, and the BlindFold
//! constraint all consume the same decomposition.
//!
//! WARNING: the advice/program-image openings are dual-role — they are *consumed*
//! by the input claim (init reconstruction) and *also* appended/serialized as
//! stage-4 openings. They therefore appear both as [`RamValCheckInputClaims`]
//! fields and in the serialized `Stage4OutputClaims` aggregate. Only their values feed
//! the input claim; their staged points are carried for completeness.

use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::program_image,
        dimensions::TraceDimensions,
        ram::{self, RamValCheckInit, RamValCheckInitContribution},
    },
    relations::ram::{RamValCheck as RamValCheckSymbolic, RamValCheckShape, RamValContribution},
    JoltAdviceKind, JoltChallengeId, JoltPublicId, JoltRelationClaims, JoltRelationId,
    RamValCheckChallenge, RamValCheckPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::{block_selector_mle_msb, LtPolynomial};
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::proof::JoltProof;
use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::stage2::outputs::Stage2ClearOutput;
use crate::verifier::CheckedInputs;
use crate::VerifierError;

use super::outputs::Stage4OutputClaims;

/// Produced RAM value-check openings (`ram_ra`, `ram_inc`) sharing one opening
/// point. Generic over the cell.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamValCheck)]
pub struct RamValCheckOutputClaims<C> {
    #[opening(RamRa)]
    pub ram_ra: C,
    #[opening(committed = RamInc)]
    pub ram_inc: C,
}

/// The staged advice openings contributing to `Val_init`: untrusted/trusted
/// advice block evaluations, each present only when its commitment is. Appended
/// before the register openings (see the `Stage4OutputClaims` field order).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamValCheck)]
pub struct RamValCheckAdviceClaims<C> {
    #[opening(untrusted_advice)]
    pub untrusted: Option<C>,
    #[opening(trusted_advice)]
    pub trusted: Option<C>,
}

/// Consumed openings of the RAM value-check claim: the read-write `val` (stage 2)
/// and output-check `val_final` (stage 2), reduced against `Val_init`, whose
/// committed pieces (advice / program image) are present only in some proof
/// configurations. Generic over the cell.
#[derive(Clone, Debug, InputClaims)]
pub struct RamValCheckInputClaims<C> {
    #[opening(RamVal, from = RamReadWriteChecking)]
    pub ram_val: C,
    #[opening(RamValFinal, from = RamOutputCheck)]
    pub ram_val_final: C,
    #[opening(untrusted_advice, from = RamValCheck)]
    pub untrusted_advice: Option<C>,
    #[opening(trusted_advice, from = RamValCheck)]
    pub trusted_advice: Option<C>,
    #[opening(ProgramImageInitContributionRw, from = RamValCheck)]
    pub program_image: Option<C>,
}

impl<F: Field> RamValCheckInputClaims<OpeningClaim<F>> {
    /// Wire the consumed openings from stage 2's RAM read-write `val` and
    /// output-check `val_final`, plus the reconstructed init contributions (the
    /// same advice / program-image openings the init evaluation is decomposed
    /// into). The init pieces carry their staged opening points for completeness,
    /// though only their values feed the input claim.
    pub fn from_upstream(
        stage2: &Stage2ClearOutput<F>,
        init: &RamValCheckInitialEvaluation<F>,
    ) -> Self {
        let advice =
            |kind: JoltAdviceKind| init.advice_contribution(kind).map(|c| c.opening.clone());
        Self {
            ram_val: OpeningClaim {
                point: stage2.output_claims.ram_read_write_point().to_vec(),
                value: stage2.output_claims.ram_read_write.val.value,
            },
            ram_val_final: OpeningClaim {
                point: stage2.output_claims.ram_output_check_point().to_vec(),
                value: stage2.output_claims.ram_output_check.val_final.value,
            },
            untrusted_advice: advice(JoltAdviceKind::Untrusted),
            trusted_advice: advice(JoltAdviceKind::Trusted),
            program_image: init.program_image_contribution.clone(),
        }
    }
}

pub struct RamValCheck<F: Field> {
    symbolic: RamValCheckSymbolic,
    claims: JoltRelationClaims<F>,
    trace_dimensions: TraceDimensions,
    ram_log_k: usize,
    gamma: F,
    /// `Val_init(r_address)`'s public portion — resolves the `InitEval` input public.
    public_eval: F,
    /// The negated block selector for each present `Val_init` contribution —
    /// resolves the `InitSelector`/`InitSelectorProgramImage` input publics.
    init_selectors: Vec<(RamValCheckPublic, F)>,
}

impl<F: Field> RamValCheck<F> {
    /// Build the relation from its per-proof init decomposition. `init` carries
    /// the public initial-RAM evaluation plus the present advice/program-image
    /// contributions; their *structure* feeds the symbolic input `Expr` and their
    /// *values* are supplied as `Public` symbols via [`resolve_public`].
    ///
    /// [`resolve_public`]: ConcreteSumcheck::resolve_public
    pub fn new(
        trace_dimensions: TraceDimensions,
        ram_log_k: usize,
        gamma: F,
        init: RamValCheckInit<F>,
    ) -> Self {
        let public_eval = init.public_eval;
        let init_selectors = init
            .contributions
            .iter()
            .map(|contribution| (contribution.selector, contribution.neg_selector))
            .collect();
        let symbolic = RamValCheckSymbolic::new(RamValCheckShape {
            dimensions: trace_dimensions,
            contributions: init
                .contributions
                .iter()
                .map(|contribution| RamValContribution {
                    selector: contribution.selector,
                    opening: contribution.opening,
                })
                .collect(),
        });
        Self {
            claims: ram::val_check(trace_dimensions, init),
            symbolic,
            trace_dimensions,
            ram_log_k,
            gamma,
            public_eval,
            init_selectors,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamValCheck,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for RamValCheck<F> {
    type Symbolic = RamValCheckSymbolic;
    type Inputs<C> = RamValCheckInputClaims<C>;
    type Outputs<C> = RamValCheckOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        inputs: &RamValCheckInputClaims<C>,
    ) -> Result<RamValCheckOutputClaims<Vec<F>>, VerifierError> {
        let log_t = self.trace_dimensions.log_t();
        let expected_len = self.ram_log_k + log_t;
        let ram_read_write_point = inputs.ram_val.point();
        if ram_read_write_point.len() != expected_len {
            return Err(public_input_failed(format!(
                "RAM read-write opening point has {} variables, expected {expected_len}",
                ram_read_write_point.len()
            )));
        }
        let (r_address, r_cycle) = ram_read_write_point.split_at(self.ram_log_k);
        let cycle = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        if cycle.len() != r_cycle.len() {
            return Err(public_input_failed(format!(
                "RAM value cycle point length mismatch: expected {}, got {}",
                r_cycle.len(),
                cycle.len()
            )));
        }
        let opening_point = [r_address, cycle.as_slice()].concat();
        Ok(RamValCheckOutputClaims {
            ram_ra: opening_point.clone(),
            ram_inc: opening_point,
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::RamValCheck(RamValCheckChallenge::Gamma) => Ok(self.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        inputs: &RamValCheckInputClaims<C>,
        outputs: Option<&RamValCheckOutputClaims<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        let JoltPublicId::RamValCheck(public_id) = id else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        match public_id {
            // The `Val_init` decomposition publics are input publics (resolved
            // with `outputs == None`): the public initial-RAM evaluation and the
            // negated committed-contribution selectors.
            RamValCheckPublic::InitEval => Ok(self.public_eval),
            RamValCheckPublic::InitSelector(_) | RamValCheckPublic::InitSelectorProgramImage => {
                self.init_selectors
                    .iter()
                    .find_map(|(selector, value)| (selector == public_id).then_some(*value))
                    .ok_or(VerifierError::MissingStageClaimPublic { id: *id })
            }
            // LtCyclePlusGamma folds the batching gamma into the `Lt` evaluation
            // of the produced cycle point against the fixed read-write cycle.
            RamValCheckPublic::LtCyclePlusGamma => {
                let outputs = outputs.ok_or(VerifierError::MissingStageClaimPublic { id: *id })?;
                let output_cycle = &outputs.ram_ra.point()[self.ram_log_k..];
                let fixed_cycle = &inputs.ram_val.point()[self.ram_log_k..];
                Ok(LtPolynomial::evaluate(output_cycle, fixed_cycle) + self.gamma)
            }
        }
    }
}

/// The verifier's reconstruction of `Val_init(r_address)`: the public initial-RAM
/// evaluation plus the present advice / program-image contributions (each carrying
/// its staged opening). Built by [`ram_val_check_initial_evaluation`] and consumed
/// when constructing the [`RamValCheck`] relation (via [`decomposition`]) and the
/// stage-4 output claims.
///
/// [`decomposition`]: Self::decomposition
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamValCheckInitialEvaluation<F: Field> {
    pub public_eval: F,
    /// The staged program-image contribution to `Val_init(r_address)` (committed
    /// program mode only): the opening claim with the full RAM address point.
    pub program_image_contribution: Option<OpeningClaim<F>>,
    pub advice_contributions: Vec<VerifiedRamValCheckAdviceContribution<F>>,
}

impl<F: Field> RamValCheckInitialEvaluation<F> {
    pub fn advice_contribution(
        &self,
        kind: JoltAdviceKind,
    ) -> Option<&VerifiedRamValCheckAdviceContribution<F>> {
        self.advice_contributions
            .iter()
            .find(|contribution| contribution.kind == kind)
    }

    /// The formula-side init decomposition: the public initial-RAM evaluation plus
    /// the present advice / program-image contributions (with negated selectors),
    /// in the canonical order the BlindFold constraint also uses — program image
    /// first, then advice in `advice_contributions` order. Shared by the verifier
    /// and the prover when building the `RamValCheck` relation, so the
    /// decomposition cannot drift between them.
    pub fn decomposition(&self) -> RamValCheckInit<F> {
        let mut contributions = Vec::new();
        if self.program_image_contribution.is_some() {
            contributions.push(RamValCheckInitContribution::program_image(-F::one()));
        }
        for contribution in &self.advice_contributions {
            let neg_selector = -contribution.selector;
            contributions.push(match contribution.kind {
                JoltAdviceKind::Trusted => RamValCheckInitContribution::trusted(neg_selector),
                JoltAdviceKind::Untrusted => RamValCheckInitContribution::untrusted(neg_selector),
            });
        }
        RamValCheckInit::decomposed(self.public_eval, contributions)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedRamValCheckAdviceContribution<F: Field> {
    pub kind: JoltAdviceKind,
    pub selector: F,
    /// The advice block opening (claim value + the address sub-point it was
    /// evaluated at) that this contribution weights by `selector`.
    pub opening: OpeningClaim<F>,
}

/// Reconstruct [`RamValCheckInitialEvaluation`] from the proof's staged advice /
/// program-image openings: record each present contribution's staged opening
/// alongside the public initial-RAM `public_eval`. Mirrors the prover's own init
/// reconstruction so both decompose `Val_init` identically.
pub(crate) fn ram_val_check_initial_evaluation<PCS, VC, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    claims: &Stage4OutputClaims<PCS::Field>,
    r_address: &[PCS::Field],
    public_eval: PCS::Field,
) -> Result<RamValCheckInitialEvaluation<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let program_image_contribution = collect_program_image_contribution(
        checked.precommitted.program_image.is_some(),
        claims.program_image_contribution,
        r_address,
    )?;
    let mut advice_contributions = Vec::new();
    let untrusted_present = proof.untrusted_advice_commitment.is_some();
    collect_advice_contribution(
        JoltAdviceKind::Untrusted,
        untrusted_present,
        claims.advice.untrusted,
        checked,
        r_address,
        &mut advice_contributions,
    )?;
    collect_advice_contribution(
        JoltAdviceKind::Trusted,
        checked.trusted_advice_commitment_present,
        claims.advice.trusted,
        checked,
        r_address,
        &mut advice_contributions,
    )?;

    Ok(RamValCheckInitialEvaluation {
        public_eval,
        program_image_contribution,
        advice_contributions,
    })
}

fn collect_program_image_contribution<F: Field>(
    committed_program: bool,
    opening_claim: Option<F>,
    r_address: &[F],
) -> Result<Option<OpeningClaim<F>>, VerifierError> {
    let opening = program_image::ram_val_check_contribution_opening();
    if !committed_program {
        if opening_claim.is_some() {
            return Err(VerifierError::UnexpectedOpeningClaim { id: opening });
        }
        return Ok(None);
    }

    let opening_claim = opening_claim.ok_or(VerifierError::MissingOpeningClaim { id: opening })?;
    Ok(Some(OpeningClaim {
        point: r_address.to_vec(),
        value: opening_claim,
    }))
}

/// The advice block's selector and opening point, derived from the memory layout
/// and the RAM address point. The prover (which produces the advice opening from
/// the witness) and the verifier (which checks the claimed opening) must compute
/// this geometry identically, so it is single-sourced here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamValCheckAdviceBlock<F: Field> {
    pub selector: F,
    pub opening_point: Vec<F>,
}

/// Compute the [`RamValCheckAdviceBlock`] for `kind` against `r_address`, using the
/// advice block's start/size from the memory layout. Shared by the verifier and the
/// prover so the advice selector and opening point cannot drift between them.
pub fn ram_val_check_advice_block<F: Field>(
    kind: JoltAdviceKind,
    checked: &CheckedInputs,
    r_address: &[F],
) -> Result<RamValCheckAdviceBlock<F>, VerifierError> {
    let layout = &checked.public_io.memory_layout;
    let (start_address, max_size) = match kind {
        JoltAdviceKind::Trusted => (layout.trusted_advice_start, layout.max_trusted_advice_size),
        JoltAdviceKind::Untrusted => (
            layout.untrusted_advice_start,
            layout.max_untrusted_advice_size,
        ),
    };
    if max_size == 0 {
        return Err(public_input_failed(format!(
            "{kind:?} advice commitment is present but configured size is zero"
        )));
    }
    let start_index = layout
        .remapped_word_address(start_address)
        .map_err(public_input_failed)? as u128;
    let advice_num_vars = ((max_size as usize) / 8).next_power_of_two().ilog2() as usize;
    if advice_num_vars > r_address.len() {
        return Err(public_input_failed(format!(
            "{kind:?} advice point needs {advice_num_vars} variables but RAM address has {}",
            r_address.len()
        )));
    }
    let selector = block_selector_mle_msb(start_index, advice_num_vars, r_address)
        .map_err(public_input_failed)?;
    let opening_point = r_address[r_address.len() - advice_num_vars..].to_vec();
    Ok(RamValCheckAdviceBlock {
        selector,
        opening_point,
    })
}

fn collect_advice_contribution<F: Field>(
    kind: JoltAdviceKind,
    present: bool,
    opening_claim: Option<F>,
    checked: &CheckedInputs,
    r_address: &[F],
    contributions: &mut Vec<VerifiedRamValCheckAdviceContribution<F>>,
) -> Result<(), VerifierError> {
    let opening = ram::val_check_advice_opening(kind);
    if !present {
        if opening_claim.is_some() {
            return Err(VerifierError::UnexpectedOpeningClaim { id: opening });
        }
        return Ok(());
    }

    let value = opening_claim.ok_or(VerifierError::MissingOpeningClaim { id: opening })?;
    let block = ram_val_check_advice_block(kind, checked, r_address)?;
    contributions.push(VerifiedRamValCheckAdviceContribution {
        kind,
        selector: block.selector,
        opening: OpeningClaim {
            point: block.opening_point,
            value,
        },
    });
    Ok(())
}
