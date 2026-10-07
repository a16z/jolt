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

use core::convert::Infallible;
pub use jolt_claims::protocols::jolt::relations::ram::{
    RamValCheckChallenges, RamValCheckInputClaims, RamValCheckOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::{
        claim_reductions::program_image,
        dimensions::TraceDimensions,
        ram::{self, RamValCheckInit, RamValCheckInitContribution},
    },
    relations::ram::{RamValCheck as RamValCheckSymbolic, RamValCheckShape, RamValContribution},
    JoltAdviceKind, JoltDerivedId, JoltRelationId, RamValCheckPublic,
};
use std::collections::BTreeMap;

use jolt_claims::{MapCells, OutputClaims, SymbolicSumcheck};
use jolt_field::JoltField;
use jolt_poly::{block_selector_mle_msb, LtPolynomial};
use jolt_transcript::{ProverTranscript, Sponge, VerifierTranscript};

use crate::stages::relations::{ClaimRoute, ClaimRoutes, ComposedOpeningId, ConcreteSumcheck};
use crate::stages::stage2::{Stage2BatchOutputClaims, Stage2BatchOutputPoints};
use crate::verifier::CheckedInputs;
use crate::VerifierError;

/// Wire the consumed opening *values* from stage 2's RAM read-write `val` and
/// output-check `val_final`, plus the reconstructed init contributions (the
/// same advice / program-image openings the init evaluation is decomposed
/// into). Only these values feed the input claim; clear-only because the values
/// come from proof claims.
pub fn ram_val_check_input_values_from_upstream<F: JoltField>(
    stage2: &Stage2BatchOutputClaims<F>,
    init: &RamValCheckInitialEvaluation<F>,
) -> RamValCheckInputClaims<F> {
    let advice = |kind: JoltAdviceKind| init.advice_contribution(kind).map(|c| c.opening_value);
    RamValCheckInputClaims {
        ram_val: stage2.ram_read_write.val,
        ram_val_final: stage2.ram_output_check.val_final,
        untrusted_advice: advice(JoltAdviceKind::Untrusted),
        trusted_advice: advice(JoltAdviceKind::Trusted),
        program_image: init
            .program_image_contribution
            .as_ref()
            .map(|(_, value)| *value),
    }
}

/// Wire the consumed opening *points* from stage 2's RAM read-write and
/// output-check openings, plus the init contributions' staged opening points
/// (carried for completeness though only the values feed the input claim).
/// ZK-agnostic: it reads the stage-2 point aggregate and the pre-branch init
/// structure, so the same wiring serves both paths.
pub fn ram_val_check_input_points_from_upstream<F: JoltField>(
    stage2: &Stage2BatchOutputPoints<F>,
    structure: &RamValCheckInitStructure<F>,
) -> RamValCheckInputClaims<Vec<F>> {
    let advice = |kind: JoltAdviceKind| {
        structure
            .advice_block(kind)
            .map(|block| block.opening_point.clone())
    };
    RamValCheckInputClaims {
        ram_val: stage2.ram_read_write_point().to_vec(),
        ram_val_final: stage2.ram_output_check_point().to_vec(),
        untrusted_advice: advice(JoltAdviceKind::Untrusted),
        trusted_advice: advice(JoltAdviceKind::Trusted),
        program_image: structure.program_image_point.clone(),
    }
}

#[derive(Clone)]
pub struct RamValCheck<F: JoltField> {
    symbolic: RamValCheckSymbolic,
    trace_dimensions: TraceDimensions,
    ram_log_k: usize,
    /// `Val_init(r_address)`'s public portion — resolves the `InitEval` input public.
    public_eval: F,
    /// The negated block selector for each present `Val_init` contribution —
    /// resolves the `InitSelector`/`InitSelectorProgramImage` input publics.
    init_selectors: Vec<(RamValCheckPublic, F)>,
    /// The points of the present staged `Val_init` contribution openings
    /// (advice / program image), which sit at the staged RAM address sub-point
    /// rather than the batch point.
    staged_points: RamValCheckStagedOpenings<Vec<F>>,
}

impl<F: JoltField> RamValCheck<F> {
    /// Build the relation from its per-proof init decomposition. `init` carries
    /// the public initial-RAM evaluation plus the present advice/program-image
    /// contributions; their *structure* feeds the symbolic input `Expr` and their
    /// *values* are supplied as `Derived` symbols via [`derive_input_term`].
    ///
    /// [`derive_input_term`]: ConcreteSumcheck::derive_input_term
    pub fn new(
        trace_dimensions: TraceDimensions,
        ram_log_k: usize,
        init: RamValCheckInit<F>,
        staged_points: RamValCheckStagedOpenings<Vec<F>>,
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
            symbolic,
            trace_dimensions,
            ram_log_k,
            public_eval,
            init_selectors,
            staged_points,
        }
    }

    pub fn trace_dimensions(&self) -> TraceDimensions {
        self.trace_dimensions
    }

    pub fn ram_log_k(&self) -> usize {
        self.ram_log_k
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamValCheck,
        reason: reason.to_string(),
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for RamValCheck<F> {
    type Symbolic = RamValCheckSymbolic;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        input_points: &RamValCheckInputClaims<Vec<F>>,
    ) -> Result<RamValCheckOutputClaims<Vec<F>>, VerifierError> {
        let log_t = self.trace_dimensions.log_t();
        #[expect(
            clippy::arithmetic_side_effects,
            reason = "ram_log_k and log_t are ilog2 results (< 64); the sum cannot overflow usize"
        )]
        let expected_len = self.ram_log_k + log_t;
        let ram_read_write_point = input_points.ram_val();
        if ram_read_write_point.len() != expected_len {
            return Err(public_input_failed(format!(
                "RAM read-write opening point has {} variables, expected {expected_len}",
                ram_read_write_point.len()
            )));
        }
        let r_address = ram_read_write_point.get(..self.ram_log_k).ok_or_else(|| {
            public_input_failed("RAM read-write opening point address prefix is out of range")
        })?;
        let cycle = self
            .trace_dimensions
            .cycle_opening_point(sumcheck_point)
            .map_err(public_input_failed)?;
        let opening_point = [r_address, cycle.as_slice()].concat();
        let RamValCheckStagedOpenings {
            untrusted_advice,
            trusted_advice,
            program_image,
        } = self.staged_points.clone();
        Ok(RamValCheckOutputClaims {
            untrusted_advice,
            trusted_advice,
            program_image,
            ram_ra: opening_point.clone(),
            ram_inc: opening_point,
        })
    }

    fn derive_input_term(
        &self,
        id: &JoltDerivedId,
        _challenges: &RamValCheckChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::RamValCheck(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
        };
        match public_id {
            RamValCheckPublic::InitEval => Ok(self.public_eval),
            RamValCheckPublic::InitSelector(_) | RamValCheckPublic::InitSelectorProgramImage => {
                self.init_selectors
                    .iter()
                    .find_map(|(selector, value)| (selector == public_id).then_some(*value))
                    .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() })
            }
            // Output public — resolved in `derive_output_term`, never in the input expr.
            RamValCheckPublic::LtCyclePlusGamma => {
                Err(VerifierError::MissingStageClaimDerived { id: (*id).into() })
            }
        }
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        input_points: &RamValCheckInputClaims<Vec<F>>,
        output_points: &RamValCheckOutputClaims<Vec<F>>,
        challenges: &RamValCheckChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::RamValCheck(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
        };
        match public_id {
            RamValCheckPublic::LtCyclePlusGamma => {
                let output_cycle =
                    output_points
                        .ram_ra()
                        .get(self.ram_log_k..)
                        .ok_or_else(|| {
                            public_input_failed(
                                "RAM value-check output opening point is shorter than the address \
                                 width",
                            )
                        })?;
                let fixed_cycle =
                    input_points
                        .ram_val()
                        .get(self.ram_log_k..)
                        .ok_or_else(|| {
                            public_input_failed(
                                "RAM read-write opening point is shorter than the address width",
                            )
                        })?;
                Ok(LtPolynomial::evaluate(output_cycle, fixed_cycle) + challenges.gamma)
            }
            // Input publics — resolved in `derive_input_term`, never in the output expr.
            RamValCheckPublic::InitEval
            | RamValCheckPublic::InitSelector(_)
            | RamValCheckPublic::InitSelectorProgramImage => {
                Err(VerifierError::MissingStageClaimDerived { id: (*id).into() })
            }
        }
    }
}

/// The mode-agnostic *structure* of the verifier's `Val_init(r_address)`
/// decomposition: the public evaluation plus each present contribution's staged
/// opening point and block selector. Computable in both proving modes before the
/// zk/clear branch (it reads only presence flags and layout geometry), so the
/// [`RamValCheck`] relation can be constructed once via [`decomposition`]; the
/// clear path attaches the claimed opening *values* afterwards via
/// `ram_val_check_initial_evaluation`.
///
/// [`decomposition`]: Self::decomposition
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamValCheckInitStructure<F: JoltField> {
    pub public_eval: F,
    /// The staged program-image contribution's opening point (committed program
    /// mode only): the full RAM address point.
    pub program_image_point: Option<Vec<F>>,
    /// Each present advice contribution's block geometry, in canonical
    /// (untrusted, then trusted) order.
    pub advice_blocks: Vec<(JoltAdviceKind, RamValCheckAdviceBlock<F>)>,
}

impl<F: JoltField> RamValCheckInitStructure<F> {
    /// The staged contribution openings' points: each advice block's opening
    /// point and the program image's full RAM address point.
    pub fn staged_points(&self) -> RamValCheckStagedOpenings<Vec<F>> {
        let advice = |kind| {
            self.advice_block(kind)
                .map(|block| block.opening_point.clone())
        };
        RamValCheckStagedOpenings {
            untrusted_advice: advice(JoltAdviceKind::Untrusted),
            trusted_advice: advice(JoltAdviceKind::Trusted),
            program_image: self.program_image_point.clone(),
        }
    }

    pub fn advice_block(&self, kind: JoltAdviceKind) -> Option<&RamValCheckAdviceBlock<F>> {
        self.advice_blocks
            .iter()
            .find_map(|(block_kind, block)| (*block_kind == kind).then_some(block))
    }

    /// The formula-side init decomposition fed to [`RamValCheck::new`]: the public
    /// initial-RAM evaluation plus the present contributions (with negated
    /// selectors), in the canonical order the BlindFold constraint also uses —
    /// program image first, then advice in `advice_blocks` order.
    ///
    /// WARNING: contribution order and selectors must stay in lockstep with
    /// BlindFold's `ram_val_check_init` (zk/blindfold/mod.rs) and the prover's own
    /// decomposition.
    pub fn decomposition(&self) -> RamValCheckInit<F> {
        let mut contributions = Vec::new();
        if self.program_image_point.is_some() {
            contributions.push(RamValCheckInitContribution::program_image(-F::one()));
        }
        for (kind, block) in &self.advice_blocks {
            let neg_selector = -block.selector;
            contributions.push(match kind {
                JoltAdviceKind::Trusted => RamValCheckInitContribution::trusted(neg_selector),
                JoltAdviceKind::Untrusted => RamValCheckInitContribution::untrusted(neg_selector),
            });
        }
        RamValCheckInit::decomposed(self.public_eval, contributions)
    }
}

/// Build the [`RamValCheckInitStructure`] from the presence flags and layout
/// geometry. Runs before the zk/clear branch in both modes; the advice selectors
/// and opening points come from `ram_val_check_advice_block`, the same
/// computation the prover uses.
pub fn ram_val_check_init_structure<F: JoltField>(
    checked: &CheckedInputs,
    untrusted_advice_present: bool,
    r_address: &[F],
    public_eval: F,
) -> Result<RamValCheckInitStructure<F>, VerifierError> {
    let program_image_point = checked
        .precommitted
        .program_image
        .is_some()
        .then(|| r_address.to_vec());
    let mut advice_blocks = Vec::new();
    for (kind, present) in [
        (JoltAdviceKind::Untrusted, untrusted_advice_present),
        (
            JoltAdviceKind::Trusted,
            checked.trusted_advice_commitment_present,
        ),
    ] {
        if present {
            advice_blocks.push((kind, ram_val_check_advice_block(kind, checked, r_address)?));
        }
    }
    Ok(RamValCheckInitStructure {
        public_eval,
        program_image_point,
        advice_blocks,
    })
}

/// The verifier's reconstruction of `Val_init(r_address)`: the public initial-RAM
/// evaluation plus the present advice / program-image contributions (each carrying
/// its staged opening). Built by `ram_val_check_initial_evaluation` from the
/// [`RamValCheckInitStructure`] and the proof's claimed opening values; consumed by
/// the stage-4 input wiring and the downstream stage-6/7 address-phase reductions.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct RamValCheckInitialEvaluation<F: JoltField> {
    pub public_eval: F,
    /// The staged program-image contribution's opening point (the full RAM address
    /// point) and value; committed-program mode only.
    pub program_image_contribution: Option<(Vec<F>, F)>,
    pub advice_contributions: Vec<VerifiedRamValCheckAdviceContribution<F>>,
}

impl<F: JoltField> RamValCheckInitialEvaluation<F> {
    pub fn advice_contribution(
        &self,
        kind: JoltAdviceKind,
    ) -> Option<&VerifiedRamValCheckAdviceContribution<F>> {
        self.advice_contributions
            .iter()
            .find(|contribution| contribution.kind == kind)
    }

    /// The staged opening values this evaluation decomposes `Val_init` into:
    /// the inverse of attaching [`RamValCheckStagedOpenings`] to the init
    /// structure.
    pub fn staged_openings(&self) -> RamValCheckStagedOpenings<F> {
        let advice = |kind| {
            self.advice_contribution(kind)
                .map(|contribution| contribution.opening_value)
        };
        RamValCheckStagedOpenings {
            untrusted_advice: advice(JoltAdviceKind::Untrusted),
            trusted_advice: advice(JoltAdviceKind::Trusted),
            program_image: self
                .program_image_contribution
                .as_ref()
                .map(|(_, value)| *value),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct VerifiedRamValCheckAdviceContribution<F: JoltField> {
    pub kind: JoltAdviceKind,
    pub selector: F,
    /// The advice block opening *point* (the address sub-point it was evaluated
    /// at) that, with `opening_value`, this contribution weights by `selector`.
    pub opening_point: Vec<F>,
    /// The advice block opening *value* this contribution weights by `selector`.
    pub opening_value: F,
}

/// The RAM value-check's staged `Val_init` contribution openings, each present
/// exactly when the init structure has that contribution: the stage-4 claim
/// cells a clear proof sends before the batch (the RAM value-check input claim
/// consumes them), and that a committed proof commits with the rest of the
/// stage's claims. Generic over the cell: the points come from
/// [`RamValCheckInitStructure::staged_points`]; the cells and their order are
/// the matching fields of [`RamValCheckOutputClaims`].
#[derive(Clone, Debug, Default, PartialEq, Eq, OutputClaims)]
#[relation(RamValCheck)]
pub struct RamValCheckStagedOpenings<C> {
    #[opening(untrusted_advice)]
    pub untrusted_advice: Option<C>,
    #[opening(trusted_advice)]
    pub trusted_advice: Option<C>,
    #[opening(ProgramImageInitContributionRw)]
    pub program_image: Option<C>,
}

impl<F: JoltField> RamValCheckStagedOpenings<F> {
    /// Sends the present openings, in canonical order.
    pub fn send<H: Sponge>(&self, transcript: &mut ProverTranscript<H>) {
        transcript.send_all(&self.opening_values());
    }

    /// Receives the openings in the shape of their staged `points`.
    pub fn receive<H: Sponge>(
        points: &RamValCheckStagedOpenings<Vec<F>>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<Self, VerifierError> {
        points.try_map_cells(&mut |_, _point| transcript.receive().map_err(Into::into))
    }

    /// The received openings by id, the pre-filled `Staged` cells of the
    /// stage-4 claims receive.
    pub fn by_id(&self) -> BTreeMap<ComposedOpeningId, F> {
        self.canonical_order()
            .into_iter()
            .map(Into::into)
            .zip(self.opening_values())
            .collect()
    }
}

impl<C: Clone> RamValCheckStagedOpenings<C> {
    /// The staged cells of the RAM value check's claims (or points).
    pub fn from_claims(claims: &RamValCheckOutputClaims<C>) -> Self {
        Self {
            untrusted_advice: claims.untrusted_advice.clone(),
            trusted_advice: claims.trusted_advice.clone(),
            program_image: claims.program_image.clone(),
        }
    }
}

impl<F: JoltField> RamValCheckStagedOpenings<Vec<F>> {
    /// The stage-4 claim routes: every staged cell is [`ClaimRoute::Staged`].
    pub fn claim_routes(&self) -> ClaimRoutes {
        let mut routes = ClaimRoutes::default();
        let Ok(_) = self.try_map_cells(&mut |id, _point| {
            routes.set(*id, ClaimRoute::Staged);
            Ok::<F, Infallible>(F::zero())
        });
        routes
    }
}

/// Attach the staged advice / program-image opening *values* to the pre-branch
/// [`RamValCheckInitStructure`], validating that each is present exactly when
/// its contribution is. Clear-only; mirrors the prover's own init reconstruction
/// so both decompose `Val_init` identically.
pub(crate) fn ram_val_check_initial_evaluation<F: JoltField>(
    structure: &RamValCheckInitStructure<F>,
    ram: &RamValCheckStagedOpenings<F>,
) -> Result<RamValCheckInitialEvaluation<F>, VerifierError> {
    let program_image_opening = program_image::ram_val_check_contribution_opening();
    let program_image_contribution = match (&structure.program_image_point, ram.program_image) {
        (None, Some(_)) => {
            return Err(VerifierError::UnexpectedOpeningClaim {
                id: program_image_opening.into(),
            });
        }
        (None, None) => None,
        (Some(_), None) => {
            return Err(VerifierError::MissingOpeningClaim {
                id: program_image_opening.into(),
            });
        }
        (Some(point), Some(value)) => Some((point.clone(), value)),
    };

    let mut advice_contributions = Vec::new();
    for (kind, opening_claim) in [
        (JoltAdviceKind::Untrusted, ram.untrusted_advice),
        (JoltAdviceKind::Trusted, ram.trusted_advice),
    ] {
        let opening = ram::val_check_advice_opening(kind);
        match (structure.advice_block(kind), opening_claim) {
            (None, Some(_)) => {
                return Err(VerifierError::UnexpectedOpeningClaim { id: opening.into() });
            }
            (None, None) => {}
            (Some(_), None) => {
                return Err(VerifierError::MissingOpeningClaim { id: opening.into() });
            }
            (Some(block), Some(value)) => {
                advice_contributions.push(VerifiedRamValCheckAdviceContribution {
                    kind,
                    selector: block.selector,
                    opening_point: block.opening_point.clone(),
                    opening_value: value,
                });
            }
        }
    }

    Ok(RamValCheckInitialEvaluation {
        public_eval: structure.public_eval,
        program_image_contribution,
        advice_contributions,
    })
}

/// The advice block's selector and opening point, derived from the memory layout
/// and the RAM address point.
///
/// WARNING: the ZK path recomputes the same geometry in `zk::blindfold`'s
/// `advice_selector`, so the two must stay in lockstep.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamValCheckAdviceBlock<F: JoltField> {
    pub selector: F,
    pub opening_point: Vec<F>,
}

/// Compute the [`RamValCheckAdviceBlock`] for `kind` against `r_address`, using the
/// advice block's start/size from the memory layout.
///
/// WARNING: the ZK path recomputes the same geometry in `zk::blindfold`'s
/// `advice_selector`, so the two must stay in lockstep.
fn ram_val_check_advice_block<F: JoltField>(
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
    let start_index = u128::from(
        layout
            .remapped_word_address(start_address)
            .map_err(public_input_failed)?,
    );
    let max_size = usize::try_from(max_size).map_err(|_| {
        public_input_failed(format!("{kind:?} advice size {max_size} exceeds usize"))
    })?;
    // Floor division mirrors the shared advice geometry in jolt-claims
    // (`geometry/dimensions.rs`): the advice block holds `max_size / 8` words.
    #[expect(
        clippy::integer_division,
        reason = "floor division bytes -> words matches the prover-shared advice geometry"
    )]
    let advice_num_vars = crate::num::ilog2((max_size / 8).next_power_of_two());
    let opening_start = r_address
        .len()
        .checked_sub(advice_num_vars)
        .ok_or_else(|| {
            public_input_failed(format!(
                "{kind:?} advice point needs {advice_num_vars} variables but RAM address has {}",
                r_address.len()
            ))
        })?;
    let selector = block_selector_mle_msb(start_index, advice_num_vars, r_address)
        .map_err(public_input_failed)?;
    let opening_point = r_address
        .get(opening_start..)
        .ok_or_else(|| public_input_failed("advice opening point is out of range"))?
        .to_vec();
    Ok(RamValCheckAdviceBlock {
        selector,
        opening_point,
    })
}
