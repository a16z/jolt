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
    JoltAdviceKind, JoltDerivedId, JoltOpeningId, JoltRelationId, RamValCheckPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::{block_selector_mle_msb, LtPolynomial};
use jolt_transcript::{LabelWithCount, Transcript};

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage2::{Stage2BatchOutputClaims, Stage2BatchOutputPoints};
use crate::verifier::CheckedInputs;
use crate::VerifierError;

use super::outputs::Stage4OutputClaims;

/// Wire the consumed opening *values* from stage 2's RAM read-write `val` and
/// output-check `val_final`, plus the reconstructed init contributions (the
/// same advice / program-image openings the init evaluation is decomposed
/// into). Only these values feed the input claim; clear-only because the values
/// come from proof claims.
pub fn ram_val_check_input_values_from_upstream<F: Field>(
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
pub fn ram_val_check_input_points_from_upstream<F: Field>(
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

pub struct RamValCheck<F: Field> {
    symbolic: RamValCheckSymbolic,
    trace_dimensions: TraceDimensions,
    ram_log_k: usize,
    /// `Val_init(r_address)`'s public portion — resolves the `InitEval` input public.
    public_eval: F,
    /// The negated block selector for each present `Val_init` contribution —
    /// resolves the `InitSelector`/`InitSelectorProgramImage` input publics.
    init_selectors: Vec<(RamValCheckPublic, F)>,
    /// The present `Val_init` contribution openings (advice / program image):
    /// staged on the stage-4 wire but consumed by this relation's *input* `Expr`
    /// (the init-eval decomposition) and the stage-6/7 reductions, so they extend
    /// [`wire_output_openings`](ConcreteSumcheck::wire_output_openings) beyond the
    /// output-`Expr` set.
    contribution_openings: Vec<JoltOpeningId>,
}

impl<F: Field> RamValCheck<F> {
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
    ) -> Self {
        let public_eval = init.public_eval;
        let init_selectors = init
            .contributions
            .iter()
            .map(|contribution| (contribution.selector, contribution.neg_selector))
            .collect();
        let contribution_openings = init
            .contributions
            .iter()
            .map(|contribution| contribution.opening)
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
            contribution_openings,
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

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn wire_output_openings(&self) -> std::collections::BTreeSet<JoltOpeningId> {
        // Wire openings beyond the output-`Expr` set (`ram_ra`/`ram_inc`): the
        // present staged `Val_init` contribution openings (advice /
        // program-image), consumed by this relation's input `Expr` and the
        // stage-6/7 reductions rather than its own output fold.
        let mut openings = self.symbolic().expected_output_openings::<F>();
        openings.extend(self.contribution_openings.iter().copied());
        openings
    }

    /// Reproduces the stage-4 inline RAM value-check gamma draw: the
    /// `b"ram_val_check_gamma"` domain separator (an empty labeled append) followed
    /// by `ram_val_check_gamma = challenge_scalar()`. The separator's empty append
    /// is part of the soundness-critical byte stream, so it is replayed here too.
    fn draw_challenges<T: Transcript<Challenge = F>>(
        &self,
        transcript: &mut T,
    ) -> Result<RamValCheckChallenges<F>, VerifierError> {
        append_ram_val_check_gamma_domain_separator(transcript);
        Ok(RamValCheckChallenges {
            gamma: transcript.challenge_scalar(),
        })
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        input_points: &RamValCheckInputClaims<Vec<F>>,
    ) -> Result<RamValCheckOutputClaims<Vec<F>>, VerifierError> {
        let log_t = self.trace_dimensions.log_t();
        let expected_len = self.ram_log_k + log_t;
        let ram_read_write_point = input_points.ram_val();
        if ram_read_write_point.len() != expected_len {
            return Err(public_input_failed(format!(
                "RAM read-write opening point has {} variables, expected {expected_len}",
                ram_read_write_point.len()
            )));
        }
        let r_address = &ram_read_write_point[..self.ram_log_k];
        let cycle = self
            .trace_dimensions
            .cycle_opening_point(sumcheck_point)
            .map_err(public_input_failed)?;
        let opening_point = [r_address, cycle.as_slice()].concat();
        // The advice / program-image points sit at the staged RAM address sub-point,
        // not the batch sumcheck point; downstream reads them from
        // `RamValCheckInitialEvaluation` (clear) or BlindFold's own init decomposition
        // (ZK), so they are left absent here.
        Ok(RamValCheckOutputClaims {
            untrusted_advice: None,
            trusted_advice: None,
            program_image: None,
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
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        match public_id {
            // The `Val_init` decomposition publics are input publics: the public
            // initial-RAM evaluation and the negated committed-contribution selectors.
            RamValCheckPublic::InitEval => Ok(self.public_eval),
            RamValCheckPublic::InitSelector(_) | RamValCheckPublic::InitSelectorProgramImage => {
                self.init_selectors
                    .iter()
                    .find_map(|(selector, value)| (selector == public_id).then_some(*value))
                    .ok_or(VerifierError::MissingStageClaimDerived { id: *id })
            }
            // Output public — resolved in `derive_output_term`, never in the input expr.
            RamValCheckPublic::LtCyclePlusGamma => {
                Err(VerifierError::MissingStageClaimDerived { id: *id })
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
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        match public_id {
            // LtCyclePlusGamma folds the batching gamma into the `Lt` evaluation of
            // the produced cycle point against the fixed read-write cycle. Gamma comes
            // from the drawn `challenges` (the value `draw_challenges` produced).
            RamValCheckPublic::LtCyclePlusGamma => {
                let output_cycle = &output_points.ram_ra()[self.ram_log_k..];
                let fixed_cycle = &input_points.ram_val()[self.ram_log_k..];
                Ok(LtPolynomial::evaluate(output_cycle, fixed_cycle) + challenges.gamma)
            }
            // Input publics — resolved in `derive_input_term`, never in the output expr.
            RamValCheckPublic::InitEval
            | RamValCheckPublic::InitSelector(_)
            | RamValCheckPublic::InitSelectorProgramImage => {
                Err(VerifierError::MissingStageClaimDerived { id: *id })
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
/// [`ram_val_check_initial_evaluation`].
///
/// [`decomposition`]: Self::decomposition
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamValCheckInitStructure<F: Field> {
    pub public_eval: F,
    /// The staged program-image contribution's opening point (committed program
    /// mode only): the full RAM address point.
    pub program_image_point: Option<Vec<F>>,
    /// Each present advice contribution's block geometry, in canonical
    /// (untrusted, then trusted) order.
    pub advice_blocks: Vec<(JoltAdviceKind, RamValCheckAdviceBlock<F>)>,
}

impl<F: Field> RamValCheckInitStructure<F> {
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
/// and opening points come from [`ram_val_check_advice_block`], the same
/// computation the prover uses.
pub fn ram_val_check_init_structure<F: Field>(
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
/// its staged opening). Built by [`ram_val_check_initial_evaluation`] from the
/// [`RamValCheckInitStructure`] and the proof's claimed opening values; consumed by
/// the stage-4 input wiring and the downstream stage-6/7 address-phase reductions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamValCheckInitialEvaluation<F: Field> {
    pub public_eval: F,
    /// The staged program-image contribution's opening point (the full RAM address
    /// point) and value; committed-program mode only.
    pub program_image_contribution: Option<(Vec<F>, F)>,
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedRamValCheckAdviceContribution<F: Field> {
    pub kind: JoltAdviceKind,
    pub selector: F,
    /// The advice block opening *point* (the address sub-point it was evaluated
    /// at) that, with `opening_value`, this contribution weights by `selector`.
    pub opening_point: Vec<F>,
    /// The advice block opening *value* this contribution weights by `selector`.
    pub opening_value: F,
}

/// Attach the proof's staged advice / program-image opening *values* to the
/// pre-branch [`RamValCheckInitStructure`], validating that each claim is present
/// exactly when its contribution is. Clear-only (the values come from proof
/// claims); mirrors the prover's own init reconstruction so both decompose
/// `Val_init` identically.
pub(crate) fn ram_val_check_initial_evaluation<F: Field>(
    structure: &RamValCheckInitStructure<F>,
    claims: &Stage4OutputClaims<F>,
) -> Result<RamValCheckInitialEvaluation<F>, VerifierError> {
    let ram = &claims.ram_val_check;
    let program_image_opening = program_image::ram_val_check_contribution_opening();
    let program_image_contribution = match (&structure.program_image_point, ram.program_image) {
        (None, Some(_)) => {
            return Err(VerifierError::UnexpectedOpeningClaim {
                id: program_image_opening,
            });
        }
        (None, None) => None,
        (Some(_), None) => {
            return Err(VerifierError::MissingOpeningClaim {
                id: program_image_opening,
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
                return Err(VerifierError::UnexpectedOpeningClaim { id: opening });
            }
            (None, None) => {}
            (Some(_), None) => {
                return Err(VerifierError::MissingOpeningClaim { id: opening });
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
pub struct RamValCheckAdviceBlock<F: Field> {
    pub selector: F,
    pub opening_point: Vec<F>,
}

/// Compute the [`RamValCheckAdviceBlock`] for `kind` against `r_address`, using the
/// advice block's start/size from the memory layout.
///
/// WARNING: the ZK path recomputes the same geometry in `zk::blindfold`'s
/// `advice_selector`, so the two must stay in lockstep.
fn ram_val_check_advice_block<F: Field>(
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

/// Absorb the Fiat-Shamir domain separator for the RAM value-check gamma: an empty
/// message labeled `b"ram_val_check_gamma"`. The prover appends this empty labeled
/// chunk before sampling the gamma, so [`RamValCheck::draw_challenges`] must
/// reproduce it byte-for-byte (label chunk + empty payload) or every challenge from
/// here on diverges.
fn append_ram_val_check_gamma_domain_separator<T: Transcript>(transcript: &mut T) {
    transcript.append(&LabelWithCount(b"ram_val_check_gamma", 0));
    transcript.append_bytes(&[]);
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stages::relations::draw_recording::{record, DrawEvent};
    use jolt_field::Fr;

    // Overrides the default to prepend the `b"ram_val_check_gamma"` domain separator
    // (an empty labeled append) before the gamma squeeze; the append is part of the
    // soundness-critical byte stream, so it must appear in `draw_challenges` too.
    #[test]
    fn draw_challenges_appends_domain_separator_then_draws_gamma() {
        let relation = RamValCheck::<Fr>::new(
            TraceDimensions::new(4),
            3,
            RamValCheckInit::from(Fr::from(0u64)),
        );

        // Inline (stage4/verify.rs L125-126): domain-separator append, then
        // `ram_val_check_gamma = challenge_scalar()`.
        let (inline_events, inline_gamma) = record(|t| {
            append_ram_val_check_gamma_domain_separator(t);
            t.challenge_scalar()
        });
        let (draw_events, challenges) = record(|t| relation.draw_challenges(t).unwrap());

        assert_eq!(draw_events, inline_events);
        // The draw is the domain-separator append(s) followed by exactly one squeeze;
        // no challenge is squeezed before the gamma.
        assert!(draw_events.len() >= 2);
        let (separator, last) = draw_events.split_at(draw_events.len() - 1);
        assert_eq!(last, [DrawEvent::Squeeze(1)]);
        assert!(separator
            .iter()
            .all(|event| matches!(event, DrawEvent::Append(_))));
        assert_eq!(challenges.gamma, inline_gamma);
    }

    #[test]
    fn ram_val_check_gamma_domain_separator_matches_core_empty_bytes_append() {
        let (events, ()) = record(append_ram_val_check_gamma_domain_separator);

        let mut packed = vec![0; 32];
        packed[..b"ram_val_check_gamma".len()].copy_from_slice(b"ram_val_check_gamma");
        assert_eq!(
            events,
            [DrawEvent::Append(packed), DrawEvent::Append(Vec::new())]
        );
    }
}
