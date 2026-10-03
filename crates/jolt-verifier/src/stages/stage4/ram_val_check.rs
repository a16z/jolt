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
use jolt_field::JoltField;
use jolt_poly::{block_selector_mle_msb, LtPolynomial};
use jolt_transcript::{LabelWithCount, Transcript};

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage2::{Stage2BatchOutputClaims, Stage2BatchOutputPoints};
use crate::verifier::CheckedInputs;
use crate::VerifierError;

use super::outputs::Stage4OutputClaims;

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
    public_eval: F,
    init_selectors: Vec<(RamValCheckPublic, F)>,
    contribution_openings: Vec<JoltOpeningId>,
}

impl<F: JoltField> RamValCheck<F> {
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

    fn wire_output_openings(&self) -> std::collections::BTreeSet<JoltOpeningId> {
        // Wire openings beyond the output-`Expr` set (`ram_ra`/`ram_inc`): the
        // present staged `Val_init` contribution openings (advice /
        // program-image), consumed by this relation's input `Expr` and the
        // stage-6/7 reductions rather than its own output fold.
        let mut openings = self.symbolic().expected_output_openings::<F>();
        openings.extend(self.contribution_openings.iter().copied());
        openings
    }

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
            RamValCheckPublic::InitEval
            | RamValCheckPublic::InitSelector(_)
            | RamValCheckPublic::InitSelectorProgramImage => {
                Err(VerifierError::MissingStageClaimDerived { id: (*id).into() })
            }
        }
    }
}

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
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct VerifiedRamValCheckAdviceContribution<F: JoltField> {
    pub kind: JoltAdviceKind,
    pub selector: F,
    pub opening_point: Vec<F>,
    pub opening_value: F,
}

pub(crate) fn ram_val_check_initial_evaluation<F: JoltField>(
    structure: &RamValCheckInitStructure<F>,
    claims: &Stage4OutputClaims<F>,
) -> Result<RamValCheckInitialEvaluation<F>, VerifierError> {
    let ram = &claims.ram_val_check;
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
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test code indexes its own fixed-size fixtures"
)]
mod tests {
    use super::*;
    use crate::stages::relations::draw_recording::{record, DrawEvent};
    use jolt_field::Fr;

    #[test]
    fn draw_challenges_appends_domain_separator_then_draws_gamma() {
        let relation = RamValCheck::<Fr>::new(
            TraceDimensions::new(4),
            3,
            RamValCheckInit::from(Fr::from(0u64)),
        );

        let (inline_events, inline_gamma) = record(|t| {
            append_ram_val_check_gamma_domain_separator(t);
            t.challenge_scalar()
        });
        let (draw_events, challenges) = record(|t| relation.draw_challenges(t).unwrap());

        assert_eq!(draw_events, inline_events);
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
