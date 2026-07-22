//! Prover-side protocol construction: the [`BlindFoldProtocol`] together with
//! the statement and baked sources it was built from, and the witness
//! assembly that turns recorder-retained sumcheck secrets into the row matrix
//! [`crate::prove`] consumes.

use jolt_field::Field;
use jolt_r1cs::{ClaimSourceTable, R1csBuilder};
use jolt_sumcheck::{CommittedSumcheckWitness, SumcheckDomain, VerifiedCommittedRound};
use rand_core::RngCore;

use crate::r1cs::{insert_output_claim_sources, StageLayout};
use crate::{BlindFoldProtocol, BlindFoldStage, BlindFoldStatement, ProverError};

/// A built protocol plus the statement and baked public/challenge sources it
/// came from. Verification needs only the protocol; the prover additionally
/// needs the statement to re-run the layout/constraint pass in assignment
/// mode (see [`Self::assign_witness`]).
#[derive(Clone, Debug)]
pub struct BlindFoldConstruction<F: Field, O, Com, P = (), Ch = usize> {
    pub protocol: BlindFoldProtocol<F, Com>,
    pub statement: BlindFoldStatement<F, O, Com, P, Ch>,
    pub publics: Vec<(P, F)>,
    pub challenges: Vec<(Ch, F)>,
}

/// The full BlindFold witness matrix and its per-row Pedersen blinds, in the
/// protocol's row order: coefficient rows, output-claim rows, auxiliary rows,
/// zero padding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssignedBlindFoldWitness<F> {
    pub rows: Vec<Vec<F>>,
    pub blindings: Vec<F>,
}

impl<F, O, Com, P, Ch> BlindFoldConstruction<F, O, Com, P, Ch>
where
    F: Field,
    O: Clone + PartialEq,
    P: Clone + PartialEq,
    Ch: Clone + PartialEq,
{
    /// Assemble the witness rows and blinds for [`crate::prove`] from the
    /// per-stage committed sumcheck witnesses (in protocol stage order) and
    /// the final-opening evaluation scalars/blinds (in final-opening order).
    ///
    /// The claim chain is derived, not supplied: each stage's input claim is
    /// the domain round-sum of its first committed round, and every chained
    /// claim is the round polynomial evaluated at the Fiat-Shamir challenge
    /// already baked into the statement's consistency. Auxiliary values
    /// (products inside the claim expressions) materialize by running the
    /// constraint pass *after* assignment: `R1csBuilder::multiply` evaluates
    /// each product eagerly from its now-known operands, and the emitted
    /// matrices are position-identical to the protocol's regardless of
    /// assignment state.
    pub fn assign_witness<R: RngCore>(
        &self,
        stage_witnesses: &[&CommittedSumcheckWitness<F>],
        eval_outputs: &[F],
        eval_blindings: &[F],
        rng: &mut R,
    ) -> Result<AssignedBlindFoldWitness<F>, ProverError<F>> {
        let statement = &self.statement;
        if stage_witnesses.len() != statement.stages.len() {
            return Err(ProverError::LengthMismatch {
                name: "stage witnesses",
                expected: statement.stages.len(),
                actual: stage_witnesses.len(),
            });
        }
        for (name, values) in [
            ("final opening evaluations", eval_outputs),
            ("final opening blindings", eval_blindings),
        ] {
            if values.len() != statement.final_openings.len() {
                return Err(ProverError::LengthMismatch {
                    name,
                    expected: statement.final_openings.len(),
                    actual: values.len(),
                });
            }
        }

        let mut builder = R1csBuilder::<F>::new();
        let layout = statement
            .allocate_layout(&mut builder)
            .map_err(crate::Error::from)?;
        debug_assert_eq!(
            layout, self.protocol.layout,
            "assignment layout must match the protocol's"
        );

        for (stage_index, ((stage, stage_layout), witness)) in statement
            .stages
            .iter()
            .zip(&layout.stages)
            .zip(stage_witnesses)
            .enumerate()
        {
            assign_stage(&mut builder, stage_index, stage, stage_layout, witness)?;
        }

        for (binding_layout, (&evaluation, &blinding)) in layout
            .final_openings
            .iter()
            .zip(eval_outputs.iter().zip(eval_blindings))
        {
            if let Some(variable) = binding_layout.evaluation {
                builder.assign(variable, evaluation)?;
            }
            if let Some(variable) = binding_layout.blinding {
                builder.assign(variable, blinding)?;
            }
        }

        let mut claim_sources = ClaimSourceTable::<F, O, P, Ch>::new();
        insert_output_claim_sources(statement, &layout, &mut claim_sources)?;
        for (id, value) in &self.publics {
            claim_sources.insert_public(id.clone(), *value);
        }
        for (id, value) in &self.challenges {
            claim_sources.insert_challenge(id.clone(), *value);
        }
        statement.append(&mut builder, &layout, &mut claim_sources)?;

        let witness = builder.witness()?;
        debug_assert!(
            self.protocol.r1cs.check_witness(&witness).is_ok(),
            "assigned BlindFold witness must satisfy the verifier R1CS"
        );

        Ok(self.slice_rows(witness, stage_witnesses, rng))
    }

    /// Slice the flat R1CS witness into the protocol's row grid and pair each
    /// row with its blind: retained round blinds, retained output-claim
    /// blinds, fresh blinds for auxiliary rows, zero for padding (padding
    /// rows are all-zero, so their Pedersen commitment is the identity).
    fn slice_rows<R: RngCore>(
        &self,
        witness: Vec<F>,
        stage_witnesses: &[&CommittedSumcheckWitness<F>],
        rng: &mut R,
    ) -> AssignedBlindFoldWitness<F> {
        let dimensions = &self.protocol.dimensions;
        let row_len = dimensions.witness.row_len;

        let mut rows = witness[1..=dimensions.coefficient_values]
            .chunks(row_len)
            .map(<[F]>::to_vec)
            .collect::<Vec<_>>();

        for row in stage_witnesses
            .iter()
            .flat_map(|witness| witness.output_claim_rows.iter())
        {
            let mut row = row.clone();
            row.resize(row_len, F::zero());
            rows.push(row);
        }

        let output_claim_values = dimensions.output_claim_rows * row_len;
        let auxiliary_values = &witness[1 + dimensions.coefficient_values + output_claim_values..];
        rows.extend(auxiliary_values.chunks(row_len).map(|chunk| {
            let mut row = chunk.to_vec();
            row.resize(row_len, F::zero());
            row
        }));
        rows.resize(
            dimensions.witness_rows.auxiliary.end,
            vec![F::zero(); row_len],
        );
        rows.resize(dimensions.witness.row_count, vec![F::zero(); row_len]);

        let mut blindings = stage_witnesses
            .iter()
            .flat_map(|witness| witness.round_blindings.iter().copied())
            .collect::<Vec<_>>();
        blindings.extend(
            stage_witnesses
                .iter()
                .flat_map(|witness| witness.output_claim_blindings.iter().copied()),
        );
        blindings.extend((0..dimensions.auxiliary_rows).map(|_| F::random(&mut *rng)));
        blindings.resize(dimensions.witness.row_count, F::zero());

        AssignedBlindFoldWitness { rows, blindings }
    }
}

fn assign_stage<F, O, Com, P, Ch>(
    builder: &mut R1csBuilder<F>,
    stage_index: usize,
    stage: &BlindFoldStage<F, O, Com, P, Ch>,
    stage_layout: &StageLayout,
    witness: &CommittedSumcheckWitness<F>,
) -> Result<(), ProverError<F>>
where
    F: Field,
{
    let rounds = &stage.consistency.rounds;
    if witness.round_coefficients.len() != rounds.len() {
        return Err(ProverError::StageWitnessShape {
            stage_index,
            name: "round count",
            expected: rounds.len(),
            actual: witness.round_coefficients.len(),
        });
    }
    let Some(first_round) = witness.round_coefficients.first() else {
        return Err(ProverError::DegenerateSumcheck {
            name: "committed stage witness",
        });
    };

    let mut claim = round_sum(stage, first_round, &rounds[0], stage_index)?;
    builder.assign(stage_layout.sumcheck.input_claim, claim)?;

    for (round_index, ((coefficients, round), round_layout)) in witness
        .round_coefficients
        .iter()
        .zip(rounds)
        .zip(&stage_layout.sumcheck.rounds)
        .enumerate()
    {
        if coefficients.len() != round.degree + 1 {
            return Err(ProverError::StageWitnessShape {
                stage_index,
                name: "round coefficient count",
                expected: round.degree + 1,
                actual: coefficients.len(),
            });
        }
        for (&variable, &coefficient) in round_layout.coefficients.iter().zip(coefficients) {
            builder.assign(variable, coefficient)?;
        }
        // The chain: claim_out = s(r) becomes the next round's claim_in. The
        // round-sum consistency of the *next* round against this value is a
        // constraint, not an assignment — an inconsistent witness fails the
        // satisfaction check, never silently reassigns.
        claim = evaluate_at(coefficients, rounds[round_index].challenge);
        builder.assign(round_layout.claim_out, claim)?;
    }

    let expected_values: usize = stage.output_claim_rows.opening_ids.len();
    let actual_values: usize = witness.output_claim_rows.iter().map(|row| row.len()).sum();
    if actual_values != expected_values {
        return Err(ProverError::StageWitnessShape {
            stage_index,
            name: "output claim values",
            expected: expected_values,
            actual: actual_values,
        });
    }
    let row_len = stage.output_claim_rows.row_len;
    let variables = stage_layout
        .output_claim_rows
        .iter()
        .flat_map(|row| row.variables.iter().take(row_len));
    let values = witness
        .output_claim_rows
        .iter()
        .flat_map(|row| row.iter().copied());
    for (&variable, value) in variables.zip(values).take(expected_values) {
        builder.assign(variable, value)?;
    }

    Ok(())
}

fn round_sum<F, O, Com, P, Ch>(
    stage: &BlindFoldStage<F, O, Com, P, Ch>,
    coefficients: &[F],
    round: &VerifiedCommittedRound<F, Com>,
    stage_index: usize,
) -> Result<F, ProverError<F>>
where
    F: Field,
{
    let weights: Vec<F> = stage.domain.round_sum_coefficients(round.degree)?;
    if weights.len() != coefficients.len() {
        return Err(ProverError::StageWitnessShape {
            stage_index,
            name: "round sum coefficient count",
            expected: weights.len(),
            actual: coefficients.len(),
        });
    }
    Ok(coefficients
        .iter()
        .zip(&weights)
        .map(|(&coefficient, &weight)| coefficient * weight)
        .sum())
}

fn evaluate_at<F: Field>(coefficients: &[F], point: F) -> F {
    coefficients
        .iter()
        .rev()
        .fold(F::zero(), |acc, &coefficient| acc * point + coefficient)
}
