//! Prover-side witness assembly against a built [`BlindFoldProtocol`]: turn
//! recorder-retained sumcheck secrets into the row matrix [`crate::prove`]
//! consumes, using only the protocol's public parts (layout, matrices,
//! consistency, dimensions).
//!
//! No statement or claim expressions are needed. Layout variables (round
//! coefficients, the claim chain, output-claim rows, final-opening scalars)
//! are assigned directly; the claim chain is *derived* — each stage's input
//! claim is the domain round-sum of its first committed round, each chained
//! claim the round polynomial evaluated at the Fiat-Shamir challenge already
//! carried in the consistency. Every remaining private value is a product
//! auxiliary from the claim-expression lowering, whose constraint has the
//! canonical `A · B = 1·v_fresh` shape (`R1csBuilder::multiply`), so one
//! forward pass over the constraint matrices solves them in emission order.
//! Unconstrained slots are the layout's zero padding.

use jolt_field::Field;
use jolt_r1cs::SparseRow;
use jolt_sumcheck::{CommittedSumcheckWitness, SumcheckDomain, SumcheckDomainSpec};
use rand_core::RngCore;

use crate::r1cs::StageLayout;
use crate::{BlindFoldProtocol, ProverError};

/// The full BlindFold witness matrix and its per-row Pedersen blinds, in the
/// protocol's row order: coefficient rows, output-claim rows, auxiliary rows,
/// zero padding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssignedBlindFoldWitness<F> {
    pub rows: Vec<Vec<F>>,
    pub blindings: Vec<F>,
}

impl<F: Field, Com> BlindFoldProtocol<F, Com> {
    /// Assemble the witness rows and blinds for [`crate::prove`] from the
    /// per-stage committed sumcheck witnesses and domains (in protocol stage
    /// order) and the final-opening evaluation scalars/blinds (in
    /// final-opening order). The domains are the caller's protocol
    /// constants — the same ones its sumchecks ran over.
    pub fn assign_witness<R: RngCore>(
        &self,
        stage_domains: &[SumcheckDomainSpec],
        stage_witnesses: &[&CommittedSumcheckWitness<F>],
        eval_outputs: &[F],
        eval_blindings: &[F],
        rng: &mut R,
    ) -> Result<AssignedBlindFoldWitness<F>, ProverError<F>> {
        for (name, actual) in [
            ("stage witnesses", stage_witnesses.len()),
            ("stage domains", stage_domains.len()),
        ] {
            if actual != self.layout.stages.len() {
                return Err(ProverError::LengthMismatch {
                    name,
                    expected: self.layout.stages.len(),
                    actual,
                });
            }
        }
        for (name, values) in [
            ("final opening evaluations", eval_outputs),
            ("final opening blindings", eval_blindings),
        ] {
            if values.len() != self.layout.final_openings.len() {
                return Err(ProverError::LengthMismatch {
                    name,
                    expected: self.layout.final_openings.len(),
                    actual: values.len(),
                });
            }
        }

        let mut witness: Vec<Option<F>> = vec![None; self.r1cs.num_vars];
        *witness
            .first_mut()
            .ok_or(ProverError::DegenerateSumcheck { name: "empty R1CS" })? = Some(F::one());

        for (stage_index, ((stage_layout, &domain), &stage_witness)) in self
            .layout
            .stages
            .iter()
            .zip(stage_domains)
            .zip(stage_witnesses)
            .enumerate()
        {
            self.assign_stage(
                &mut witness,
                stage_index,
                stage_layout,
                domain,
                stage_witness,
            )?;
        }

        for (binding_layout, (&evaluation, &blinding)) in self
            .layout
            .final_openings
            .iter()
            .zip(eval_outputs.iter().zip(eval_blindings))
        {
            if let Some(variable) = binding_layout.evaluation {
                assign(&mut witness, variable.index(), evaluation)?;
            }
            if let Some(variable) = binding_layout.blinding {
                assign(&mut witness, variable.index(), blinding)?;
            }
        }

        solve_products(&self.r1cs.a, &self.r1cs.b, &self.r1cs.c, &mut witness)?;

        let witness: Vec<F> = witness
            .into_iter()
            .map(|value| value.unwrap_or_else(F::zero))
            .collect();
        debug_assert!(
            self.r1cs.check_witness(&witness).is_ok(),
            "assigned BlindFold witness must satisfy the verifier R1CS"
        );

        Ok(self.slice_rows(witness, stage_witnesses, rng))
    }

    fn assign_stage(
        &self,
        witness: &mut [Option<F>],
        stage_index: usize,
        stage_layout: &StageLayout,
        domain: SumcheckDomainSpec,
        stage_witness: &CommittedSumcheckWitness<F>,
    ) -> Result<(), ProverError<F>> {
        let rounds = &stage_layout.sumcheck.rounds;
        let consistency =
            self.sumcheck_consistency
                .get(stage_index)
                .ok_or(ProverError::LengthMismatch {
                    name: "stage consistency",
                    expected: stage_index + 1,
                    actual: self.sumcheck_consistency.len(),
                })?;
        if stage_witness.round_coefficients.len() != rounds.len()
            || consistency.rounds.len() != rounds.len()
        {
            return Err(ProverError::StageWitnessShape {
                stage_index,
                name: "round count",
                expected: rounds.len(),
                actual: stage_witness.round_coefficients.len(),
            });
        }
        let Some(first_round) = stage_witness.round_coefficients.first() else {
            return Err(ProverError::DegenerateSumcheck {
                name: "committed stage witness",
            });
        };

        let weights: Vec<F> = domain.round_sum_coefficients(first_round.len().saturating_sub(1))?;
        if weights.len() != first_round.len() {
            return Err(ProverError::StageWitnessShape {
                stage_index,
                name: "round sum coefficient count",
                expected: weights.len(),
                actual: first_round.len(),
            });
        }
        let mut claim: F = first_round
            .iter()
            .zip(&weights)
            .map(|(&coefficient, &weight)| coefficient * weight)
            .sum();
        assign(witness, stage_layout.sumcheck.input_claim.index(), claim)?;

        for ((coefficients, verified), round_layout) in stage_witness
            .round_coefficients
            .iter()
            .zip(&consistency.rounds)
            .zip(rounds)
        {
            if coefficients.len() != round_layout.coefficients.len()
                || coefficients.len() != verified.degree + 1
            {
                return Err(ProverError::StageWitnessShape {
                    stage_index,
                    name: "round coefficient count",
                    expected: round_layout.coefficients.len(),
                    actual: coefficients.len(),
                });
            }
            for (variable, &coefficient) in round_layout.coefficients.iter().zip(coefficients) {
                assign(witness, variable.index(), coefficient)?;
            }
            // The chain: claim_out = s(r) becomes the next round's claim_in.
            // The next round's round-sum against this value is a constraint,
            // not an assignment — inconsistent data fails satisfaction, never
            // silently reassigns.
            claim = evaluate_at(coefficients, verified.challenge);
            assign(witness, round_layout.claim_out.index(), claim)?;
        }

        if stage_layout.output_claim_rows.len() != stage_witness.output_claim_rows.len() {
            return Err(ProverError::StageWitnessShape {
                stage_index,
                name: "output claim row count",
                expected: stage_layout.output_claim_rows.len(),
                actual: stage_witness.output_claim_rows.len(),
            });
        }
        for (row_layout, values) in stage_layout
            .output_claim_rows
            .iter()
            .zip(&stage_witness.output_claim_rows)
        {
            if values.len() > row_layout.variables.len() {
                return Err(ProverError::StageWitnessShape {
                    stage_index,
                    name: "output claim row length",
                    expected: row_layout.variables.len(),
                    actual: values.len(),
                });
            }
            for (variable, &value) in row_layout.variables.iter().zip(values) {
                assign(witness, variable.index(), value)?;
            }
        }

        Ok(())
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
        let dimensions = &self.dimensions;
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

fn assign<F: Field>(
    witness: &mut [Option<F>],
    index: usize,
    value: F,
) -> Result<(), ProverError<F>> {
    let slot = witness
        .get_mut(index)
        .ok_or(ProverError::WitnessVariableOutOfBounds { index })?;
    if slot.is_some() {
        return Err(ProverError::WitnessVariableReassigned { index });
    }
    *slot = Some(value);
    Ok(())
}

/// Solve the product auxiliaries in constraint-emission order. Every
/// `R1csBuilder::multiply` emits `A · B = 1·v` with `v` freshly allocated
/// *after* its operands, so a single forward pass assigns each in turn; the
/// equality constraints (round sums, claim bindings, final openings) have a
/// zero C side and are skipped — they are checks, not definitions.
fn solve_products<F: Field>(
    a: &[SparseRow<F>],
    b: &[SparseRow<F>],
    c: &[SparseRow<F>],
    witness: &mut [Option<F>],
) -> Result<(), ProverError<F>> {
    for (constraint, ((a_row, b_row), c_row)) in a.iter().zip(b).zip(c).enumerate() {
        let [(target, coefficient)] = c_row.as_slice() else {
            continue;
        };
        if *coefficient != F::one() || witness.get(*target).is_none_or(|slot| slot.is_some()) {
            continue;
        }
        let Some(product) = evaluate_sparse(a_row, witness)
            .zip(evaluate_sparse(b_row, witness))
            .map(|(a_value, b_value)| a_value * b_value)
        else {
            return Err(ProverError::UnsolvableProduct { constraint });
        };
        witness[*target] = Some(product);
    }
    Ok(())
}

fn evaluate_sparse<F: Field>(row: &SparseRow<F>, witness: &[Option<F>]) -> Option<F> {
    row.iter()
        .map(|&(index, coefficient)| {
            witness
                .get(index)
                .copied()
                .flatten()
                .map(|value| value * coefficient)
        })
        .sum()
}

fn evaluate_at<F: Field>(coefficients: &[F], point: F) -> F {
    coefficients
        .iter()
        .rev()
        .fold(F::zero(), |acc, &coefficient| acc * point + coefficient)
}
