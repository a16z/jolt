//! The stage 1 `SpartanOuter` remainder sumcheck instance.
//!
//! A self-contained relation object driven by the verifier after checking the
//! Spartan outer remainder sumcheck. It owns the remainder opening-point derivation
//! and resolves the expanded quadratic R1CS form's `SpartanOuterPublic` coefficients
//! from [`JoltSpartanOuterRemainder::public_coefficients`] — the same source the
//! BlindFold constraint uses — so the output-claim algebra lives here once and stays
//! in lockstep with that constraint, which evaluates the same expanded
//! `spartan::outer_remainder` formula.
//!
//! The expanded `output_expression` (`Σ q[l,r]·o[l]·o[r] + Σ ℓ[i]·o[i] + c`) is the
//! distributed form of the factored quadratic
//! `tau_kernel · (Σ az[i]·o[i] + az_c) · (Σ bz[i]·o[i] + bz_c)` that
//! [`JoltSpartanOuterRemainder::expected_output_claim`] computes; the two are
//! value-equivalent (see `output_matches_factored_form`).
//!
//! The companion Spartan outer *uni-skip* first round is a univariate skip rather
//! than a batched-Boolean [`ConcreteSumcheck`] verification, so it stays hand-coded
//! in the stage-1 verifier; this relation consumes that uni-skip's reduced opening
//! as its input claim.

use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
pub use jolt_claims::protocols::jolt::relations::spartan::{
    OuterRemainderInputClaims, OuterRemainderOutputClaims,
};
use jolt_claims::protocols::jolt::{relations, JoltDerivedId, JoltRelationId, SpartanOuterPublic};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_r1cs::constraints::jolt::{
    JoltSpartanOuterPublic, JoltSpartanOuterRemainder, JoltSpartanOuterRemainderChallenges,
};

use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

/// Wire the consumed opening *value* from the Spartan outer uni-skip's reduced output
/// claim: only the value feeds the input claim (the output point comes from this
/// relation's own sumcheck point).
pub fn outer_remainder_input_values_from_uniskip_output<F: Field>(
    uniskip_output_claim: F,
) -> OuterRemainderInputClaims<F> {
    OuterRemainderInputClaims {
        outer_uniskip: uniskip_output_claim,
    }
}

/// Wire the consumed opening *point* from the Spartan outer uni-skip. The remainder
/// reads only the uni-skip's value, so the input point is left empty.
pub fn outer_remainder_input_points_from_uniskip_output<F: Field>(
) -> OuterRemainderInputClaims<Vec<F>> {
    OuterRemainderInputClaims {
        outer_uniskip: Vec::new(),
    }
}

/// The expanded quadratic-form coefficients, indexed for O(1) resolution. Built
/// once from [`JoltSpartanOuterRemainder::public_coefficients`] in the constructor
/// so `derive_output_term` (called ~`n² + n + 1` times per proof) never rebuilds the
/// `JoltSpartanOuterRemainder` matrix work.
struct OuterRemainderCoefficients<F> {
    variable_count: usize,
    /// Row-major `quadratic[left * variable_count + right]`.
    quadratic: Vec<F>,
    linear: Vec<F>,
    constant: F,
}

impl<F: Field> OuterRemainderCoefficients<F> {
    fn from_public_coefficients(
        variable_count: usize,
        coefficients: Vec<(JoltSpartanOuterPublic, F)>,
    ) -> Self {
        let mut quadratic = vec![F::zero(); variable_count * variable_count];
        let mut linear = vec![F::zero(); variable_count];
        let mut constant = F::zero();
        for (id, value) in coefficients {
            match id {
                JoltSpartanOuterPublic::QuadraticCoefficient { left, right } => {
                    quadratic[left * variable_count + right] = value;
                }
                JoltSpartanOuterPublic::LinearCoefficient(index) => {
                    linear[index] = value;
                }
                JoltSpartanOuterPublic::ConstantCoefficient => {
                    constant = value;
                }
            }
        }
        Self {
            variable_count,
            quadratic,
            linear,
            constant,
        }
    }

    fn resolve(&self, id: SpartanOuterPublic) -> Option<F> {
        match id {
            SpartanOuterPublic::QuadraticCoefficient { left, right } => self
                .quadratic
                .get(left * self.variable_count + right)
                .filter(|_| left < self.variable_count && right < self.variable_count)
                .copied(),
            SpartanOuterPublic::LinearCoefficient(index) => self.linear.get(index).copied(),
            SpartanOuterPublic::ConstantCoefficient => Some(self.constant),
        }
    }
}

pub struct OuterRemainder<F: Field> {
    symbolic: relations::spartan::OuterRemainder,
    coefficients: OuterRemainderCoefficients<F>,
}

impl<F: Field> OuterRemainder<F> {
    pub fn new(
        dimensions: SpartanOuterDimensions,
        tau: &[F],
        uniskip_challenge: F,
        remainder_challenges: &[F],
    ) -> Result<Self, VerifierError> {
        let variable_count = dimensions.variables().len();
        let formula = JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
            tau,
            uniskip: uniskip_challenge,
            remainder: remainder_challenges,
        })
        .map_err(public_input_failed)?;
        let coefficients = OuterRemainderCoefficients::from_public_coefficients(
            variable_count,
            formula.public_coefficients(),
        );
        Ok(Self {
            symbolic: relations::spartan::OuterRemainder::new(dimensions),
            coefficients,
        })
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanOuter,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for OuterRemainder<F> {
    type Symbolic = relations::spartan::OuterRemainder;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &OuterRemainderInputClaims<Vec<F>>,
    ) -> Result<OuterRemainderOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(OuterRemainderOutputClaims {
            left_instruction_input: opening_point.clone(),
            right_instruction_input: opening_point.clone(),
            product: opening_point.clone(),
            should_branch: opening_point.clone(),
            pc: opening_point.clone(),
            unexpanded_pc: opening_point.clone(),
            imm: opening_point.clone(),
            ram_address: opening_point.clone(),
            rs1_value: opening_point.clone(),
            rs2_value: opening_point.clone(),
            rd_write_value: opening_point.clone(),
            ram_read_value: opening_point.clone(),
            ram_write_value: opening_point.clone(),
            left_lookup_operand: opening_point.clone(),
            right_lookup_operand: opening_point.clone(),
            next_unexpanded_pc: opening_point.clone(),
            next_pc: opening_point.clone(),
            next_is_virtual: opening_point.clone(),
            next_is_first_in_sequence: opening_point.clone(),
            lookup_output: opening_point.clone(),
            should_jump: opening_point.clone(),
            add_operands: opening_point.clone(),
            subtract_operands: opening_point.clone(),
            multiply_operands: opening_point.clone(),
            load: opening_point.clone(),
            store: opening_point.clone(),
            jump: opening_point.clone(),
            write_lookup_output_to_rd: opening_point.clone(),
            virtual_instruction: opening_point.clone(),
            assert: opening_point.clone(),
            do_not_update_unexpanded_pc: opening_point.clone(),
            advice: opening_point.clone(),
            is_compressed: opening_point.clone(),
            is_first_in_sequence: opening_point.clone(),
            is_last_in_sequence: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &OuterRemainderInputClaims<Vec<F>>,
        _output_points: &OuterRemainderOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::SpartanOuter(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        self.coefficients
            .resolve(*public_id)
            .ok_or(VerifierError::MissingStageClaimDerived { id: *id })
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stages::relations::OutputClaims;
    use crate::stages::stage1::outputs::outer_remainder_outputs_from_r1cs_inputs;
    use jolt_claims::protocols::jolt::geometry::spartan::SPARTAN_OUTER_R1CS_INPUTS;
    use jolt_field::{Fr, FromPrimitiveInt};

    /// The assembled `OuterRemainderOutputValues` field (declaration) order is the
    /// canonical `SPARTAN_OUTER_R1CS_INPUTS` order, so `opening_values()` (which the
    /// generated `append_to_transcript` iterates) reproduces the input order. Equal
    /// value sequences ⇒ byte-identical Fiat-Shamir appends.
    #[test]
    fn append_order_matches_r1cs_input_order() {
        // Distinct values, one per R1CS input, in canonical order.
        let values = SPARTAN_OUTER_R1CS_INPUTS
            .iter()
            .copied()
            .enumerate()
            .map(|(index, variable)| (variable, Fr::from_u64(1_000 + index as u64)))
            .collect::<Vec<_>>();
        let expected = values.iter().map(|(_, value)| *value).collect::<Vec<_>>();
        let relation_form = outer_remainder_outputs_from_r1cs_inputs(values).unwrap();

        assert_eq!(relation_form.opening_values(), expected);
    }

    /// Fill all 35 produced opening *values* with the given values (in canonical
    /// field / `SPARTAN_OUTER_R1CS_INPUTS` order).
    fn output_values_from(values: &[Fr]) -> OuterRemainderOutputClaims<Fr> {
        let mut iter = values.iter().copied();
        let mut next = || iter.next().unwrap();
        OuterRemainderOutputClaims {
            left_instruction_input: next(),
            right_instruction_input: next(),
            product: next(),
            should_branch: next(),
            pc: next(),
            unexpanded_pc: next(),
            imm: next(),
            ram_address: next(),
            rs1_value: next(),
            rs2_value: next(),
            rd_write_value: next(),
            ram_read_value: next(),
            ram_write_value: next(),
            left_lookup_operand: next(),
            right_lookup_operand: next(),
            next_unexpanded_pc: next(),
            next_pc: next(),
            next_is_virtual: next(),
            next_is_first_in_sequence: next(),
            lookup_output: next(),
            should_jump: next(),
            add_operands: next(),
            subtract_operands: next(),
            multiply_operands: next(),
            load: next(),
            store: next(),
            jump: next(),
            write_lookup_output_to_rd: next(),
            virtual_instruction: next(),
            assert: next(),
            do_not_update_unexpanded_pc: next(),
            advice: next(),
            is_compressed: next(),
            is_first_in_sequence: next(),
            is_last_in_sequence: next(),
        }
    }

    /// All 35 produced opening *points* sharing a single opening point.
    fn output_points_at(point: &[Fr]) -> OuterRemainderOutputClaims<Vec<Fr>> {
        let next = || point.to_vec();
        OuterRemainderOutputClaims {
            left_instruction_input: next(),
            right_instruction_input: next(),
            product: next(),
            should_branch: next(),
            pc: next(),
            unexpanded_pc: next(),
            imm: next(),
            ram_address: next(),
            rs1_value: next(),
            rs2_value: next(),
            rd_write_value: next(),
            ram_read_value: next(),
            ram_write_value: next(),
            left_lookup_operand: next(),
            right_lookup_operand: next(),
            next_unexpanded_pc: next(),
            next_pc: next(),
            next_is_virtual: next(),
            next_is_first_in_sequence: next(),
            lookup_output: next(),
            should_jump: next(),
            add_operands: next(),
            subtract_operands: next(),
            multiply_operands: next(),
            load: next(),
            store: next(),
            jump: next(),
            write_lookup_output_to_rd: next(),
            virtual_instruction: next(),
            assert: next(),
            do_not_update_unexpanded_pc: next(),
            advice: next(),
            is_compressed: next(),
            is_first_in_sequence: next(),
            is_last_in_sequence: next(),
        }
    }

    /// The relation's expanded `expected_output` evaluates bit-identically to the
    /// factored `JoltSpartanOuterRemainder::expected_output_claim` on the production
    /// 35-variable rv64 shape. This is the equivalence the clear stage-1 path now
    /// relies on (it switched from the factored matrix form to the expanded relation
    /// form); muldiv non-ZK is the end-to-end gate, this pins it at unit level.
    #[test]
    fn expected_output_matches_factored_form_on_rv64_shape() {
        let log_t = 3usize;
        let dimensions = SpartanOuterDimensions::rv64(log_t);
        let variable_count = dimensions.variables().len();
        assert_eq!(variable_count, 35);

        // `tau` has `log_t + 2` entries; the remainder challenge vector has
        // `1 + log_t` entries (so `tau.len() == remainder.len() + 1`).
        let tau_len = log_t + 2;
        let remainder_len = 1 + log_t;
        let tau = (0..tau_len)
            .map(|i| Fr::from_u64(2 + i as u64))
            .collect::<Vec<_>>();
        let remainder_challenges = (0..remainder_len)
            .map(|i| Fr::from_u64(100 + i as u64))
            .collect::<Vec<_>>();
        let uniskip_challenge = Fr::from_u64(17);

        let factored = JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
            tau: &tau,
            uniskip: uniskip_challenge,
            remainder: &remainder_challenges,
        })
        .unwrap();
        let openings = (0..variable_count)
            .map(|i| Fr::from_u64(1_000 + i as u64))
            .collect::<Vec<_>>();
        let factored_output = factored.expected_output_claim(&openings).unwrap();

        let relation =
            OuterRemainder::new(dimensions, &tau, uniskip_challenge, &remainder_challenges)
                .unwrap();
        let point = vec![Fr::from_u64(7); 1 + log_t];
        let output_values = output_values_from(&openings);
        let output_points = output_points_at(&point);
        let input_points = outer_remainder_input_points_from_uniskip_output::<Fr>();
        let expanded_output = relation
            .expected_output(
                &input_points,
                &output_values,
                &output_points,
                &NoChallenges::default(),
            )
            .unwrap();

        assert_eq!(expanded_output, factored_output);
    }
}
