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

/// The factored-form constituents, indexed for O(1) resolution. Built once
/// from [`JoltSpartanOuterRemainder::public_coefficients`] in the constructor
/// so `derive_output_term` (called ~`2n + 3` times per proof) never rebuilds
/// the `JoltSpartanOuterRemainder` matrix work.
struct OuterRemainderCoefficients<F> {
    tau_kernel: F,
    az_weights: Vec<F>,
    bz_weights: Vec<F>,
    az_constant: F,
    bz_constant: F,
}

impl<F: Field> OuterRemainderCoefficients<F> {
    fn from_public_coefficients(
        variable_count: usize,
        coefficients: Vec<(JoltSpartanOuterPublic, F)>,
    ) -> Self {
        let mut tau_kernel = F::zero();
        let mut az_weights = vec![F::zero(); variable_count];
        let mut bz_weights = vec![F::zero(); variable_count];
        let mut az_constant = F::zero();
        let mut bz_constant = F::zero();
        for (id, value) in coefficients {
            match id {
                JoltSpartanOuterPublic::TauKernel => tau_kernel = value,
                JoltSpartanOuterPublic::AzWeight(index) => az_weights[index] = value,
                JoltSpartanOuterPublic::BzWeight(index) => bz_weights[index] = value,
                JoltSpartanOuterPublic::AzConstant => az_constant = value,
                JoltSpartanOuterPublic::BzConstant => bz_constant = value,
            }
        }
        Self {
            tau_kernel,
            az_weights,
            bz_weights,
            az_constant,
            bz_constant,
        }
    }

    fn resolve(&self, id: SpartanOuterPublic) -> Option<F> {
        match id {
            SpartanOuterPublic::TauKernel => Some(self.tau_kernel),
            SpartanOuterPublic::AzWeight(index) => self.az_weights.get(index).copied(),
            SpartanOuterPublic::BzWeight(index) => self.bz_weights.get(index).copied(),
            SpartanOuterPublic::AzConstant => Some(self.az_constant),
            SpartanOuterPublic::BzConstant => Some(self.bz_constant),
        }
    }
}

impl<F: Field> OuterRemainder<F> {
    pub fn tau(&self) -> &[F] {
        &self.tau
    }

    pub fn uniskip_challenge(&self) -> F {
        self.uniskip_challenge
    }
}

pub struct OuterRemainder<F: Field> {
    symbolic: relations::spartan::OuterRemainder,
    variable_count: usize,
    /// The stage-1 `tau` draw and the uni-skip reduction challenge — two of the
    /// three inputs to the `SpartanOuterPublic` coefficient table. Both exist
    /// before this relation is constructed (the uni-skip step completes first).
    tau: Vec<F>,
    uniskip_challenge: F,
    /// The relation's own bound point (the remainder challenges), captured by
    /// [`derive_opening_points`](ConcreteSumcheck::derive_opening_points) — the
    /// third coefficient-table input, which exists only after the batch reduces.
    bound_point: std::sync::OnceLock<Vec<F>>,
    /// The expanded coefficient table, built lazily on the first
    /// `derive_output_term` call so the ZK path (which never evaluates the output
    /// expression) skips the `JoltSpartanOuterRemainder` matrix work entirely.
    coefficients: std::sync::OnceLock<OuterRemainderCoefficients<F>>,
}

impl<F: Field> OuterRemainder<F> {
    pub fn new(dimensions: SpartanOuterDimensions, tau: Vec<F>, uniskip_challenge: F) -> Self {
        let variable_count = dimensions.variables().len();
        Self {
            symbolic: relations::spartan::OuterRemainder::new(dimensions),
            variable_count,
            tau,
            uniskip_challenge,
            bound_point: std::sync::OnceLock::new(),
            coefficients: std::sync::OnceLock::new(),
        }
    }

    /// The expanded `SpartanOuterPublic` coefficient table, built on first use from
    /// `tau`, the uni-skip reduction challenge, and the captured bound point.
    /// Sourced from [`JoltSpartanOuterRemainder::public_coefficients`] — the same
    /// source the BlindFold constraint uses — so the output-claim algebra cannot
    /// drift from that constraint.
    fn coefficients(&self) -> Result<&OuterRemainderCoefficients<F>, VerifierError> {
        if self.coefficients.get().is_none() {
            let bound_point = self.bound_point.get().ok_or_else(|| {
                public_input_failed(
                    "Spartan outer remainder point not bound (derive_opening_points must run \
                     before the output expression is evaluated)",
                )
            })?;
            let formula = JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
                tau: &self.tau,
                uniskip: self.uniskip_challenge,
                remainder: bound_point,
            })
            .map_err(public_input_failed)?;
            let _ = self
                .coefficients
                .set(OuterRemainderCoefficients::from_public_coefficients(
                    self.variable_count,
                    formula.public_coefficients(),
                ));
        }
        self.coefficients
            .get()
            .ok_or_else(|| public_input_failed("Spartan outer remainder coefficients unavailable"))
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
        // Capture the bound point for the lazy coefficient-table build; reject a
        // rebind at a different point (one bind per verification).
        let bound_point = self
            .bound_point
            .get_or_init(|| sumcheck_point.to_vec())
            .as_slice();
        if bound_point != sumcheck_point {
            return Err(public_input_failed(
                "Spartan outer remainder point already bound at a different point",
            ));
        }
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
        self.coefficients()?
            .resolve(*public_id)
            .ok_or(VerifierError::MissingStageClaimDerived { id: *id })
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stages::relations::OutputClaims;
    use jolt_claims::protocols::jolt::geometry::spartan::SPARTAN_OUTER_R1CS_INPUTS;
    use jolt_claims::protocols::jolt::JoltOpeningId;
    use jolt_field::{Fr, FromPrimitiveInt};

    /// The produced `OuterRemainderOutputClaims` field (declaration) order is the
    /// canonical `SPARTAN_OUTER_R1CS_INPUTS` order, so the generated absorb
    /// (`append_output_claims`) reproduces the input order,
    /// byte-identically. `canonical_order()` surfaces that field order as opening ids,
    /// which must line up one-for-one with the R1CS inputs.
    #[test]
    fn append_order_matches_r1cs_input_order() {
        let expected = SPARTAN_OUTER_R1CS_INPUTS
            .iter()
            .copied()
            .map(|variable| {
                JoltOpeningId::virtual_polynomial(variable, JoltRelationId::SpartanOuter)
            })
            .collect::<Vec<_>>();
        let openings = (0..SPARTAN_OUTER_R1CS_INPUTS.len() as u64)
            .map(Fr::from_u64)
            .collect::<Vec<_>>();
        let relation_form = output_values_from(&openings);

        assert_eq!(relation_form.canonical_order(), expected);
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

        let relation = OuterRemainder::new(dimensions, tau, uniskip_challenge);
        let input_points = OuterRemainderInputClaims::<Vec<Fr>>::default();
        // Capture the bound point (the third coefficient-table input); the table
        // itself is built lazily by the first `derive_output_term` call.
        let _ = relation
            .derive_opening_points(&remainder_challenges, &input_points)
            .unwrap();
        let point = vec![Fr::from_u64(7); 1 + log_t];
        let output_values = output_values_from(&openings);
        let output_points = output_points_at(&point);
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
