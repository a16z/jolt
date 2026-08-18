//! The stage 5 `InstructionReadRaf` sumcheck instance.
//!
//! The most intricate stage 5 relation: its output `Expr` references indexed
//! opening families (lookup-table flags, virtual RA chunks) and point-derived
//! *publics* (`EqTableValue`, `EqRafConstant`, `EqRafFlag`) computed from the
//! instruction address/cycle points and the upstream claim-reduction point. The
//! full instruction address is split across the virtual-RA opening points, so
//! `derive_output_term` reconstructs it from the output opening cells.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionReadRafChallenges, InstructionReadRafInputClaims, InstructionReadRafOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::instruction::{
        upper_half_all_ones, InstructionReadRafDimensions, CANONICAL_INSTRUCTION_ADDRESS,
    },
    InstructionReadRafPublic, JoltDerivedId, JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_poly::{
    try_eq_mle, IdentityPolynomial, MultilinearEvaluation, OperandPolynomial, OperandSide,
};

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage2::{Stage2BatchOutputClaims, Stage2BatchOutputPoints};
use crate::VerifierError;

/// Wire the consumed opening *values* from the upstream instruction claim-reduction
/// (stage 2). The reduced `lookup_output` wire cell is a cross-relation alias of
/// the product remainder's; stage 2's generated `validate_aliases` (run inside its
/// `expected_final_claim`) enforces their equality before this wiring reads it.
/// Takes the ZK-agnostic stage-2 output-claims aggregate (both the clear and ZK
/// stage-2 outputs expose it).
pub fn instruction_read_raf_input_values_from_upstream<F: Field>(
    stage2: &Stage2BatchOutputClaims<F>,
) -> InstructionReadRafInputClaims<F> {
    let reduction = &stage2.instruction_claim_reduction;
    InstructionReadRafInputClaims {
        lookup_output: reduction.lookup_output,
        left_lookup_operand: reduction.left_lookup_operand,
        right_lookup_operand: reduction.right_lookup_operand,
    }
}

/// Wire the consumed opening *points* from the upstream instruction claim-reduction
/// (stage 2). All three share the claim-reduction opening point.
pub fn instruction_read_raf_input_points_from_upstream<F: Field>(
    stage2: &Stage2BatchOutputPoints<F>,
) -> InstructionReadRafInputClaims<Vec<F>> {
    let point = stage2.instruction_claim_reduction_point().to_vec();
    InstructionReadRafInputClaims {
        lookup_output: point.clone(),
        left_lookup_operand: point.clone(),
        right_lookup_operand: point,
    }
}

#[derive(Clone)]
pub struct InstructionReadRaf<F: Field> {
    symbolic: relations::instruction::ReadRaf,
    dimensions: InstructionReadRafDimensions,
    _field: core::marker::PhantomData<F>,
}

impl<F: Field> InstructionReadRaf<F> {
    pub fn new(dimensions: InstructionReadRafDimensions) -> Self {
        Self {
            symbolic: relations::instruction::ReadRaf::new(dimensions),
            dimensions,
            _field: core::marker::PhantomData,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::InstructionReadRaf,
        reason: reason.to_string(),
    }
}

/// Reconstruct the instruction address point from the virtual-RA opening points:
/// each RA opening point is `chunk ++ r_cycle`, and the chunks tile the address
/// in order, so stripping the trailing cycle and concatenating recovers it.
pub(crate) fn reconstruct_r_address<F: Field>(
    output_points: &InstructionReadRafOutputClaims<Vec<F>>,
    cycle_len: usize,
) -> Vec<F> {
    output_points
        .instruction_ra()
        .iter()
        .flat_map(|point| {
            // Each point is `chunk ++ r_cycle` by construction
            // (`derive_opening_points`), so the saturation never engages.
            let chunk_len = point.len().saturating_sub(cycle_len);
            point.iter().take(chunk_len).copied()
        })
        .collect()
}

impl<F: Field> InstructionReadRaf<F> {
    pub fn dimensions(&self) -> InstructionReadRafDimensions {
        self.dimensions
    }
}

impl<F: Field> ConcreteSumcheck<F> for InstructionReadRaf<F> {
    type Symbolic = relations::instruction::ReadRaf;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &InstructionReadRafInputClaims<Vec<F>>,
    ) -> Result<InstructionReadRafOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .dimensions
            .opening_point(sumcheck_point)
            .map_err(public_input_failed)?;
        let ra_count = self.dimensions.num_virtual_ra_polys();
        let chunk_size = opening_point
            .r_address
            .len()
            .checked_div(ra_count)
            .filter(|chunk_size| {
                chunk_size.checked_mul(ra_count) == Some(opening_point.r_address.len())
            })
            .ok_or_else(|| {
                public_input_failed(format!(
                    "instruction address point length {} is not divisible by virtual RA count {ra_count}",
                    opening_point.r_address.len()
                ))
            })?;
        let instruction_ra = opening_point
            .r_address
            .chunks(chunk_size)
            .map(|chunk| [chunk, opening_point.r_cycle.as_slice()].concat())
            .collect::<Vec<_>>();
        let lookup_table_flags =
            vec![opening_point.r_cycle.clone(); LookupTableKind::<RISCV_XLEN>::COUNT];
        Ok(InstructionReadRafOutputClaims {
            lookup_table_flags,
            instruction_ra,
            instruction_raf_flag: opening_point.r_cycle,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        input_points: &InstructionReadRafInputClaims<Vec<F>>,
        output_points: &InstructionReadRafOutputClaims<Vec<F>>,
        challenges: &InstructionReadRafChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::InstructionReadRaf(public) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let r_cycle = output_points.instruction_raf_flag();
        let r_address = reconstruct_r_address(output_points, r_cycle.len());
        // eq over the upstream instruction claim-reduction cycle point; all three
        // consumed openings share that point, so the lookup-output input carries it.
        let eq_reduction =
            try_eq_mle(input_points.lookup_output(), r_cycle).map_err(public_input_failed)?;
        let address_bits = self.dimensions.instruction_address_bits();
        // `upper_half_all_ones` below slices the leading half of this point, so a
        // reconstruction of the wrong length would silently guard the wrong bits
        // rather than fail. The point is verifier-derived, but the mis-slice is
        // invisible if that ever stops being true.
        if r_address.len() != address_bits {
            return Err(public_input_failed(format!(
                "instruction address point has {} coordinates, expected {address_bits}",
                r_address.len()
            )));
        }
        let left = || OperandPolynomial::new(address_bits, OperandSide::Left).evaluate(&r_address);
        let right =
            || OperandPolynomial::new(address_bits, OperandSide::Right).evaluate(&r_address);
        // The RAF publics fold the batching gamma into the operand evaluations.
        let gamma = challenges.gamma;
        let gamma2 = gamma * gamma;
        match public {
            InstructionReadRafPublic::EqTableValue(index) => {
                let table = LookupTableKind::<RISCV_XLEN>::iter()
                    .find(|table| table.index() == *index)
                    .ok_or_else(|| {
                        public_input_failed(format!("unknown lookup table index {index}"))
                    })?;
                Ok(eq_reduction * table.evaluate_mle::<F, F>(&r_address))
            }
            InstructionReadRafPublic::EqRafConstant => {
                Ok(eq_reduction * (gamma * left() + gamma2 * right()))
            }
            InstructionReadRafPublic::EqRafFlag => {
                let identity = IdentityPolynomial::new(address_bits).evaluate(&r_address);
                let mut raf = gamma2 * identity - gamma * left() - gamma2 * right();
                if CANONICAL_INSTRUCTION_ADDRESS {
                    // The input claim carries no γ³ term, so the sumcheck can only
                    // close if Σ_j eq(r_red,j)·RafFlag_j·U(k_j) = 0 — i.e. no
                    // identity-RAF cycle has an all-ones upper address half. Every
                    // fp128 alias `k = r + p < 2^128` has exactly that shape, and
                    // `U(k) = 0` implies `k < 2^128 - 2^64 < p`, so the surviving
                    // address is the canonical representative of `k mod p` and the
                    // identity leg pins it exactly. See `CANONICAL_INSTRUCTION_ADDRESS`.
                    raf += gamma2 * gamma * upper_half_all_ones(&r_address);
                }
                Ok(eq_reduction * raf)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::relations::ConcreteSumcheck;
    use jolt_claims::protocols::jolt::geometry::instruction::read_raf_output_openings;
    use jolt_claims::SymbolicSumcheck;
    use jolt_field::{Fr, FromPrimitiveInt as _};

    /// Locks the `expected_output_openings` invariant for the one stage-5 relation
    /// with a size-parameter-dependent shape: the openings the read-RAF output `Expr`
    /// references (looping over every lookup table and RA chunk) must be exactly the
    /// geometry's `read_raf_output_openings` set. If they drift, the ZK
    /// commitment-count and clear shape-check derived from the `Expr` would be wrong.
    #[test]
    #[expect(clippy::unwrap_used)]
    fn expected_output_openings_matches_geometry_shape() {
        let dimensions = InstructionReadRafDimensions::try_from((5, 128, 3)).unwrap();
        let expected: std::collections::BTreeSet<_> = {
            let openings = read_raf_output_openings(dimensions);
            openings
                .lookup_table_flags
                .into_iter()
                .chain(openings.instruction_ra)
                .chain(std::iter::once(openings.instruction_raf_flag))
                .collect()
        };
        assert_eq!(
            InstructionReadRaf::<Fr>::new(dimensions)
                .symbolic()
                .expected_output_openings::<Fr>(),
            expected,
        );
    }

    /// Pins the canonical-address term inside `EqRafFlag`.
    ///
    /// The term is what forces `Σ_j eq(r_red,j)·RafFlag_j·U(k_j) = 0`, and it is
    /// invisible in the symbolic relation — it rides inside a derived public. A
    /// refactor that dropped it would still typecheck, still produce a
    /// well-formed proof, and silently reopen the fp128 alias. Asserting the
    /// closed form (rather than just "nonzero") also pins the γ *power* and the
    /// leading-half slice, the two transposition traps.
    #[test]
    #[expect(
        clippy::unwrap_used,
        clippy::as_conversions,
        clippy::indexing_slicing,
        clippy::integer_division,
        reason = "test fixture arithmetic over compile-time dimensions"
    )]
    fn eq_raf_flag_carries_the_canonical_address_term() {
        const LOG_T: usize = 5;
        const ADDRESS_BITS: usize = 128;

        let dimensions = InstructionReadRafDimensions::try_from((LOG_T, ADDRESS_BITS, 4)).unwrap();
        let relation = InstructionReadRaf::<Fr>::new(dimensions);

        // Distinct non-Boolean coordinates, so no factor can vanish by accident.
        let sumcheck_point: Vec<Fr> = (0..ADDRESS_BITS + LOG_T)
            .map(|i| Fr::from_u64(i as u64 + 2))
            .collect();
        let opening = dimensions.opening_point(&sumcheck_point).unwrap();

        let input_points = InstructionReadRafInputClaims {
            lookup_output: opening.r_cycle.clone(),
            left_lookup_operand: opening.r_cycle.clone(),
            right_lookup_operand: opening.r_cycle.clone(),
        };
        let output_points = relation
            .derive_opening_points(&sumcheck_point, &input_points)
            .unwrap();
        let gamma = Fr::from_u64(7);
        let challenges = InstructionReadRafChallenges { gamma };

        let actual = relation
            .derive_output_term(
                &JoltDerivedId::InstructionReadRaf(InstructionReadRafPublic::EqRafFlag),
                &input_points,
                &output_points,
                &challenges,
            )
            .unwrap();

        let r_address = reconstruct_r_address(&output_points, opening.r_cycle.len());
        assert_eq!(r_address.len(), ADDRESS_BITS);
        let gamma2 = gamma * gamma;
        let baseline = gamma2 * IdentityPolynomial::new(ADDRESS_BITS).evaluate(&r_address)
            - gamma * OperandPolynomial::new(ADDRESS_BITS, OperandSide::Left).evaluate(&r_address)
            - gamma2
                * OperandPolynomial::new(ADDRESS_BITS, OperandSide::Right).evaluate(&r_address);

        // Computed independently of `upper_half_all_ones`: the product of the
        // *leading* half of the address point.
        let canonical_term = r_address[..ADDRESS_BITS / 2]
            .iter()
            .fold(Fr::from_u64(1), |acc, coordinate| acc * *coordinate);
        assert_ne!(
            canonical_term,
            Fr::from_u64(0),
            "test point must exercise a nonzero canonical term"
        );

        // Note `eq(x, x) != 1` at a non-Boolean point, so the shared reduction
        // point still contributes a factor.
        let eq_reduction = try_eq_mle(&opening.r_cycle, &opening.r_cycle).unwrap();
        if CANONICAL_INSTRUCTION_ADDRESS {
            assert_eq!(
                actual,
                eq_reduction * (baseline + gamma2 * gamma * canonical_term)
            );
            assert_ne!(
                actual,
                eq_reduction * baseline,
                "the canonical-address term was dropped"
            );
        } else {
            assert_eq!(actual, eq_reduction * baseline);
        }
    }
}
