//! K16/D64 degree-one Lagrange terminal opening; not a complete verifier.
//! Exact source specialization and historical-fixture limits are recorded in
//! `specs/akita-wrapper/scalar-constraints.md`. All handles must share a builder,
//! ONE is externally fixed, and proof-byte/transcript/profile binding is external.
use jolt_field::Fr;
use jolt_poly::r1cs::{lagrange_weights_fp128_bn254, LagrangeR1csError};
use jolt_r1cs::fp128_bn254::{Fp128Error, Fp128Var};
use jolt_r1cs::integer_bn254::{IntegerError, SignedVar};
use jolt_r1cs::R1csBuilder;
use thiserror::Error;

use super::terminal::TerminalZ;

/// Invalid fixed-profile shape, handle, or constrained arithmetic input.
#[derive(Debug, Error)]
pub enum ScalarR1csError {
    #[error("expected the K16/D64 degree-one profile shape")]
    Shape,
    #[error(transparent)]
    Field(#[from] Fp128Error),
    #[error(transparent)]
    Polynomial(#[from] LagrangeR1csError),
    #[error(transparent)]
    Integer(#[from] IntegerError),
}

/// Point-derived handles for one degree-one terminal opening.
///
/// Coordinates 0..6 are packed-inner bits, 6..14 position bits, 14..17 block
/// bits, each in low-bit-first Boolean index order. Exactly seven blocks are
/// live; the eighth weight is omitted without renormalization.
pub struct TerminalPointVar {
    inner: Vec<Fp128Var>,
    positions: Vec<Fp128Var>,
    blocks: Vec<Fp128Var>,
}

impl TerminalPointVar {
    /// Prepare exactly 17 already-canonical point coordinates. The full-length
    /// supported profile is explicit: no implicit zero-padding or other basis.
    pub fn prepare(
        builder: &mut R1csBuilder<Fr>,
        point: &[Fp128Var],
    ) -> Result<Self, ScalarR1csError> {
        if point.len() != 17 {
            return Err(ScalarR1csError::Shape);
        }
        for coordinate in point {
            coordinate.validate_indices(builder)?;
        }
        let (inner, outer) = point.split_at(6);
        let (positions, blocks) = outer.split_at(8);
        let inner = lagrange_weights_fp128_bn254(builder, inner)?;
        let positions = lagrange_weights_fp128_bn254(builder, positions)?;
        let mut blocks = lagrange_weights_fp128_bn254(builder, blocks)?;
        blocks.truncate(7);
        Ok(Self {
            inner,
            positions,
            blocks,
        })
    }

    /// Bind point-derived position weights into one existing consistency row.
    /// Centered assignments are constrained, not trusted coefficient inputs.
    /// Sparse RHS routing and quotient assignments retain the terminal-row contract.
    pub fn enforce_consistency_row(
        &self,
        builder: &mut R1csBuilder<Fr>,
        z: &TerminalZ,
        coordinate: usize,
        rhs: &[(i8, &SignedVar)],
        centered_witness: &[Option<i128>],
        quotient: Option<i128>,
    ) -> Result<(), ScalarR1csError> {
        if centered_witness.len() != self.positions.len() {
            return Err(ScalarR1csError::Shape);
        }
        let weights = self
            .positions
            .iter()
            .zip(centered_witness)
            .map(|(value, &assignment)| SignedVar::centered(builder, value, assignment))
            .collect::<Result<Vec<_>, _>>()?;
        z.enforce_consistency_row(builder, coordinate, &weights, rhs, quotient)?;
        Ok(())
    }

    /// Enforce 64 seven-block reductions followed by their 64-term inner dot
    /// product, all modulo q. e is block-major [7][64]; scale and row weight are
    /// one in this profile. Every full-width product uses the CRT field gadget.
    pub fn enforce_scalar_opening(
        &self,
        builder: &mut R1csBuilder<Fr>,
        e: &[Fp128Var],
        claim: &Fp128Var,
    ) -> Result<(), ScalarR1csError> {
        if e.len() != 448 {
            return Err(ScalarR1csError::Shape);
        }
        for value in self
            .inner
            .iter()
            .chain(&self.blocks)
            .chain(e)
            .chain(std::iter::once(claim))
        {
            value.validate_indices(builder)?;
        }
        let mut result: Option<Fp128Var> = None;
        for (coordinate, inner) in self.inner.iter().enumerate() {
            let mut outer: Option<Fp128Var> = None;
            for (block, weight) in self.blocks.iter().enumerate() {
                let value = e
                    .get(64 * block + coordinate)
                    .ok_or(ScalarR1csError::Shape)?;
                let product = value.multiply(builder, weight)?;
                outer = Some(match outer {
                    Some(sum) => sum.add(builder, &product)?,
                    None => product,
                });
            }
            let product = outer
                .ok_or(ScalarR1csError::Shape)?
                .multiply(builder, inner)?;
            result = Some(match result {
                Some(sum) => sum.add(builder, &product)?,
                None => product,
            });
        }
        builder.assert_equal(
            result.ok_or(ScalarR1csError::Shape)?.variable(),
            claim.variable(),
        );
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests inspect fixed profile vectors and mutate complete assignments"
)]
mod tests {
    use super::*;
    use jolt_field::Ring;

    #[test]
    fn scalar_opening_selects_low_bit_first_inner_and_block_coordinates() {
        let mut builder = R1csBuilder::new();
        let point_values = [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0];
        let point: Vec<_> = point_values
            .into_iter()
            .map(|x| Fp128Var::allocate(&mut builder, Some(x)).unwrap())
            .collect();
        let prepared = TerminalPointVar::prepare(&mut builder, &point).unwrap();
        let e: Vec<_> = (0..448)
            .map(|i| {
                Fp128Var::allocate(&mut builder, Some(if i == 2 * 64 + 5 { 43 } else { 0 }))
                    .unwrap()
            })
            .collect();
        let claim_start = builder.num_vars();
        let claim = Fp128Var::allocate(&mut builder, Some(43)).unwrap();
        let claim_end = builder.num_vars();
        prepared
            .enforce_scalar_opening(&mut builder, &e, &claim)
            .unwrap();
        let witness = builder.witness().unwrap();
        for (weights, selected) in [
            (&prepared.inner, 5),
            (&prepared.positions, 18),
            (&prepared.blocks, 2),
        ] {
            for (i, weight) in weights.iter().enumerate() {
                assert_eq!(
                    witness[weight.variable().index()],
                    Fr::from_u64(u64::from(i == selected))
                );
            }
        }
        let matrices = builder.into_matrices();
        let mut unknown = R1csBuilder::new();
        let unknown_point: Vec<_> = (0..17)
            .map(|_| Fp128Var::allocate(&mut unknown, None).unwrap())
            .collect();
        let unknown_prepared = TerminalPointVar::prepare(&mut unknown, &unknown_point).unwrap();
        let unknown_e: Vec<_> = (0..448)
            .map(|_| Fp128Var::allocate(&mut unknown, None).unwrap())
            .collect();
        let unknown_claim = Fp128Var::allocate(&mut unknown, None).unwrap();
        unknown_prepared
            .enforce_scalar_opening(&mut unknown, &unknown_e, &unknown_claim)
            .unwrap();
        let unknown = unknown.into_matrices();
        assert_eq!(matrices.num_vars, unknown.num_vars);
        assert_eq!(matrices.a, unknown.a);
        assert_eq!(matrices.b, unknown.b);
        assert_eq!(matrices.c, unknown.c);
        assert!(matrices.check_witness(&witness).is_ok());
        let mut other_claim = R1csBuilder::new();
        let _value = Fp128Var::allocate(&mut other_claim, Some(44)).unwrap();
        let mut wrong_claim = witness.clone();
        wrong_claim[claim_start..claim_end].copy_from_slice(&other_claim.witness().unwrap()[1..]);
        assert_eq!(
            matrices.check_witness(&wrong_claim),
            Err(matrices.num_constraints - 1)
        );
        for variable in [
            claim.variable(),
            e[2 * 64 + 5].variable(),
            point[0].variable(),
            prepared.blocks[2].variable(),
        ] {
            let mut wrong = witness.clone();
            wrong[variable.index()] += Fr::from_u64(1);
            assert!(matrices.check_witness(&wrong).is_err());
        }
    }

    #[test]
    fn eighth_block_weight_is_omitted_without_renormalization() {
        use jolt_r1cs::fp128_bn254::MODULUS;
        let mut builder = R1csBuilder::new();
        let point: Vec<_> = (0..17)
            .map(|i| {
                Fp128Var::allocate(
                    &mut builder,
                    Some(if i < 14 { 0 } else { (i - 12) as u128 }),
                )
                .unwrap()
            })
            .collect();
        let prepared = TerminalPointVar::prepare(&mut builder, &point).unwrap();
        let witness = builder.witness().unwrap();
        assert_eq!(prepared.blocks.len(), 7);
        for (value, expected) in prepared.blocks.iter().zip([
            MODULUS - 6,
            12,
            9,
            MODULUS - 18,
            8,
            MODULUS - 16,
            MODULUS - 12,
        ]) {
            assert_eq!(witness[value.variable().index()], Fr::from_u128(expected));
        }
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn profile_rejects_wrong_shape_and_unknown_indices_before_emission() {
        let mut builder = R1csBuilder::new();
        assert!(matches!(
            TerminalPointVar::prepare(&mut builder, &[]),
            Err(ScalarR1csError::Shape)
        ));
        assert_eq!(builder.num_vars(), 1);
        let point: Vec<_> = (0..17)
            .map(|_| Fp128Var::allocate(&mut builder, None).unwrap())
            .collect();
        let mut fresh = R1csBuilder::new();
        assert!(matches!(
            TerminalPointVar::prepare(&mut fresh, &point),
            Err(ScalarR1csError::Field(Fp128Error::UnknownVariable { .. }))
        ));
        assert_eq!(fresh.num_vars(), 1);
    }
}
