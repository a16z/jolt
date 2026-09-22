//! Constrained polynomial evaluation using the source field's arithmetic.

use crate::lagrange::{centered_domain_start, CenteredIntegerDomainError};
use jolt_field::{Field, Fr, Prime128OffsetA7F7, Ring};
use jolt_r1cs::fp128_bn254::{Fp128Error, Fp128Var};
use jolt_r1cs::{LinearCombination, R1csBuilder};

/// Evaluate little-endian coefficients over `Prime128OffsetA7F7` in BN254 R1CS.
///
/// All inputs must have been allocated in `builder`. Coefficient count fixes
/// the circuit shape; coefficient values and the evaluation point remain
/// constrained variables. The empty polynomial evaluates to constrained zero.
pub fn evaluate_fp128_bn254(
    builder: &mut R1csBuilder<Fr>,
    coefficients: &[Fp128Var],
    point: &Fp128Var,
) -> Result<Fp128Var, Fp128Error> {
    let Some((leading, remaining)) = coefficients.split_last() else {
        let zero = Fp128Var::allocate(builder, Some(0))?;
        builder.assert_equal(
            zero.variable(),
            LinearCombination::constant(Fr::from_u64(0)),
        );
        return Ok(zero);
    };
    leading.validate_indices(builder)?;
    let mut result = leading.clone();
    for coefficient in remaining.iter().rev() {
        result = result.multiply(builder, point)?.add(builder, coefficient)?;
    }
    Ok(result)
}

/// Invalid field handle or an unrepresentable Boolean table allocation.
#[derive(Debug, thiserror::Error)]
pub enum LagrangeR1csError {
    #[error(transparent)]
    Field(#[from] Fp128Error),
    #[error("Boolean Lagrange table exceeds the addressable allocation")]
    DomainTooLarge,
    #[error(transparent)]
    Domain(#[from] CenteredIntegerDomainError),
    #[error("paired Eq points have different lengths")]
    PointLengthMismatch,
    #[error("Lagrange denominator vanishes in q")]
    ZeroDenominator,
}

/// Boolean Lagrange weights in low-bit-first index order over the source field.
///
/// Entry i is product_j (bit_j(i) ? point[j] : 1-point[j]). Points are
/// constrained canonical handles in this builder; point length fixes the shape.
/// Prefix sharing needs one product and subtraction per existing entry per round.
pub fn lagrange_weights_fp128_bn254(
    builder: &mut R1csBuilder<Fr>,
    point: &[Fp128Var],
) -> Result<Vec<Fp128Var>, LagrangeR1csError> {
    for coordinate in point {
        coordinate.validate_indices(builder)?;
    }
    let shift = u32::try_from(point.len()).map_err(|_| LagrangeR1csError::DomainTooLarge)?;
    let size = 1usize
        .checked_shl(shift)
        .ok_or(LagrangeR1csError::DomainTooLarge)?;
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(size)
        .map_err(|_| LagrangeR1csError::DomainTooLarge)?;
    let one = Fp128Var::allocate(builder, Some(1))?;
    builder.assert_equal(one.variable(), LinearCombination::one());
    weights.push(one);
    for coordinate in point {
        let mut right = Vec::with_capacity(weights.len());
        for weight in &mut weights {
            let product = weight.multiply(builder, coordinate)?;
            *weight = weight.subtract(builder, &product)?;
            right.push(product);
        }
        weights.extend(right);
    }
    Ok(weights)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests inspect circuit assignments"
)]
mod tests {
    use super::*;
    use jolt_r1cs::fp128_bn254::MODULUS;
    use num_bigint::BigUint;

    #[test]
    fn horner_matches_independent_integer_evaluation_and_rejects_tampering() {
        let coefficients = [MODULUS - 1, 17, MODULUS - 2, 42];
        let point = MODULUS - 3;
        let mut builder = R1csBuilder::new();
        let vars: Vec<_> = coefficients
            .iter()
            .map(|&c| Fp128Var::allocate(&mut builder, Some(c)).unwrap())
            .collect();
        let x = Fp128Var::allocate(&mut builder, Some(point)).unwrap();
        let result = evaluate_fp128_bn254(&mut builder, &vars, &x).unwrap();
        let q = BigUint::from(MODULUS);
        let expected = coefficients
            .iter()
            .enumerate()
            .fold(BigUint::from(0u8), |sum, (i, &c)| {
                sum + BigUint::from(c) * BigUint::from(point).pow(i.try_into().unwrap())
            })
            % q;
        let expected: u128 = expected.try_into().unwrap();
        let witness = builder.witness().unwrap();
        assert_eq!(witness[result.variable().index()], Fr::from_u128(expected));
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        for variable in [x.variable(), vars[1].variable(), result.variable()] {
            let mut tampered = witness.clone();
            tampered[variable.index()] += Fr::from_u64(1);
            assert!(matrices.check_witness(&tampered).is_err());
        }
    }

    #[test]
    fn empty_and_constant_polynomials_are_constrained() {
        let mut builder = R1csBuilder::new();
        let x = Fp128Var::allocate(&mut builder, Some(9)).unwrap();
        let zero = evaluate_fp128_bn254(&mut builder, &[], &x).unwrap();
        let constant = Fp128Var::allocate(&mut builder, Some(7)).unwrap();
        let result = evaluate_fp128_bn254(&mut builder, &[constant], &x).unwrap();
        let witness = builder.witness().unwrap();
        assert_eq!(witness[zero.variable().index()], Fr::from_u64(0));
        assert_eq!(witness[result.variable().index()], Fr::from_u64(7));
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn constant_polynomial_rejects_out_of_range_coefficient() {
        let mut source = R1csBuilder::new();
        let coefficient = Fp128Var::allocate(&mut source, Some(7)).unwrap();
        let point = Fp128Var::allocate(&mut source, Some(9)).unwrap();
        let variable = coefficient.variable();
        let mut destination = R1csBuilder::new();
        let result = evaluate_fp128_bn254(&mut destination, &[coefficient], &point);
        assert!(
            matches!(result, Err(Fp128Error::UnknownVariable { variable: rejected }) if rejected == variable)
        );
        assert_eq!(destination.num_vars(), 1);
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "independent small Lagrange vector and matrix tests"
)]
mod lagrange_tests {
    use super::*;
    use jolt_r1cs::fp128_bn254::MODULUS;

    #[test]
    fn lagrange_table_uses_low_bit_first_order_and_fixed_shape() {
        let mut known = R1csBuilder::new();
        let point: Vec<_> = [2, 3]
            .into_iter()
            .map(|x| Fp128Var::allocate(&mut known, Some(x)).unwrap())
            .collect();
        let table = lagrange_weights_fp128_bn254(&mut known, &point).unwrap();
        let witness = known.witness().unwrap();
        for (value, expected) in table.iter().zip([2, MODULUS - 4, MODULUS - 3, 6]) {
            assert_eq!(witness[value.variable().index()], Fr::from_u128(expected));
        }
        let mut unknown = R1csBuilder::new();
        let point: Vec<_> = (0..2)
            .map(|_| Fp128Var::allocate(&mut unknown, None).unwrap())
            .collect();
        let _table = lagrange_weights_fp128_bn254(&mut unknown, &point).unwrap();
        let known = known.into_matrices();
        let unknown = unknown.into_matrices();
        assert_eq!(known.a, unknown.a);
        assert_eq!(known.b, unknown.b);
        assert_eq!(known.c, unknown.c);
        assert!(known.check_witness(&witness).is_ok());
        let mut swapped = witness;
        swapped.swap(table[1].variable().index(), table[2].variable().index());
        assert!(known.check_witness(&swapped).is_err());
    }
}

/// Public centered integer nodes and inverse denominators; private nodes never branch.
pub struct Fp128CenteredLagrangeShape {
    nodes: Vec<u128>,
    denominators: Vec<u128>,
}
impl Fp128CenteredLagrangeShape {
    pub fn new(size: usize) -> Result<Self, LagrangeR1csError> {
        let start = centered_domain_start(size)?;
        let nodes = (0..size)
            .map(|i| {
                let i = i64::try_from(i).map_err(|_| LagrangeR1csError::DomainTooLarge)?;
                let node = start
                    .checked_add(i)
                    .ok_or(LagrangeR1csError::DomainTooLarge)?;
                Ok(Prime128OffsetA7F7::from_i64(node))
            })
            .collect::<Result<Vec<_>, LagrangeR1csError>>()?;
        let denominators = nodes
            .iter()
            .enumerate()
            .map(|(i, x)| {
                nodes
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .fold(Prime128OffsetA7F7::from_u64(1), |a, (_, y)| a * (*x - *y))
                    .inverse()
                    .map(|x| x.to_canonical_u128())
                    .ok_or(LagrangeR1csError::ZeroDenominator)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            nodes: nodes.iter().map(|x| x.to_canonical_u128()).collect(),
            denominators,
        })
    }
    /// Product form works at grid nodes and emits the same rows for unknown points.
    pub fn evaluate(
        &self,
        builder: &mut R1csBuilder<Fr>,
        point: &Fp128Var,
    ) -> Result<Vec<Fp128Var>, LagrangeR1csError> {
        point.validate_indices(builder)?;
        let differences = self
            .nodes
            .iter()
            .map(|&node| {
                let node = Fp128Var::constant(builder, node)?;
                point.subtract(builder, &node)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.denominators
            .iter()
            .enumerate()
            .map(|(i, &denominator)| {
                let mut product = Fp128Var::constant(builder, denominator)?;
                for (j, difference) in differences.iter().enumerate() {
                    if i != j {
                        product = product.multiply(builder, difference)?;
                    }
                }
                Ok(product)
            })
            .collect()
    }
}

/// Paired-point Eq polynomial without materializing an exponential Boolean table.
pub fn eq_fp128_bn254(
    builder: &mut R1csBuilder<Fr>,
    left: &[Fp128Var],
    right: &[Fp128Var],
) -> Result<Fp128Var, LagrangeR1csError> {
    if left.len() != right.len() {
        return Err(LagrangeR1csError::PointLengthMismatch);
    }
    for value in left.iter().chain(right) {
        value.validate_indices(builder)?;
    }
    let one = Fp128Var::constant(builder, 1)?;
    let mut product = one.clone();
    for (x, y) in left.iter().zip(right) {
        let xy = x.multiply(builder, y)?;
        let nx = one.subtract(builder, x)?;
        let ny = one.subtract(builder, y)?;
        let complement = nx.multiply(builder, &ny)?;
        let factor = xy.add(builder, &complement)?;
        product = product.multiply(builder, &factor)?;
    }
    Ok(product)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "small native parity and private-handle mutation tests"
)]
mod centered_tests {
    use super::*;
    use crate::lagrange::centered_lagrange_evals;
    use crate::EqPolynomial;
    #[test]
    fn centered_private_nodes_and_eq_match_native() {
        let shape = Fp128CenteredLagrangeShape::new(10).unwrap();
        for value in (-4..=5)
            .map(Prime128OffsetA7F7::from_i64)
            .chain([Prime128OffsetA7F7::from_u64(19)])
        {
            let mut builder = R1csBuilder::new();
            let point = Fp128Var::allocate(&mut builder, Some(value.to_canonical_u128())).unwrap();
            let output = shape.evaluate(&mut builder, &point).unwrap();
            let native = centered_lagrange_evals(10, value).unwrap();
            let mut witness = builder.witness().unwrap();
            for (actual, expected) in output.iter().zip(native) {
                assert_eq!(
                    witness[actual.variable().index()],
                    Fr::from_u128(expected.to_canonical_u128())
                );
            }
            let matrices = builder.into_matrices();
            matrices.check_witness(&witness).unwrap();
            witness[output[0].variable().index()] += Fr::from_u64(1);
            assert!(matrices.check_witness(&witness).is_err());
        }
        let mut builder = R1csBuilder::new();
        let left = [2, 5, 9].map(|v| Fp128Var::allocate(&mut builder, Some(v)).unwrap());
        let right = [3, 7, 11].map(|v| Fp128Var::allocate(&mut builder, Some(v)).unwrap());
        let result = eq_fp128_bn254(&mut builder, &left, &right).unwrap();
        let native = EqPolynomial::<Prime128OffsetA7F7>::mle(
            &[2, 5, 9].map(Prime128OffsetA7F7::from_u64),
            &[3, 7, 11].map(Prime128OffsetA7F7::from_u64),
        );
        let witness = builder.witness().unwrap();
        assert_eq!(
            witness[result.variable().index()],
            Fr::from_u128(native.to_canonical_u128())
        );
        builder.into_matrices().check_witness(&witness).unwrap();
    }
    #[test]
    fn centered_shape_is_witness_independent_and_rejects_foreign_indices() {
        let shape = Fp128CenteredLagrangeShape::new(2).unwrap();
        let mut known = R1csBuilder::new();
        let x = Fp128Var::allocate(&mut known, Some(0)).unwrap();
        let _ = shape.evaluate(&mut known, &x).unwrap();
        let mut unknown = R1csBuilder::new();
        let y = Fp128Var::allocate(&mut unknown, None).unwrap();
        let _ = shape.evaluate(&mut unknown, &y).unwrap();
        let a = known.into_matrices();
        let b = unknown.into_matrices();
        assert_eq!(a.a, b.a);
        assert_eq!(a.b, b.b);
        assert_eq!(a.c, b.c);
        let mut empty = R1csBuilder::new();
        assert!(shape.evaluate(&mut empty, &x).is_err());
        assert_eq!(empty.num_vars(), 1);
        assert!(Fp128CenteredLagrangeShape::new(0).is_err());
    }
}
