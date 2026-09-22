//! K16/D64 terminal integer rows from the reviewed frozen muldiv schedule.
//!
//! This is a relation component, not acceptance. Callers must authenticate the
//! profile/setup, derive weights from the bound point, bind z/e/t to proof bytes,
//! and constrain sparse challenge routing. A coefficients and RHS references
//! here are public pre-routed circuit inputs; witness-dependent routing is absent.
use jolt_field::Fr;
use jolt_r1cs::fp128_bn254::MODULUS;
use jolt_r1cs::integer_bn254::{IntegerError, IntegerTerm, SignedVar};
use jolt_r1cs::R1csBuilder;

const WIDTH: usize = 256;
const DEGREE: usize = 64;
const LENGTH: usize = WIDTH * DEGREE;
const CAP: u128 = 546_507_225;
const B: u128 = 23_377;
const Q: u128 = (MODULUS - 1) / 2;

/// The complete terminal z vector, range-checked and subject to its full L2 cap.
pub struct TerminalZ {
    coordinates: Vec<SignedVar>,
}

impl TerminalZ {
    /// Allocate exactly 16,384 coordinates and enforce the complete scheduled norm.
    /// None entries generate the identical layout without assignments.
    pub fn allocate(
        builder: &mut R1csBuilder<Fr>,
        witness: &[Option<i128>],
    ) -> Result<Self, IntegerError> {
        if witness.len() != LENGTH {
            return Err(IntegerError::Shape);
        }
        let coordinates = witness
            .iter()
            .map(|&x| SignedVar::allocate(builder, B, x))
            .collect::<Result<Vec<_>, _>>()?;
        SignedVar::enforce_squared_l2(builder, &coordinates, CAP)?;
        Ok(Self { coordinates })
    }

    /// Enforce one A coefficient row with authenticated, negacyclically routed
    /// setup constants, one per z coordinate. RHS is the routed challenge*t sum.
    pub fn enforce_a_row(
        &self,
        builder: &mut R1csBuilder<Fr>,
        coefficients: &[i128],
        rhs: &[(i8, &SignedVar)],
        quotient: Option<i128>,
    ) -> Result<(), IntegerError> {
        if coefficients.len() != LENGTH || coefficients.iter().any(|a| a.unsigned_abs() > Q) {
            return Err(IntegerError::Shape);
        }
        let mut terms = Self::rhs_terms(rhs)?;
        terms.extend(
            coefficients
                .iter()
                .zip(&self.coordinates)
                .map(|(&coefficient, value)| IntegerTerm::Linear { coefficient, value }),
        );
        let quotient = SignedVar::allocate(builder, 191_504_570, quotient)?;
        IntegerTerm::enforce_row(builder, &terms, MODULUS, &quotient)
    }

    /// Enforce one consistency coefficient row: sum_p weight[p]*z[64p+j]
    /// minus the routed challenge*e sum. Weights must be centered q handles.
    pub fn enforce_consistency_row(
        &self,
        builder: &mut R1csBuilder<Fr>,
        coordinate: usize,
        weights: &[SignedVar],
        rhs: &[(i8, &SignedVar)],
        quotient: Option<i128>,
    ) -> Result<(), IntegerError> {
        if coordinate >= DEGREE || weights.len() != WIDTH || weights.iter().any(|w| w.bound() != Q)
        {
            return Err(IntegerError::Shape);
        }
        let mut terms = Self::rhs_terms(rhs)?;
        for (position, weight) in weights.iter().enumerate() {
            let z = self
                .coordinates
                .get(DEGREE * position + coordinate)
                .ok_or(IntegerError::Shape)?;
            terms.push(IntegerTerm::Product {
                coefficient: 1,
                lhs: weight,
                rhs: z,
            });
        }
        let quotient = SignedVar::allocate(builder, 2_992_442, quotient)?;
        IntegerTerm::enforce_row(builder, &terms, MODULUS, &quotient)
    }

    fn rhs_terms<'a>(rhs: &[(i8, &'a SignedVar)]) -> Result<Vec<IntegerTerm<'a>>, IntegerError> {
        if rhs.len() != 294
            || rhs
                .iter()
                .any(|(a, x)| !matches!(a.unsigned_abs(), 1 | 2) || x.bound() != Q)
            || rhs
                .iter()
                .map(|(a, _)| u128::from(a.unsigned_abs()))
                .sum::<u128>()
                != 371
        {
            return Err(IntegerError::Shape);
        }
        Ok(rhs
            .iter()
            .map(|&(a, value)| IntegerTerm::Linear {
                coefficient: -i128::from(a),
                value,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn terminal_requires_complete_vector_and_exact_routing_shape() {
        let mut builder = R1csBuilder::new();
        assert!(matches!(
            TerminalZ::allocate(&mut builder, &[Some(0); 64]),
            Err(IntegerError::Shape)
        ));
        assert_eq!(builder.num_vars(), 1);
        let terminal = TerminalZ {
            coordinates: Vec::new(),
        };
        assert_eq!(
            terminal.enforce_a_row(&mut builder, &[], &[], Some(0)),
            Err(IntegerError::Shape)
        );
        assert_eq!(
            terminal.enforce_consistency_row(&mut builder, 64, &[], &[], Some(0)),
            Err(IntegerError::Shape)
        );
    }
}
