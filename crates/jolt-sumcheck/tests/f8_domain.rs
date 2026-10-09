#![cfg(feature = "binary")]

use std::error::Error;

use jolt_field::{CanonicalEncoding, Field, F128, F192, F64, F8};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    ClearProof, ClearSumcheckProof, F8Domain, SumcheckClaim, SumcheckDomain, SumcheckError,
    SumcheckProof, UNISKIP_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};

fn horner<F: Field>(coefficients: &[F], point: F) -> F {
    coefficients
        .iter()
        .rev()
        .fold(F::zero(), |value, &coefficient| value * point + coefficient)
}

fn round_sum_rule<F: Field + From<F8>>() -> Result<(), SumcheckError<F>> {
    for size in [1, 2, 3, 10, 64] {
        let domain = F8Domain::new(size);
        assert_eq!(domain.size(), size);
        let mut degrees = vec![0, 1, size - 1, size, 2 * size];
        degrees.sort_unstable();
        degrees.dedup();
        for degree in degrees {
            let coefficients: Vec<_> = [0x91, 0xa3, 0xb5, 0xc7, 0xd9]
                .into_iter()
                .cycle()
                .take(degree + 1)
                .map(|word| F::from(F8::from_raw(word)))
                .collect();
            let moments: Vec<F> = domain.round_sum_coefficients(degree)?;
            assert_eq!(moments.len(), degree + 1);
            let weighted_sum: F = coefficients
                .iter()
                .zip(&moments)
                .map(|(&coefficient, &moment)| coefficient * moment)
                .sum();
            let direct_sum: F = (0..size)
                .map(|index| horner(&coefficients, F::from(F8::from_raw(index as u8))))
                .sum();
            assert_eq!(weighted_sum, direct_sum, "size={size}, degree={degree}");
        }
    }
    Ok(())
}

fn subspace_power_sums<F: Field + From<F8>>() -> Result<(), SumcheckError<F>> {
    for exponent in 1..=6 {
        let size = 1 << exponent;
        let moments: Vec<F> = F8Domain::new(size).round_sum_coefficients(size - 1)?;
        assert_eq!(moments.len(), size);
        assert!(moments.iter().take(size - 1).all(|moment| moment.is_zero()));
        let nonzero_product: F = (1..size)
            .map(|index| F::from(F8::from_raw(index as u8)))
            .product();
        assert!(!nonzero_product.is_zero());
        assert_eq!(moments.last(), Some(&nonzero_product));
    }
    Ok(())
}

fn domain_errors<F: Field + From<F8>>() {
    for size in [0, 257, usize::MAX] {
        let result: Result<Vec<F>, _> = F8Domain::new(size).round_sum_coefficients(2);
        assert!(matches!(result, Err(SumcheckError::InvalidF8Domain { size: got }) if got == size));
    }
    let overflow: Result<Vec<F>, _> = F8Domain::new(4).round_sum_coefficients(usize::MAX);
    assert!(matches!(
        overflow,
        Err(SumcheckError::DegreeOverflow { degree: usize::MAX })
    ));
    let invalid_first: Result<Vec<F>, _> = F8Domain::new(0).round_sum_coefficients(usize::MAX);
    assert!(matches!(
        invalid_first,
        Err(SumcheckError::InvalidF8Domain { size: 0 })
    ));
}

fn full_clear_round<F: Field>(coefficients: Vec<F>) -> SumcheckProof<F, ()> {
    SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof {
        round_polynomials: vec![UnivariatePoly::new(coefficients)],
    }))
}

fn one_round_reduction<F>() -> Result<(), Box<dyn Error>>
where
    F: Field + From<F8> + CanonicalEncoding + AppendToTranscript,
{
    let coefficients = [0x31, 0x47, 0x59, 0x6d, 0x83, 0x97].map(|word| F::from(F8::from_raw(word)));
    let direct_sum: F = (0..4)
        .map(|index| horner(&coefficients, F::from(F8::from_raw(index))))
        .sum();
    let claim = SumcheckClaim::new(1, 5, direct_sum);
    let mut transcript = Blake2bTranscript::<F>::new(b"f8-uniskip-reduction");
    let reduced = full_clear_round(coefficients.to_vec()).verify(
        &claim,
        F8Domain::new(4),
        UNISKIP_ROUND_TRANSCRIPT_LABEL,
        &mut transcript,
    )?;
    assert_eq!(reduced.point.as_slice().len(), 1);
    let challenge = *reduced
        .point
        .as_slice()
        .first()
        .ok_or("missing challenge")?;
    assert_eq!(reduced.value, horner(&coefficients, challenge));

    let changed_cubic = coefficients
        .iter()
        .enumerate()
        .map(|(index, &coefficient)| {
            coefficient
                + if index == 3 {
                    F::from(F8::from_raw(0x9b))
                } else {
                    F::zero()
                }
        })
        .collect();
    let mut transcript = Blake2bTranscript::<F>::new(b"f8-uniskip-reduction");
    assert!(matches!(
        full_clear_round(changed_cubic).verify(
            &claim,
            F8Domain::new(4),
            UNISKIP_ROUND_TRANSCRIPT_LABEL,
            &mut transcript,
        ),
        Err(SumcheckError::RoundCheckFailed { round: 0, .. })
    ));

    let changed_constant = coefficients
        .iter()
        .enumerate()
        .map(|(index, &coefficient)| coefficient + if index == 0 { F::one() } else { F::zero() })
        .collect();
    let mut transcript = Blake2bTranscript::<F>::new(b"f8-uniskip-reduction");
    let reduced = full_clear_round(changed_constant).verify(
        &claim,
        F8Domain::new(4),
        UNISKIP_ROUND_TRANSCRIPT_LABEL,
        &mut transcript,
    )?;
    let challenge = *reduced
        .point
        .as_slice()
        .first()
        .ok_or("missing challenge")?;
    assert_ne!(reduced.value, horner(&coefficients, challenge));
    assert_eq!(reduced.value, horner(&coefficients, challenge) + F::one());
    Ok(())
}

macro_rules! field_tests {
    ($module:ident, $field:ty) => {
        mod $module {
            use super::*;

            #[test]
            fn polynomial_round_sum_matches_direct_evaluation() -> Result<(), SumcheckError<$field>>
            {
                round_sum_rule::<$field>()
            }

            #[test]
            fn subspace_moments_vanish_below_top_power() -> Result<(), SumcheckError<$field>> {
                subspace_power_sums::<$field>()
            }

            #[test]
            fn invalid_size_precedes_degree_overflow() {
                domain_errors::<$field>();
            }

            #[test]
            fn full_round_reduction_and_coefficient_tampering() -> Result<(), Box<dyn Error>> {
                one_round_reduction::<$field>()
            }
        }
    };
}

field_tests!(f64, F64);
field_tests!(f128, F128);
field_tests!(f192, F192);
