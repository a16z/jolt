use std::error::Error;

use jolt_field::{Field, F128, F192, F64, F8};
use jolt_poly::lagrange::F8DomainSizeError;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    grind_round, ClearProof, ClearSumcheckProof, F8Domain, RoundMessage, SumcheckClaim,
    SumcheckDomain, SumcheckError, SumcheckProof,
};
use jolt_transcript::{
    prover_transcript, Blake2b512, Encoding, FieldTranscript, ProverState, ProverTranscript,
    ScriptedTranscript,
};

const LABEL: &[u8] = b"f8-uniskip-reduction";

fn blake2b() -> ProverState<Blake2b512> {
    prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(LABEL), b"")
}

fn horner<F: Field>(coefficients: &[F], point: F) -> F {
    coefficients
        .iter()
        .rev()
        .fold(F::zero(), |value, &coefficient| value * point + coefficient)
}

fn round_sum_rule<F: Field + From<F8> + 'static>() -> Result<(), Box<dyn Error>> {
    for size in [1, 2, 3, 10, 64] {
        let domain = F8Domain::new(size)?;
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

fn subspace_power_sums<F: Field + From<F8> + 'static>() -> Result<(), Box<dyn Error>> {
    for exponent in 1..=6 {
        let size = 1 << exponent;
        let moments: Vec<F> = F8Domain::new(size)?.round_sum_coefficients(size - 1)?;
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

fn degree_overflow<F: Field + From<F8>>() -> Result<(), F8DomainSizeError> {
    let overflow: Result<Vec<F>, _> = F8Domain::new(4)?.round_sum_coefficients(usize::MAX);
    assert!(matches!(
        overflow,
        Err(SumcheckError::DegreeOverflow { degree: usize::MAX })
    ));
    Ok(())
}

/// A one-round proof of `coefficients` for `claim`, with the round nonce
/// ground as the prover grinds it: after the round message, before the
/// challenge.
fn full_clear_round<F, T>(
    claim: &SumcheckClaim<F>,
    coefficients: Vec<F>,
    mut transcript: T,
) -> Result<SumcheckProof<F, ()>, SumcheckError<F>>
where
    F: Field + Encoding,
    T: FieldTranscript<F> + ProverTranscript,
{
    let round_polynomial = UnivariatePoly::new(coefficients);
    round_polynomial.append_to_transcript(&mut transcript);
    let mut pow_nonces = Vec::new();
    grind_round::<F, T>(claim.degree, 0, &mut pow_nonces, &mut transcript)?;
    Ok(SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof {
        round_polynomials: vec![round_polynomial],
        pow_nonces,
    })))
}

fn one_round_reduction<F, T>(new_transcript: impl Fn() -> T) -> Result<(), Box<dyn Error>>
where
    F: Field + From<F8> + Encoding + 'static,
    T: FieldTranscript<F> + ProverTranscript,
{
    let coefficients = [0x31, 0x47, 0x59, 0x6d, 0x83, 0x97].map(|word| F::from(F8::from_raw(word)));
    let direct_sum: F = (0..4)
        .map(|index| horner(&coefficients, F::from(F8::from_raw(index))))
        .sum();
    let claim = SumcheckClaim::new(1, 5, direct_sum);
    let domain = F8Domain::new(4)?;
    let mut transcript = new_transcript();
    let reduced = full_clear_round(&claim, coefficients.to_vec(), new_transcript())?.verify(
        &claim,
        domain,
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
    let mut transcript = new_transcript();
    assert!(matches!(
        full_clear_round(&claim, changed_cubic, new_transcript())?.verify(
            &claim,
            domain,
            &mut transcript,
        ),
        Err(SumcheckError::RoundCheckFailed { round: 0, .. })
    ));

    let changed_constant = coefficients
        .iter()
        .enumerate()
        .map(|(index, &coefficient)| coefficient + if index == 0 { F::one() } else { F::zero() })
        .collect();
    let mut transcript = new_transcript();
    let reduced = full_clear_round(&claim, changed_constant, new_transcript())?.verify(
        &claim,
        domain,
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
    ($module:ident, $field:ty, $transcript:expr) => {
        mod $module {
            use super::*;

            #[test]
            fn polynomial_round_sum_matches_direct_evaluation() -> Result<(), Box<dyn Error>> {
                round_sum_rule::<$field>()
            }

            #[test]
            fn subspace_moments_vanish_below_top_power() -> Result<(), Box<dyn Error>> {
                subspace_power_sums::<$field>()
            }

            #[test]
            fn degree_overflow_is_an_error() -> Result<(), F8DomainSizeError> {
                degree_overflow::<$field>()
            }

            #[test]
            fn full_round_reduction_and_coefficient_tampering() -> Result<(), Box<dyn Error>> {
                one_round_reduction::<$field, _>($transcript)
            }
        }
    };
}

// `F64` challenges carry 64 bits, which no supported proof of work lifts to
// the 128-bit target, so a priced transcript rejects every round; the `F64`
// reduction runs unpriced and checks the domain's algebra, not Fiat-Shamir
// security.
field_tests!(f64, F64, || ScriptedTranscript::<_, F64>::new(blake2b()));
field_tests!(f128, F128, blake2b);
field_tests!(f192, F192, blake2b);

#[test]
fn new_rejects_sizes_outside_1_to_256() {
    for size in [0, 257, usize::MAX] {
        assert_eq!(F8Domain::new(size), Err(F8DomainSizeError { size }));
    }
}
