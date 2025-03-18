use super::{
    inner_products::{InnerProduct, MultiexponentiationInnerProduct},
    Error,
};
use crate::field::JoltField;
use crate::poly::commitment::kzg::KZGVerifierKey;
use crate::poly::unipoly::UniPoly;
use ark_ec::scalar_mul::fixed_base::FixedBase;
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{Field, One, PrimeField, Zero};
use ark_std::{end_timer, start_timer};
use itertools::Itertools;

pub mod structured_scalar_message;

/// Returns the proof that the polynomial
///
/// Calculate commitment of quotient polynomial such that w(X).(X-z) = P(X) - v
///
/// `kzg_challenge`: X
/// `srs_powers`:
pub fn prove_commitment_key_kzg_opening<G: CurveGroup>(
    srs_powers: &[G],
    transcript: &[G::ScalarField],
    r_shift: &G::ScalarField,
    kzg_challenge: &G::ScalarField,
) -> Result<G, Error>
where
    G::ScalarField: JoltField,
{
    let ck_polynomial =
        UniPoly::from_coeff(polynomial_coefficients_from_transcript(transcript, r_shift));
    assert_eq!(srs_powers.len(), ck_polynomial.coeffs.len());

    let eval = start_timer!(|| "polynomial eval");
    let ck_polynomial_c_eval =
        polynomial_evaluation_product_form_from_transcript(transcript, kzg_challenge, r_shift);
    end_timer!(eval);

    let quotient = start_timer!(|| "polynomial quotient");
    let (quotient_polynomial, _remainder) = (ck_polynomial
        - UniPoly::from_coeff(vec![ck_polynomial_c_eval]))
    .divide_with_remainder(&UniPoly::from_coeff(vec![
        -*kzg_challenge,
        <G::ScalarField>::one(),
    ]))
    .unwrap();
    end_timer!(quotient);

    let mut quotient_polynomial_coeffs = quotient_polynomial.coeffs;
    quotient_polynomial_coeffs.resize(srs_powers.len(), <G::ScalarField>::zero());

    let multiexp = start_timer!(|| "opening multiexp");
    let opening =
        MultiexponentiationInnerProduct::inner_product(srs_powers, &quotient_polynomial_coeffs)?;
    end_timer!(multiexp);
    Ok(opening)
}

//TODO: Figure out how to avoid needing two separate methods for verification of opposite groups
pub fn verify_commitment_key_g2_kzg_opening<P: Pairing>(
    v_srs: &KZGVerifierKey<P>,
    ck_final: &P::G2,
    ck_opening: &P::G2,
    transcript: &[P::ScalarField],
    r_shift: &P::ScalarField,
    kzg_challenge: &P::ScalarField,
) -> Result<bool, Error> {
    let ck_polynomial_c_eval =
        polynomial_evaluation_product_form_from_transcript(transcript, kzg_challenge, r_shift);
    Ok(P::pairing(
        v_srs.g1.into_group(),
        *ck_final - v_srs.g2.into_group() * ck_polynomial_c_eval,
    ) == P::pairing(
        v_srs.alpha_g1.into_group() - v_srs.g1 * kzg_challenge,
        *ck_opening,
    ))
}

pub fn structured_generators_scalar_power<G: CurveGroup>(
    num: usize,
    g: &G,
    s: &G::ScalarField,
) -> Vec<G> {
    assert!(num > 0);
    let mut powers_of_scalar = vec![];
    let mut pow_s = G::ScalarField::one();
    for _ in 0..num {
        powers_of_scalar.push(pow_s);
        pow_s *= s;
    }

    let window_size = FixedBase::get_mul_window_size(num);

    let scalar_bits = G::ScalarField::MODULUS_BIT_SIZE as usize;
    let g_table = FixedBase::get_window_table(scalar_bits, window_size, *g);
    FixedBase::msm::<G>(scalar_bits, window_size, &g_table, &powers_of_scalar)
}

fn polynomial_evaluation_product_form_from_transcript<F: Field>(
    transcript: &[F],
    z: &F,
    r_shift: &F,
) -> F {
    let mut power_2_zr = (*z * z) * r_shift;
    let mut product_form = Vec::new();
    for x in transcript.iter() {
        product_form.push(F::one() + (*x * power_2_zr));
        power_2_zr *= power_2_zr;
    }
    product_form.iter().product()
}

/// We create a polynomial using the transcript
/// This is why we need 2x srs for g2 we interleave it with zeroes
fn polynomial_coefficients_from_transcript<F: Field>(transcript: &[F], r_shift: &F) -> Vec<F> {
    let mut coefficients = vec![F::one()];
    let mut power_2_r = *r_shift;
    for (i, x) in transcript.iter().enumerate() {
        for j in 0..(2_usize).pow(i as u32) {
            coefficients.push(coefficients[j] * (*x * power_2_r));
        }
        power_2_r *= power_2_r;
    }
    // Interleave with 0 coefficients
    coefficients
        .iter()
        .interleave([F::zero()].iter().cycle().take(coefficients.len() - 1))
        .cloned()
        .collect()
}
