use super::constants::get_frobenius_coefficients;
use ark_bn254::G2Projective;
use ark_std::Zero;

/// Compute the Frobenius endomorphism ψ^k on a BN254 G2 point (projective).
/// Applies conjugation in Fq2 followed by multiplication by precomputed coefficients.
pub fn frobenius_psi_power_projective(p: &G2Projective, k: usize) -> G2Projective {
    if p.is_zero() {
        return *p;
    }

    let mut res = *p;
    let coeffs = get_frobenius_coefficients();

    // Odd power: apply conjugation in Fq2
    if (k & 1) == 1 {
        let _ = res.x.conjugate_in_place();
        let _ = res.y.conjugate_in_place();
        let _ = res.z.conjugate_in_place();
    }

    match k % 4 {
        0 => res,
        1 => {
            res.x *= coeffs.psi1_coef2;
            res.y *= coeffs.psi1_coef3;
            res
        }
        2 => {
            res.x *= coeffs.psi2_coef2;
            res.y *= coeffs.psi2_coef3;
            res
        }
        3 => {
            res.x *= coeffs.psi3_coef2;
            res.y *= coeffs.psi3_coef3;
            res
        }
        _ => unreachable!(),
    }
}
