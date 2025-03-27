use crate::field::JoltField;

// Helper functions for small value sumcheck

// Number of rounds to use small value sumcheck for
pub const NUM_SMALL_VALUE_ROUNDS: usize = 4;

// The coefficients for the randomness, used in Algorithm 4
pub fn r_coeffs<F: JoltField>(r: F) -> [F; 3] {
    [r.square(), (F::one() - r).square(), r * (F::one() - r)]
}
