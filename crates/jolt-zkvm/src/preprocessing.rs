//! Preprocessing: circuit key construction and PCS setup.
//!
//! [`preprocess`] builds the [`JoltProvingKey`] from a [`JoltConfig`] by:
//! 1. Constructing the [`UniformSpartanKey`] from the Jolt R1CS constraints
//! 2. Setting up the polynomial commitment scheme for the required polynomial size
//!
//! The preprocessing output is circuit-specific but witness-independent — it can
//! be reused across multiple prove() calls for the same circuit configuration.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_spartan::UniformSpartanKey;

use crate::proof::JoltProvingKey;
use crate::r1cs;

/// Configuration for a Jolt circuit instance.
///
/// Determines the R1CS dimensions and PCS parameter sizes.
#[derive(Clone, Debug)]
pub struct JoltConfig {
    /// Number of execution cycles (padded to next power of two internally).
    pub num_cycles: usize,
}

/// Builds a Jolt proving key from circuit configuration.
///
/// Constructs the uniform Spartan key encoding 24 constraints × 41 variables
/// per cycle, then sets up PCS parameters sized for the largest polynomial
/// in the system (the interleaved witness).
///
/// # Returns
///
/// A `JoltProvingKey` containing:
/// - Uniform Spartan key (per-cycle sparse constraint matrices)
/// - PCS prover setup (SRS/generators)
/// - PCS verifier setup
#[tracing::instrument(skip_all, name = "preprocess")]
pub fn preprocess<F: Field, PCS: CommitmentScheme<Field = F>>(
    config: &JoltConfig,
    setup_fn: impl FnOnce(usize) -> (PCS::ProverSetup, PCS::VerifierSetup),
) -> JoltProvingKey<F, PCS> {
    let spartan_key: UniformSpartanKey<F> = r1cs::build_jolt_spartan_key(config.num_cycles);

    // The largest polynomial is the interleaved witness: total_cols padded to power-of-two.
    let max_poly_vars = spartan_key.num_col_vars();

    let (pcs_prover_setup, pcs_verifier_setup) = setup_fn(max_poly_vars);

    JoltProvingKey {
        spartan_key,
        pcs_prover_setup,
        pcs_verifier_setup,
    }
}

/// Builds per-cycle witness vectors into a flat interleaved witness polynomial.
///
/// The flat witness has `total_cols_padded` elements, with each cycle's variables
/// occupying a contiguous block of `num_vars_padded` elements. Variable 0 in each
/// cycle is the constant 1 (set by the caller).
///
/// # Arguments
///
/// * `key` — Uniform Spartan key (defines dimensions)
/// * `cycle_witnesses` — Per-cycle variable vectors, each of length `key.num_vars`
///
/// # Returns
///
/// Flat interleaved witness suitable for `UniformSpartanStage::prove()`.
#[tracing::instrument(skip_all, name = "interleave_witnesses")]
pub fn interleave_witnesses<F: Field>(
    key: &UniformSpartanKey<F>,
    cycle_witnesses: &[Vec<F>],
) -> Vec<F> {
    let total_cols_padded = key.total_cols().next_power_of_two();
    let mut flat = vec![F::zero(); total_cols_padded];

    for (c, w) in cycle_witnesses.iter().enumerate() {
        let base = c * key.num_vars_padded;
        for (v, &val) in w.iter().enumerate().take(key.num_vars) {
            flat[base + v] = val;
        }
    }

    flat
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_openings::mock::MockCommitmentScheme;

    type MockPCS = MockCommitmentScheme<Fr>;

    #[test]
    fn preprocess_builds_correct_key_dimensions() {
        let config = JoltConfig { num_cycles: 4 };
        let key = preprocess::<Fr, MockPCS>(&config, |_vars| ((), ()));

        assert_eq!(
            key.spartan_key.num_constraints,
            r1cs::NUM_CONSTRAINTS_PER_CYCLE
        );
        assert_eq!(key.spartan_key.num_vars, r1cs::NUM_VARS_PER_CYCLE);
        assert_eq!(key.spartan_key.num_cycles, 4);
    }

    #[test]
    fn preprocess_passes_correct_poly_size() {
        let config = JoltConfig { num_cycles: 8 };
        let mut captured_vars = 0;

        let _key = preprocess::<Fr, MockPCS>(&config, |vars| {
            captured_vars = vars;
            ((), ())
        });

        // num_col_vars = log2(num_cycles) + log2(num_vars_padded)
        // With 8 cycles and 41 vars (padded to 64), that's 3 + 6 = 9
        assert!(captured_vars > 0);
        assert_eq!(captured_vars, _key.spartan_key.num_col_vars());
    }

    #[test]
    fn interleave_witnesses_layout() {
        let config = JoltConfig { num_cycles: 2 };
        let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ())).spartan_key;

        let w0: Vec<Fr> = (0..key.num_vars).map(|i| Fr::from_u64(i as u64)).collect();
        let w1: Vec<Fr> = (0..key.num_vars)
            .map(|i| Fr::from_u64(100 + i as u64))
            .collect();

        let flat = interleave_witnesses(&key, &[w0.clone(), w1.clone()]);

        // Cycle 0 variables at [0..num_vars]
        assert_eq!(&flat[..key.num_vars], &w0[..]);

        // Cycle 1 variables at [num_vars_padded..num_vars_padded+num_vars]
        let c1_start = key.num_vars_padded;
        assert_eq!(&flat[c1_start..c1_start + key.num_vars], &w1[..]);

        // Padding between cycles should be zero
        for &val in &flat[key.num_vars..key.num_vars_padded] {
            assert_eq!(val, Fr::from_u64(0));
        }
    }

    #[test]
    fn interleave_empty_cycles() {
        let config = JoltConfig { num_cycles: 1 };
        let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ())).spartan_key;

        let w: Vec<Fr> = (0..key.num_vars).map(|_| Fr::from_u64(1)).collect();
        let flat = interleave_witnesses(&key, &[w]);

        assert!(flat.len().is_power_of_two());
        assert!(flat.len() >= key.num_vars);
    }
}
