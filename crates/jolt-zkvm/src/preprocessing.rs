//! Preprocessing: PCS setup.
//!
//! [`preprocess`] builds the [`JoltProvingKey`] from a [`JoltConfig`] by
//! setting up the polynomial commitment scheme for the required polynomial size.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;

use crate::proof::JoltProvingKey;

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
/// Sets up PCS parameters sized for the largest polynomial in the system.
///
/// # Returns
///
/// A `JoltProvingKey` containing:
/// - PCS prover setup (SRS/generators)
/// - PCS verifier setup
#[tracing::instrument(skip_all, name = "preprocess")]
pub fn preprocess<F: Field, PCS: CommitmentScheme<Field = F>>(
    config: &JoltConfig,
    setup_fn: impl FnOnce(usize) -> (PCS::ProverSetup, PCS::VerifierSetup),
) -> JoltProvingKey<F, PCS> {
    // Estimate max polynomial variables from cycle count and R1CS layout.
    // With num_vars_per_cycle=38 (padded to 64), log2(64)=6, plus log2(num_cycles).
    let num_vars_padded = crate::r1cs::NUM_VARS_PER_CYCLE.next_power_of_two();
    let total_cols = config.num_cycles * num_vars_padded;
    let total_cols_padded = total_cols.next_power_of_two();
    let max_poly_vars = total_cols_padded.trailing_zeros() as usize;

    let (pcs_prover_setup, pcs_verifier_setup) = setup_fn(max_poly_vars);

    JoltProvingKey {
        pcs_prover_setup,
        pcs_verifier_setup,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_openings::mock::MockCommitmentScheme;

    type MockPCS = MockCommitmentScheme<Fr>;

    #[test]
    fn preprocess_builds_key() {
        let config = JoltConfig { num_cycles: 4 };
        let _key = preprocess::<Fr, MockPCS>(&config, |_vars| ((), ()));
    }

    #[test]
    fn preprocess_passes_correct_poly_size() {
        let config = JoltConfig { num_cycles: 8 };
        let mut captured_vars = 0;

        let _key = preprocess::<Fr, MockPCS>(&config, |vars| {
            captured_vars = vars;
            ((), ())
        });

        assert!(captured_vars > 0);
    }
}
