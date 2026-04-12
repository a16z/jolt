//! Preprocessing: PCS setup from a compiled module.
//!
//! [`preprocess`] inspects the module's polynomial declarations to determine
//! the maximum number of variables, then calls [`CommitmentScheme::setup`]
//! to generate the prover and verifier SRS.

use jolt_compiler::module::Module;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;

use crate::proving_key::JoltProvingKey;

/// Computes the maximum number of variables across all polynomials in a module.
pub fn max_num_vars(module: &Module) -> usize {
    module
        .polys
        .iter()
        .map(|p| p.num_elements.max(1).trailing_zeros() as usize)
        .max()
        .unwrap_or(0)
}

/// Build a proving key from a compiled module.
///
/// The caller provides scheme-specific setup parameters. Use
/// [`max_num_vars`] to compute the maximum polynomial size from the module
/// if needed for constructing `PCS::SetupParams`.
pub fn preprocess<F: Field, PCS: CommitmentScheme<Field = F>>(
    setup_params: PCS::SetupParams,
) -> JoltProvingKey<F, PCS> {
    let (pcs_prover_setup, pcs_verifier_setup) = PCS::setup(setup_params);

    JoltProvingKey {
        pcs_prover_setup,
        pcs_verifier_setup,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_compiler::ir::PolyKind;
    use jolt_compiler::module::{PolyDecl, Schedule, VerifierSchedule};
    use jolt_field::Fr;
    use jolt_openings::mock::MockCommitmentScheme;

    type MockPCS = MockCommitmentScheme<Fr>;

    fn test_module(poly_sizes: &[usize]) -> Module {
        let polys = poly_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| PolyDecl {
                name: format!("p{i}"),
                kind: PolyKind::Committed,
                num_elements: size,
                committed_num_vars: None,
            })
            .collect();
        Module {
            polys,
            challenges: vec![],
            prover: Schedule {
                ops: vec![],
                kernels: vec![],
                batched_sumchecks: vec![],
            },
            verifier: VerifierSchedule {
                ops: vec![],
                num_challenges: 0,
                num_polys: poly_sizes.len(),
                num_stages: 0,
            },
        }
    }

    #[test]
    fn preprocess_builds_key() {
        let _key = preprocess::<Fr, MockPCS>(());
    }

    #[test]
    fn max_num_vars_from_module() {
        let module = test_module(&[16, 256]);
        assert_eq!(max_num_vars(&module), 8); // 256 = 2^8
    }
}
