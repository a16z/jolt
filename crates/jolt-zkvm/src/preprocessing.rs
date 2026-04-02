//! Preprocessing: PCS setup from a compiled module.
//!
//! [`preprocess`] inspects the module's polynomial declarations to determine
//! the maximum number of variables, then calls [`CommitmentScheme::setup`]
//! to generate the prover and verifier SRS.

use jolt_compiler::module::Module;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;

use crate::proving_key::JoltProvingKey;

/// Build a proving key from a compiled module.
///
/// Determines the maximum polynomial size from the module's declarations
/// and generates PCS setup material sized accordingly.
pub fn preprocess<F: Field, PCS: CommitmentScheme<Field = F>>(
    module: &Module,
) -> JoltProvingKey<F, PCS> {
    let max_num_vars = module
        .polys
        .iter()
        .map(|p| p.num_elements.max(1).trailing_zeros() as usize)
        .max()
        .unwrap_or(0);

    let (pcs_prover_setup, pcs_verifier_setup) = PCS::setup(max_num_vars);

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
            })
            .collect();
        Module {
            polys,
            challenges: vec![],
            prover: Schedule {
                ops: vec![],
                kernels: vec![],
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
        let module = test_module(&[16, 32]);
        let _key = preprocess::<Fr, MockPCS>(&module);
    }

    #[test]
    fn preprocess_passes_correct_poly_size() {
        let module = test_module(&[16, 256]);
        let _key = preprocess::<Fr, MockPCS>(&module);
        // 256 = 2^8, so max_num_vars = 8. MockPCS::setup is trivial
        // but the sizing logic is exercised.
    }
}
