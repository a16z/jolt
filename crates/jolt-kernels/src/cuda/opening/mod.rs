//! Initial-RAM advice evaluation uploads the oracle table and reduces it on the
//! GPU, while program-image evaluation uses the canonical host helper.
//!
//! CUDA builds the equality weights and sums their products with the advice
//! values on device; optimized performs the same computation on the host.

use jolt_claims::protocols::jolt::geometry::claim_reductions::advice::ram_val_check_advice_opening;
use jolt_field::JoltField;
use jolt_witness::JoltWitnessPlane;

use super::common::device::{fr_into, require_fr_slice};
use super::{require_context, CudaBackend};
use crate::opening::{RamInitialOpening, RamInitialOpeningEvaluation};
use crate::reference::views::dense_view;
use crate::{KernelError, ProofSession};

impl<F: JoltField> RamInitialOpeningEvaluation<F> for CudaBackend {
    fn evaluate(
        &self,
        _session: &mut ProofSession,
        openings: &[RamInitialOpening<'_, F>],
        witness: &dyn JoltWitnessPlane<F>,
    ) -> Result<Vec<F>, KernelError<F>> {
        let context = require_context::<F>()?;
        openings
            .iter()
            .map(|opening| {
                let (table, point, scale) = match opening {
                    RamInitialOpening::Advice { kind, point } => (
                        dense_view(witness, ram_val_check_advice_opening(*kind))?,
                        *point,
                        F::one(),
                    ),
                    RamInitialOpening::ProgramImage { layout, point } => {
                        // Main owns sparse segment geometry, including non-aligned image starts.
                        return Ok(crate::opening::evaluate_program_image(
                            layout, point, witness,
                        ));
                    }
                };
                if table.len() != 1usize << point.len() {
                    return Err(KernelError::TableSizeMismatch {
                        table: "initial RAM opening".to_owned(),
                        expected: 1usize << point.len(),
                        got: table.len(),
                    });
                }
                let device = context.upload(require_fr_slice(&table)?)?;
                let eq = context.eq_evals(require_fr_slice(point)?)?;
                let products = context.mul(&device, &eq)?;
                let value = context.sum(&products)?;
                fr_into::<F>(value)
                    .map(|value| value * scale)
                    .ok_or(KernelError::Unsupported {
                        reason: "CUDA kernels support only the BN254 scalar field",
                    })
            })
            .collect()
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::JoltAdviceKind;
    use jolt_field::{Fr, Ring};
    use jolt_witness::JoltWitnessPlane;
    use proptest::prelude::*;

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{advice_plane, fr};
    use crate::opening::{RamInitialOpening, RamInitialOpeningEvaluation};
    use crate::reference::ReferenceBackend;
    use crate::ProofSession;

    const ADVICE_BYTES: usize = 4096;

    const ADVICE_KINDS: [JoltAdviceKind; 2] = [JoltAdviceKind::Trusted, JoltAdviceKind::Untrusted];

    fn advice_opening(
        backend: &dyn RamInitialOpeningEvaluation<Fr>,
        witness: &dyn JoltWitnessPlane<Fr>,
        kind: JoltAdviceKind,
        point: &[Fr],
    ) -> Fr {
        backend
            .evaluate(
                &mut ProofSession::default(),
                &[RamInitialOpening::Advice { kind, point }],
                witness,
            )
            .expect("advice opening")[0]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2))]

        #[test]
        fn advice_opening_matches_reference(seed in any::<u64>()) {
            let Some(_) = shared_context() else { return Ok(()); };
            let fixture = advice_plane(ADVICE_BYTES, seed);
            let vars = fixture.trusted.len().ilog2() as usize;
            let witness = &fixture.plane as &dyn JoltWitnessPlane<Fr>;
            let point: Vec<Fr> = (0..vars)
                .map(|index| fr(seed ^ (index as u64).wrapping_mul(17).wrapping_add(5)))
                .collect();
            let expected: Vec<Fr> = ADVICE_KINDS
                .iter()
                .map(|&kind| advice_opening(&ReferenceBackend, witness, kind, &point))
                .collect();
            for (&kind, &expected) in ADVICE_KINDS.iter().zip(&expected) {
                let got = advice_opening(&CudaBackend, witness, kind, &point);
                prop_assert_eq!(got, expected, "{:?} advice opening diverged", kind);
            }
        }
    }

    #[test]
    fn fixture_advice_columns_discriminate() {
        let fixture = advice_plane(ADVICE_BYTES, 41);
        assert_eq!(
            fixture.trusted.len(),
            fixture.untrusted.len(),
            "the two advice columns must share a width for one point to serve both",
        );
        assert_ne!(
            fixture.trusted, fixture.untrusted,
            "the two advice columns are identical, so a kernel that ignored the kind would pass",
        );
        assert!(
            fixture
                .trusted
                .iter()
                .any(|value| *value != Fr::from_u64(0)),
            "the trusted advice column is all zeros",
        );
        assert!(
            fixture.trusted.len().is_power_of_two() && fixture.trusted.len() > 1,
            "the advice column must span at least one variable",
        );
    }
}
