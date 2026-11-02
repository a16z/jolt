use jolt_claims::protocols::jolt::lattice::increments::FusedIncLayout;
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, TransparentObjectSetup};
use jolt_poly::Polynomial;
use jolt_witness::witnesses::FusedInc;
use jolt_witness::{collect_bundles, JoltWitnessPlane, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::akita::commitment::FusedIncObject;
use crate::KernelError;

#[derive(Clone, Copy, WitnessBundle)]
struct IncrementRow {
    increment: FusedInc,
}

/// The honest coefficient is a difference of two u64 values, so the bounded
/// dense PCS's signed 65-bit source domain contains every RV64 increment.
#[tracing::instrument(skip_all, name = "akita_fused_increment_commit")]
pub(super) fn commit_fused_inc<F, PCS>(
    setup: &PCS::ProverSetup,
    log_t: usize,
    witness: &dyn JoltWitnessPlane<F>,
) -> Result<FusedIncObject<PCS>, KernelError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup,
{
    let layout = FusedIncLayout::new(log_t);
    let cycles = 1usize << log_t;
    let mut values = if let Some(access) = witness.random_access().filter(|a| cycles <= a.cycles())
    {
        let extract = |cycle| -> Result<F, WitnessError> {
            Ok(F::from_i128(
                access.window::<IncrementRow>(cycle)?.increment.0,
            ))
        };
        let physical = access.physical_cycles().min(cycles);
        #[cfg(feature = "parallel")]
        let values = (0..physical)
            .into_par_iter()
            .map(extract)
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(not(feature = "parallel"))]
        let values = (0..physical).map(extract).collect::<Result<Vec<_>, _>>()?;
        values
    } else {
        collect_bundles::<IncrementRow>(witness, cycles)?
            .into_iter()
            .map(|row| F::from_i128(row.increment.0))
            .collect()
    };
    values.resize(1usize << layout.num_vars(), F::zero());
    let (object_setup, _) = PCS::transparent_object_setup(
        PCS::transparent_setup_context(setup),
        layout.num_vars(),
        layout.layout_digest(),
    )?;
    let (commitment, hint) = PCS::commit(&Polynomial::new(values), &object_setup)?;
    Ok(FusedIncObject { commitment, hint })
}
