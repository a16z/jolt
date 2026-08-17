//! Metal offload seams for the Hamming-weight claim-reduction kernel: the
//! per-family pushforward selectors and the flat-mass finish the GPU
//! pushforward feeds.

use jolt_field::Field;
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_witness::witnesses::UnsignedIncLane;

use crate::metal::solinas::BooleanitySelector;
use crate::optimized::hamming_weight_claim_reduction::{FamilySelectors, HammingWeightPreparePlan};
use crate::{KernelError, SumcheckKernel};

impl FamilySelectors {
    fn metal_selectors(&self) -> Vec<BooleanitySelector> {
        self.instruction
            .iter()
            .map(|selector| BooleanitySelector::Lookup {
                shift: selector.shift() as u32,
            })
            .chain(
                self.bytecode
                    .iter()
                    .map(|selector| BooleanitySelector::Bytecode {
                        shift: selector.shift() as u32,
                    }),
            )
            .chain(self.ram.iter().map(|selector| BooleanitySelector::Ram {
                shift: selector.shift() as u32,
            }))
            .chain(self.unsigned_inc.iter().map(|lane| match lane {
                UnsignedIncLane::Chunk { width, index } => BooleanitySelector::FusedInc {
                    shift: (width * index) as u32,
                },
                UnsignedIncLane::Msb { .. } => BooleanitySelector::FusedIncMsb,
            }))
            .collect()
    }
}

impl<F: Field> HammingWeightPreparePlan<F> {
    pub(crate) fn metal_selectors(&self) -> Vec<BooleanitySelector> {
        self.selectors.metal_selectors()
    }

    pub(crate) fn finish_flat(
        self,
        flat_g_evals: Vec<F>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = HammingWeightClaimReduction<F>>>, KernelError<F>>
    {
        let expected = self
            .selectors
            .len()
            .checked_mul(self.k_chunk)
            .ok_or_else(|| KernelError::InvalidGeometry {
                reason: "Hamming-weight pushforward mass count overflows usize".to_owned(),
            })?;
        if flat_g_evals.len() != expected {
            return Err(KernelError::TableSizeMismatch {
                table: "Metal Hamming-weight pushforward masses".to_owned(),
                expected,
                got: flat_g_evals.len(),
            });
        }
        let g_evals = flat_g_evals
            .chunks_exact(self.k_chunk)
            .map(<[F]>::to_vec)
            .collect();
        self.finish(g_evals)
    }
}
