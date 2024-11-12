#![allow(clippy::too_many_arguments)]

pub mod grand_product;
pub mod grand_product_quarks;
pub mod sparse_grand_product;
pub mod sumcheck;

#[derive(Clone, Copy, Debug, Default)]
pub enum QuarkHybridLayerDepth {
    #[default]
    Default,
    Min,
    Max,
    Custom(usize),
}

impl QuarkHybridLayerDepth {
    // The depth in the product tree of the grand product at which the
    // hybrid implementation will switch to using quarks grand product proofs
    pub fn get_crossover_depth(&self) -> usize {
        match self {
            QuarkHybridLayerDepth::Min => 0, // Always use quarks
            QuarkHybridLayerDepth::Default => 4,
            QuarkHybridLayerDepth::Max => usize::MAX, // Never use quarks
            QuarkHybridLayerDepth::Custom(depth) => *depth,
        }
    }
}
