use std::sync::Arc;

use jolt_kernels::optimized::hamming_weight_claim_reduction::bench::HammingWeightPushforwardFixture;

use crate::objective::{Objective, OptimizationObjective, PerformanceObjective};

pub const HAMMING_WEIGHT_PUSHFORWARD: OptimizationObjective = OptimizationObjective::Performance(
    PerformanceObjective::HammingWeightPushforward(HammingWeightPushforwardObjective),
);

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct HammingWeightPushforwardObjective;

impl Objective for HammingWeightPushforwardObjective {
    type Setup = Arc<HammingWeightPushforwardFixture>;

    fn name(&self) -> &str {
        "hamming_weight_pushforward"
    }

    fn description(&self) -> String {
        "Stage-7 one-hot pushforward at 2^22 cycles with the production 8-bit chunk and 16/2/3 RA-family geometry".to_string()
    }

    fn setup(&self) -> Self::Setup {
        thread_local! {
            static FIXTURE: Arc<HammingWeightPushforwardFixture> =
                Arc::new(HammingWeightPushforwardFixture::production_geometry());
        }
        FIXTURE.with(Arc::clone)
    }

    fn run(&self, setup: Self::Setup) {
        std::hint::black_box(setup.compute());
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}
