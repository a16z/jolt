use jolt_field::arkworks::bn254::Fr;
use jolt_field::Field;

use crate::objective::{Objective, OptimizationObjective, PerformanceObjective};

pub const MUL_U64: OptimizationObjective =
    OptimizationObjective::Performance(PerformanceObjective::MulU64(MulU64Objective));
pub const MUL_I64: OptimizationObjective =
    OptimizationObjective::Performance(PerformanceObjective::MulI64(MulI64Objective));
pub const MUL_U128: OptimizationObjective =
    OptimizationObjective::Performance(PerformanceObjective::MulU128(MulU128Objective));
pub const MUL_I128: OptimizationObjective =
    OptimizationObjective::Performance(PerformanceObjective::MulI128(MulI128Objective));

const NUM_ITERS: usize = 10_000;

// -- mul_u64 --

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MulU64Objective;

impl Objective for MulU64Objective {
    type Setup = (Fr, u64);

    fn name(&self) -> &str {
        "field_mul_u64"
    }

    fn description(&self) -> String {
        format!("Wall-clock time of Field::mul_u64 ({NUM_ITERS} iterations)")
    }

    fn setup(&self) -> Self::Setup {
        let mut rng = rand::thread_rng();
        (Fr::random(&mut rng), rand::Rng::gen(&mut rng))
    }

    fn run(&self, setup: Self::Setup) {
        let (field, scalar) = setup;
        let mut acc = field;
        for _ in 0..NUM_ITERS {
            acc = acc.mul_u64(scalar);
        }
        std::hint::black_box(acc);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}

// -- mul_i64 --

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MulI64Objective;

impl Objective for MulI64Objective {
    type Setup = (Fr, i64);

    fn name(&self) -> &str {
        "field_mul_i64"
    }

    fn description(&self) -> String {
        format!("Wall-clock time of Field::mul_i64 ({NUM_ITERS} iterations)")
    }

    fn setup(&self) -> Self::Setup {
        let mut rng = rand::thread_rng();
        (Fr::random(&mut rng), rand::Rng::gen(&mut rng))
    }

    fn run(&self, setup: Self::Setup) {
        let (field, scalar) = setup;
        let mut acc = field;
        for _ in 0..NUM_ITERS {
            acc = acc.mul_i64(scalar);
        }
        std::hint::black_box(acc);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}

// -- mul_u128 --

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MulU128Objective;

impl Objective for MulU128Objective {
    type Setup = (Fr, u128);

    fn name(&self) -> &str {
        "field_mul_u128"
    }

    fn description(&self) -> String {
        format!("Wall-clock time of Field::mul_u128 ({NUM_ITERS} iterations)")
    }

    fn setup(&self) -> Self::Setup {
        let mut rng = rand::thread_rng();
        (Fr::random(&mut rng), rand::Rng::gen(&mut rng))
    }

    fn run(&self, setup: Self::Setup) {
        let (field, scalar) = setup;
        let mut acc = field;
        for _ in 0..NUM_ITERS {
            acc = acc.mul_u128(scalar);
        }
        std::hint::black_box(acc);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}

// -- mul_i128 --

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MulI128Objective;

impl Objective for MulI128Objective {
    type Setup = (Fr, i128);

    fn name(&self) -> &str {
        "field_mul_i128"
    }

    fn description(&self) -> String {
        format!("Wall-clock time of Field::mul_i128 ({NUM_ITERS} iterations)")
    }

    fn setup(&self) -> Self::Setup {
        let mut rng = rand::thread_rng();
        (Fr::random(&mut rng), rand::Rng::gen(&mut rng))
    }

    fn run(&self, setup: Self::Setup) {
        let (field, scalar) = setup;
        let mut acc = field;
        for _ in 0..NUM_ITERS {
            acc = acc.mul_i128(scalar);
        }
        std::hint::black_box(acc);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}
