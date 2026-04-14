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

struct MulShared {
    field: Fr,
    u64_scalar: u64,
    i64_scalar: i64,
    u128_scalar: u128,
    i128_scalar: i128,
}

impl MulShared {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            field: Fr::random(&mut rng),
            u64_scalar: rand::Rng::gen(&mut rng),
            i64_scalar: rand::Rng::gen(&mut rng),
            u128_scalar: rand::Rng::gen(&mut rng),
            i128_scalar: rand::Rng::gen(&mut rng),
        }
    }
}

// -- mul_u64 --

pub struct MulU64Setup {
    pub field: Fr,
    pub scalar: u64,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MulU64Objective;

impl Objective for MulU64Objective {
    type Setup = MulU64Setup;

    fn name(&self) -> &str {
        "field_mul_u64"
    }

    fn description(&self) -> String {
        format!("Wall-clock time of Field::mul_u64 ({NUM_ITERS} iterations)")
    }

    fn setup(&self) -> MulU64Setup {
        thread_local! { static SHARED: MulShared = MulShared::new(); }
        SHARED.with(|s| MulU64Setup {
            field: s.field,
            scalar: s.u64_scalar,
        })
    }

    fn run(&self, setup: MulU64Setup) {
        let mut acc = setup.field;
        for _ in 0..NUM_ITERS {
            acc = acc.mul_u64(setup.scalar);
        }
        std::hint::black_box(acc);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}

// -- mul_i64 --

pub struct MulI64Setup {
    pub field: Fr,
    pub scalar: i64,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MulI64Objective;

impl Objective for MulI64Objective {
    type Setup = MulI64Setup;

    fn name(&self) -> &str {
        "field_mul_i64"
    }

    fn description(&self) -> String {
        format!("Wall-clock time of Field::mul_i64 ({NUM_ITERS} iterations)")
    }

    fn setup(&self) -> MulI64Setup {
        thread_local! { static SHARED: MulShared = MulShared::new(); }
        SHARED.with(|s| MulI64Setup {
            field: s.field,
            scalar: s.i64_scalar,
        })
    }

    fn run(&self, setup: MulI64Setup) {
        let mut acc = setup.field;
        for _ in 0..NUM_ITERS {
            acc = acc.mul_i64(setup.scalar);
        }
        std::hint::black_box(acc);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}

// -- mul_u128 --

pub struct MulU128Setup {
    pub field: Fr,
    pub scalar: u128,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MulU128Objective;

impl Objective for MulU128Objective {
    type Setup = MulU128Setup;

    fn name(&self) -> &str {
        "field_mul_u128"
    }

    fn description(&self) -> String {
        format!("Wall-clock time of Field::mul_u128 ({NUM_ITERS} iterations)")
    }

    fn setup(&self) -> MulU128Setup {
        thread_local! { static SHARED: MulShared = MulShared::new(); }
        SHARED.with(|s| MulU128Setup {
            field: s.field,
            scalar: s.u128_scalar,
        })
    }

    fn run(&self, setup: MulU128Setup) {
        let mut acc = setup.field;
        for _ in 0..NUM_ITERS {
            acc = acc.mul_u128(setup.scalar);
        }
        std::hint::black_box(acc);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}

// -- mul_i128 --

pub struct MulI128Setup {
    pub field: Fr,
    pub scalar: i128,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MulI128Objective;

impl Objective for MulI128Objective {
    type Setup = MulI128Setup;

    fn name(&self) -> &str {
        "field_mul_i128"
    }

    fn description(&self) -> String {
        format!("Wall-clock time of Field::mul_i128 ({NUM_ITERS} iterations)")
    }

    fn setup(&self) -> MulI128Setup {
        thread_local! { static SHARED: MulShared = MulShared::new(); }
        SHARED.with(|s| MulI128Setup {
            field: s.field,
            scalar: s.i128_scalar,
        })
    }

    fn run(&self, setup: MulI128Setup) {
        let mut acc = setup.field;
        for _ in 0..NUM_ITERS {
            acc = acc.mul_i128(setup.scalar);
        }
        std::hint::black_box(acc);
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}
