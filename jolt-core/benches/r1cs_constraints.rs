use ark_ff::biginteger::{S128, S64};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jolt_core::field::tracked_ark::TrackedFr;
use jolt_core::utils::small_scalar::SmallScalar;
use jolt_core::zkvm::instruction::CircuitFlags;
use jolt_core::zkvm::r1cs::constraints::{eval_az_by_name, eval_bz_by_name, UNIFORM_R1CS};
use jolt_core::zkvm::r1cs::inputs::{JoltR1CSInputs, WitnessRowAccessor};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

/// Random test value covering the typed accessors used by named evaluators
enum TestValue {
    Bool(bool),
    U8(u8),
    U64(u64),
    S64(S64),
    U128(u128),
    S128(S128),
}

/// Random test accessor for benchmarking
struct RandomTestAccessor {
    values: Vec<TestValue>,
    rows: usize,
}

impl RandomTestAccessor {
    fn new(num_rows: usize, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let num_inputs = JoltR1CSInputs::num_inputs();
        let mut values = Vec::with_capacity(num_rows * num_inputs);

        for _row in 0..num_rows {
            for i in 0..num_inputs {
                let inp = JoltR1CSInputs::from_index(i);
                let val = match inp {
                    JoltR1CSInputs::PC => TestValue::U64(rng.next_u64() % (1 << 32)),
                    JoltR1CSInputs::UnexpandedPC => TestValue::U64(rng.next_u64() % (1 << 32)),
                    JoltR1CSInputs::Rd => TestValue::U8((rng.next_u32() % 32) as u8),
                    JoltR1CSInputs::Imm => {
                        let v = (rng.next_u64() as i64) % (1i64 << 20);
                        let mag = v.unsigned_abs();
                        TestValue::S64(S64::from_u64_with_sign(mag, v >= 0))
                    }
                    JoltR1CSInputs::RamAddress => TestValue::U64(rng.next_u64()),
                    JoltR1CSInputs::Rs1Value => TestValue::U64(rng.next_u64()),
                    JoltR1CSInputs::Rs2Value => TestValue::U64(rng.next_u64()),
                    JoltR1CSInputs::RdWriteValue => TestValue::U64(rng.next_u64()),
                    JoltR1CSInputs::RamReadValue => TestValue::U64(rng.next_u64()),
                    JoltR1CSInputs::RamWriteValue => TestValue::U64(rng.next_u64()),
                    JoltR1CSInputs::LeftInstructionInput => TestValue::U64(rng.next_u64()),
                    JoltR1CSInputs::RightInstructionInput => {
                        let v = rng.next_u64() as i64;
                        let mag = v.unsigned_abs();
                        TestValue::S64(S64::from_u64_with_sign(mag, v >= 0))
                    }
                    JoltR1CSInputs::LeftLookupOperand => TestValue::U64(rng.next_u64()),
                    JoltR1CSInputs::RightLookupOperand => {
                        TestValue::U128(((rng.next_u64() as u128) << 64) | (rng.next_u64() as u128))
                    }
                    JoltR1CSInputs::Product => {
                        // Random s128
                        let hi = rng.next_u64() as i128;
                        let lo = rng.next_u64() as i128;
                        let v = (hi << 64) ^ lo;
                        TestValue::S128(S128::from_i128(v))
                    }
                    JoltR1CSInputs::WriteLookupOutputToRD => {
                        TestValue::U8((rng.next_u32() % 2) as u8)
                    }
                    JoltR1CSInputs::WritePCtoRD => TestValue::U8((rng.next_u32() % 2) as u8),
                    JoltR1CSInputs::ShouldBranch => TestValue::U8((rng.next_u32() % 2) as u8),
                    JoltR1CSInputs::NextUnexpandedPC => TestValue::U64(rng.next_u64() % (1 << 32)),
                    JoltR1CSInputs::NextPC => TestValue::U64(rng.next_u64() % (1 << 32)),
                    JoltR1CSInputs::LookupOutput => TestValue::U64(rng.next_u64()),
                    JoltR1CSInputs::NextIsNoop => TestValue::Bool(rng.next_u32() % 2 == 0),
                    JoltR1CSInputs::ShouldJump => TestValue::Bool(rng.next_u32() % 2 == 0),
                    JoltR1CSInputs::OpFlags(flag) => {
                        let b = match flag {
                            CircuitFlags::LeftOperandIsRs1Value => rng.next_u32() % 2 == 0,
                            CircuitFlags::RightOperandIsRs2Value => rng.next_u32() % 2 == 0,
                            CircuitFlags::LeftOperandIsPC => rng.next_u32() % 2 == 0,
                            CircuitFlags::RightOperandIsImm => rng.next_u32() % 2 == 0,
                            CircuitFlags::AddOperands => rng.next_u32() % 3 == 0,
                            CircuitFlags::SubtractOperands => rng.next_u32() % 3 == 1,
                            CircuitFlags::MultiplyOperands => rng.next_u32() % 3 == 2,
                            CircuitFlags::Load => rng.next_u32() % 4 == 0,
                            CircuitFlags::Store => rng.next_u32() % 4 == 1,
                            CircuitFlags::Jump => rng.next_u32() % 4 == 2,
                            CircuitFlags::Branch => rng.next_u32() % 4 == 3,
                            CircuitFlags::WriteLookupOutputToRD => rng.next_u32() % 2 == 0,
                            CircuitFlags::InlineSequenceInstruction => rng.next_u32() % 10 == 0,
                            CircuitFlags::Assert => rng.next_u32() % 20 == 0,
                            CircuitFlags::DoNotUpdateUnexpandedPC => rng.next_u32() % 5 == 0,
                            CircuitFlags::Advice => rng.next_u32() % 20 == 0,
                            CircuitFlags::IsNoop => rng.next_u32() % 10 == 0,
                            CircuitFlags::IsCompressed => rng.next_u32() % 3 == 0,
                        };
                        TestValue::Bool(b)
                    }
                };
                values.push(val);
            }
        }

        RandomTestAccessor {
            values,
            rows: num_rows,
        }
    }
}

impl RandomTestAccessor {
    #[inline]
    fn get(&self, input_index: JoltR1CSInputs, row: usize) -> &TestValue {
        let idx = row * JoltR1CSInputs::num_inputs() + input_index.to_index();
        &self.values[idx]
    }
}

impl WitnessRowAccessor<TrackedFr, JoltR1CSInputs> for RandomTestAccessor {
    fn num_steps(&self) -> usize {
        self.rows
    }

    fn value_at_field(&self, input_index: JoltR1CSInputs, row: usize) -> TrackedFr {
        match self.get(input_index, row) {
            TestValue::Bool(v) => v.to_field(),
            TestValue::U8(v) => v.to_field(),
            TestValue::U64(v) => v.to_field(),
            TestValue::S64(v) => v.to_field(),
            TestValue::U128(v) => v.to_field(),
            TestValue::S128(v) => (*v).to_field(),
        }
    }

    fn value_at_bool(&self, input_index: JoltR1CSInputs, row: usize) -> bool {
        match self.get(input_index, row) {
            TestValue::Bool(v) => *v,
            TestValue::U8(v) => *v != 0,
            TestValue::U64(v) => *v != 0,
            _ => false,
        }
    }

    fn value_at_u8(&self, input_index: JoltR1CSInputs, row: usize) -> u8 {
        match self.get(input_index, row) {
            TestValue::U8(v) => *v,
            TestValue::Bool(v) => {
                if *v {
                    1
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    fn value_at_u64(&self, input_index: JoltR1CSInputs, row: usize) -> u64 {
        match self.get(input_index, row) {
            TestValue::U64(v) => *v,
            TestValue::U8(v) => *v as u64,
            TestValue::Bool(v) => {
                if *v {
                    1
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    fn value_at_s64(&self, input_index: JoltR1CSInputs, row: usize) -> S64 {
        match self.get(input_index, row) {
            TestValue::S64(v) => *v,
            _ => S64::from_u64_with_sign(0, true),
        }
    }

    fn value_at_u128(&self, input_index: JoltR1CSInputs, row: usize) -> u128 {
        match self.get(input_index, row) {
            TestValue::U128(v) => *v,
            _ => 0,
        }
    }

    fn value_at_s128(&self, input_index: JoltR1CSInputs, row: usize) -> S128 {
        match self.get(input_index, row) {
            TestValue::S128(v) => *v,
            _ => S128::from_i128(0),
        }
    }
}

fn bench_az_comparison(c: &mut Criterion) {
    let accessor = RandomTestAccessor::new(1000, 12345);
    let accessor_ref: &dyn WitnessRowAccessor<TrackedFr, JoltR1CSInputs> = &accessor;

    c.bench_function("az_evaluation_named", |b| {
        b.iter(|| {
            for constraint in UNIFORM_R1CS.iter() {
                for row in 0..black_box(10) {
                    let _result = eval_az_by_name::<TrackedFr>(
                        black_box(constraint),
                        black_box(accessor_ref),
                        black_box(row),
                    );
                }
            }
        })
    });

    // Compare against generic LC evaluation over the field
    c.bench_function("az_evaluation_generic_field", |b| {
        b.iter(|| {
            for constraint in UNIFORM_R1CS.iter() {
                for row in 0..black_box(10) {
                    let _result = constraint
                        .cons
                        .a
                        .evaluate_row_with::<TrackedFr>(black_box(accessor_ref), black_box(row));
                }
            }
        })
    });
}

fn bench_bz_comparison(c: &mut Criterion) {
    let accessor = RandomTestAccessor::new(1000, 54321);
    let accessor_ref: &dyn WitnessRowAccessor<TrackedFr, JoltR1CSInputs> = &accessor;

    c.bench_function("bz_evaluation_named", |b| {
        b.iter(|| {
            for constraint in UNIFORM_R1CS.iter() {
                for row in 0..black_box(10) {
                    let _result = eval_bz_by_name::<TrackedFr>(
                        black_box(constraint),
                        black_box(accessor_ref),
                        black_box(row),
                    );
                }
            }
        })
    });

    // Compare against generic LC evaluation over the field
    c.bench_function("bz_evaluation_generic_field", |b| {
        b.iter(|| {
            for constraint in UNIFORM_R1CS.iter() {
                for row in 0..black_box(10) {
                    let _result = constraint
                        .cons
                        .b
                        .evaluate_row_with::<TrackedFr>(black_box(accessor_ref), black_box(row));
                }
            }
        })
    });
}

fn bench_single_constraint_comparison(c: &mut Criterion) {
    let accessor = RandomTestAccessor::new(1000, 98765);
    let accessor_ref: &dyn WitnessRowAccessor<TrackedFr, JoltR1CSInputs> = &accessor;

    // Test a few representative constraints
    let test_constraints = [0, UNIFORM_R1CS.len() / 2, UNIFORM_R1CS.len() - 1];

    for &idx in &test_constraints {
        let constraint = &UNIFORM_R1CS[idx];

        c.bench_function(&format!("single_constraint_{}_az_named", idx), |b| {
            b.iter(|| {
                for row in 0..black_box(100) {
                    let _result = eval_az_by_name::<TrackedFr>(
                        black_box(constraint),
                        black_box(accessor_ref),
                        black_box(row),
                    );
                }
            })
        });

        c.bench_function(
            &format!("single_constraint_{}_az_generic_field", idx),
            |b| {
                b.iter(|| {
                    for row in 0..black_box(100) {
                        let _result = constraint.cons.a.evaluate_row_with::<TrackedFr>(
                            black_box(accessor_ref),
                            black_box(row),
                        );
                    }
                })
            },
        );

        c.bench_function(&format!("single_constraint_{}_bz_named", idx), |b| {
            b.iter(|| {
                for row in 0..black_box(100) {
                    let _result = eval_bz_by_name::<TrackedFr>(
                        black_box(constraint),
                        black_box(accessor_ref),
                        black_box(row),
                    );
                }
            })
        });

        c.bench_function(
            &format!("single_constraint_{}_bz_generic_field", idx),
            |b| {
                b.iter(|| {
                    for row in 0..black_box(100) {
                        let _result = constraint.cons.b.evaluate_row_with::<TrackedFr>(
                            black_box(accessor_ref),
                            black_box(row),
                        );
                    }
                })
            },
        );
    }
}

fn bench_az_evaluation(c: &mut Criterion) {
    let accessor = RandomTestAccessor::new(1000, 12345);
    let accessor_ref: &dyn WitnessRowAccessor<TrackedFr, JoltR1CSInputs> = &accessor;

    c.bench_function("az_evaluation_all_constraints", |b| {
        b.iter(|| {
            for constraint in UNIFORM_R1CS.iter() {
                for row in 0..black_box(10) {
                    let _result = eval_az_by_name::<TrackedFr>(
                        black_box(constraint),
                        black_box(accessor_ref),
                        black_box(row),
                    );
                }
            }
        })
    });
}

fn bench_bz_evaluation(c: &mut Criterion) {
    let accessor = RandomTestAccessor::new(1000, 54321);
    let accessor_ref: &dyn WitnessRowAccessor<TrackedFr, JoltR1CSInputs> = &accessor;

    c.bench_function("bz_evaluation_all_constraints", |b| {
        b.iter(|| {
            for constraint in UNIFORM_R1CS.iter() {
                for row in 0..black_box(10) {
                    let _result = eval_bz_by_name::<TrackedFr>(
                        black_box(constraint),
                        black_box(accessor_ref),
                        black_box(row),
                    );
                }
            }
        })
    });
}

fn bench_constraint_evaluation_single(c: &mut Criterion) {
    let accessor = RandomTestAccessor::new(1000, 98765);
    let accessor_ref: &dyn WitnessRowAccessor<TrackedFr, JoltR1CSInputs> = &accessor;

    // Benchmark a few representative constraints individually
    let representative_constraints = [
        &UNIFORM_R1CS[0],                      // First constraint
        &UNIFORM_R1CS[UNIFORM_R1CS.len() / 2], // Middle constraint
        &UNIFORM_R1CS[UNIFORM_R1CS.len() - 1], // Last constraint
    ];

    for (i, constraint) in representative_constraints.iter().enumerate() {
        c.bench_function(&format!("single_constraint_az_{}", i), |b| {
            b.iter(|| {
                for row in 0..black_box(100) {
                    let _result = eval_az_by_name::<TrackedFr>(
                        black_box(constraint),
                        black_box(accessor_ref),
                        black_box(row),
                    );
                }
            })
        });

        c.bench_function(&format!("single_constraint_bz_{}", i), |b| {
            b.iter(|| {
                for row in 0..black_box(100) {
                    let _result = eval_bz_by_name::<TrackedFr>(
                        black_box(constraint),
                        black_box(accessor_ref),
                        black_box(row),
                    );
                }
            })
        });
    }
}

fn bench_constraint_scaling(c: &mut Criterion) {
    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        let accessor = RandomTestAccessor::new(size, 11111);
        let accessor_ref: &dyn WitnessRowAccessor<TrackedFr, JoltR1CSInputs> = &accessor;

        c.bench_function(&format!("az_evaluation_size_{}", size), |b| {
            b.iter(|| {
                // Evaluate first 5 constraints on all rows
                for constraint in UNIFORM_R1CS.iter().take(5) {
                    for row in 0..black_box(size.min(100)) {
                        let _result = eval_az_by_name::<TrackedFr>(
                            black_box(constraint),
                            black_box(accessor_ref),
                            black_box(row),
                        );
                    }
                }
            })
        });
    }
}

criterion_group!(
    benches,
    bench_az_comparison,
    bench_bz_comparison,
    bench_single_constraint_comparison,
    bench_az_evaluation,
    bench_bz_evaluation,
    bench_constraint_evaluation_single,
    bench_constraint_scaling
);
criterion_main!(benches);
