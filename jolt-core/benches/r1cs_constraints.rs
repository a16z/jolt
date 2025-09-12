use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jolt_core::field::tracked_ark::TrackedFr;
use jolt_core::utils::small_scalar::SmallScalar;
use jolt_core::zkvm::instruction::CircuitFlags;
use jolt_core::zkvm::r1cs::constraints::{eval_az_by_name, eval_bz_by_name, UNIFORM_R1CS};
use jolt_core::zkvm::r1cs::inputs::{
    eval_az_generic, eval_bz_generic, JoltR1CSInputs, WitnessRowAccessor,
};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

/// Random test accessor for benchmarking
struct RandomTestAccessor {
    values: Vec<SmallScalar>,
    rows: usize,
}

impl RandomTestAccessor {
    fn new(num_rows: usize, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let num_inputs = JoltR1CSInputs::num_inputs();
        let mut values = vec![SmallScalar::U8(0); num_rows * num_inputs];

        for row in 0..num_rows {
            // Generate random but reasonable values for each input
            for i in 0..num_inputs {
                let inp = JoltR1CSInputs::from_index(i);
                let val = match inp {
                    JoltR1CSInputs::PC => SmallScalar::U64(rng.next_u64() % (1 << 32)),
                    JoltR1CSInputs::UnexpandedPC => SmallScalar::U64(rng.next_u64() % (1 << 32)),
                    JoltR1CSInputs::Rd => SmallScalar::U8((rng.next_u32() % 32) as u8),
                    JoltR1CSInputs::Imm => {
                        SmallScalar::I128(((rng.next_u64() as i64) % (1i64 << 20)) as i128)
                    }
                    JoltR1CSInputs::RamAddress => SmallScalar::U64(rng.next_u64()),
                    JoltR1CSInputs::Rs1Value => SmallScalar::U64(rng.next_u64()),
                    JoltR1CSInputs::Rs2Value => SmallScalar::U64(rng.next_u64()),
                    JoltR1CSInputs::RdWriteValue => SmallScalar::U64(rng.next_u64()),
                    JoltR1CSInputs::RamReadValue => SmallScalar::U64(rng.next_u64()),
                    JoltR1CSInputs::RamWriteValue => SmallScalar::U64(rng.next_u64()),
                    JoltR1CSInputs::LeftInstructionInput => SmallScalar::U64(rng.next_u64()),
                    JoltR1CSInputs::RightInstructionInput => {
                        SmallScalar::I64(rng.next_u64() as i64)
                    }
                    JoltR1CSInputs::LeftLookupOperand => SmallScalar::U64(rng.next_u64()),
                    JoltR1CSInputs::RightLookupOperand => SmallScalar::U128(
                        ((rng.next_u64() as u128) << 64) | (rng.next_u64() as u128),
                    ),
                    JoltR1CSInputs::Product => SmallScalar::U128(
                        ((rng.next_u64() as u128) << 64) | (rng.next_u64() as u128),
                    ),
                    JoltR1CSInputs::WriteLookupOutputToRD => {
                        SmallScalar::U8(rng.next_u32() as u8 % 2)
                    }
                    JoltR1CSInputs::WritePCtoRD => SmallScalar::U8(rng.next_u32() as u8 % 2),
                    JoltR1CSInputs::ShouldBranch => SmallScalar::U8(rng.next_u32() as u8 % 2),
                    JoltR1CSInputs::NextUnexpandedPC => {
                        SmallScalar::U64(rng.next_u64() % (1 << 32))
                    }
                    JoltR1CSInputs::NextPC => SmallScalar::U64(rng.next_u64() % (1 << 32)),
                    JoltR1CSInputs::LookupOutput => SmallScalar::U64(rng.next_u64()),
                    JoltR1CSInputs::NextIsNoop => SmallScalar::Bool(rng.next_u32() % 2 == 0),
                    JoltR1CSInputs::ShouldJump => SmallScalar::U8(rng.next_u32() as u8 % 2),
                    JoltR1CSInputs::CompressedDoNotUpdateUnexpPC => {
                        SmallScalar::U8(rng.next_u32() as u8 % 2)
                    }
                    JoltR1CSInputs::OpFlags(flag) => {
                        // Generate random but reasonable flags
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
                        SmallScalar::Bool(b)
                    }
                };
                values[row * num_inputs + i] = val;
            }
        }

        RandomTestAccessor {
            values,
            rows: num_rows,
        }
    }
}

impl WitnessRowAccessor<TrackedFr> for RandomTestAccessor {
    fn num_steps(&self) -> usize {
        self.rows
    }

    fn value_at(&self, input_index: usize, row: usize) -> SmallScalar {
        self.values[row * JoltR1CSInputs::num_inputs() + input_index]
    }
}

fn bench_az_comparison(c: &mut Criterion) {
    let accessor = RandomTestAccessor::new(1000, 12345);
    let accessor_ref: &dyn WitnessRowAccessor<TrackedFr> = &accessor;

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

    c.bench_function("az_evaluation_generic", |b| {
        b.iter(|| {
            for constraint in UNIFORM_R1CS.iter() {
                for row in 0..black_box(10) {
                    let _result = eval_az_generic::<TrackedFr>(
                        black_box(&constraint.cons.a),
                        black_box(accessor_ref),
                        black_box(row),
                    );
                }
            }
        })
    });
}

fn bench_bz_comparison(c: &mut Criterion) {
    let accessor = RandomTestAccessor::new(1000, 54321);
    let accessor_ref: &dyn WitnessRowAccessor<TrackedFr> = &accessor;

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

    c.bench_function("bz_evaluation_generic", |b| {
        b.iter(|| {
            for constraint in UNIFORM_R1CS.iter() {
                for row in 0..black_box(10) {
                    let _result = eval_bz_generic::<TrackedFr>(
                        black_box(&constraint.cons.b),
                        black_box(accessor_ref),
                        black_box(row),
                    );
                }
            }
        })
    });
}

fn bench_single_constraint_comparison(c: &mut Criterion) {
    let accessor = RandomTestAccessor::new(1000, 98765);
    let accessor_ref: &dyn WitnessRowAccessor<TrackedFr> = &accessor;

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

        c.bench_function(&format!("single_constraint_{}_az_generic", idx), |b| {
            b.iter(|| {
                for row in 0..black_box(100) {
                    let _result = eval_az_generic::<TrackedFr>(
                        black_box(&constraint.cons.a),
                        black_box(accessor_ref),
                        black_box(row),
                    );
                }
            })
        });

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

        c.bench_function(&format!("single_constraint_{}_bz_generic", idx), |b| {
            b.iter(|| {
                for row in 0..black_box(100) {
                    let _result = eval_bz_generic::<TrackedFr>(
                        black_box(&constraint.cons.b),
                        black_box(accessor_ref),
                        black_box(row),
                    );
                }
            })
        });
    }
}

fn bench_az_evaluation(c: &mut Criterion) {
    let accessor = RandomTestAccessor::new(1000, 12345);
    let accessor_ref: &dyn WitnessRowAccessor<TrackedFr> = &accessor;

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
    let accessor_ref: &dyn WitnessRowAccessor<TrackedFr> = &accessor;

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
    let accessor_ref: &dyn WitnessRowAccessor<TrackedFr> = &accessor;

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
        let accessor_ref: &dyn WitnessRowAccessor<TrackedFr> = &accessor;

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
