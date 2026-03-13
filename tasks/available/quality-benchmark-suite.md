# benchmark-suite: Performance Benchmarks for All Crates

**Scope:** all new crates

**Depends:** all implementation and testing tasks

**Verifier:** ./verifiers/scoped.sh /workdir

**Context:**

Create comprehensive performance benchmarks for all crates to track performance, identify bottlenecks, and prevent regressions.

### Benchmark Structure

Each crate gets a `benches/` directory with targeted benchmarks for performance-critical operations.

### 1. jolt-poly Benchmarks

**benches/polynomial_ops.rs:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use jolt_poly::{DensePolynomial, CompactPolynomial, MultilinearPolynomial, EqPolynomial};
use jolt_field::ark_bn254::Fr as TestField;
use rand::thread_rng;

fn bench_polynomial_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_evaluation");
    let mut rng = thread_rng();

    for num_vars in [4, 8, 12, 16, 20] {
        // Dense polynomial evaluation
        let dense_poly = DensePolynomial::<TestField>::random(num_vars, &mut rng);
        let point: Vec<TestField> = (0..num_vars)
            .map(|_| TestField::random(&mut rng))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("dense", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    black_box(dense_poly.evaluate(black_box(&point)))
                })
            },
        );

        // Compact polynomial evaluation (u8)
        let size = 1 << num_vars;
        let scalars: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let compact_poly = CompactPolynomial::<u8, TestField>::new(scalars);

        group.bench_with_input(
            BenchmarkId::new("compact_u8", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    black_box(compact_poly.evaluate(black_box(&point)))
                })
            },
        );
    }

    group.finish();
}

fn bench_polynomial_binding(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_binding");
    let mut rng = thread_rng();

    for num_vars in [4, 8, 12, 16] {
        let poly = DensePolynomial::<TestField>::random(num_vars, &mut rng);
        let scalar = TestField::random(&mut rng);

        // Measure binding (creates new polynomial)
        group.bench_with_input(
            BenchmarkId::new("bind", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    let poly_clone = poly.clone();
                    black_box(poly_clone.bind(black_box(scalar)))
                })
            },
        );

        // Measure in-place binding
        group.bench_with_input(
            BenchmarkId::new("bind_in_place", num_vars),
            &num_vars,
            |b, _| {
                b.iter_batched(
                    || poly.clone(),
                    |mut p| {
                        p.bind_in_place(black_box(scalar));
                        black_box(p)
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

fn bench_eq_polynomial(c: &mut Criterion) {
    let mut group = c.benchmark_group("eq_polynomial");
    let mut rng = thread_rng();

    for num_vars in [4, 8, 12, 16] {
        let point: Vec<TestField> = (0..num_vars)
            .map(|_| TestField::random(&mut rng))
            .collect();
        let eq_poly = EqPolynomial::new(point.clone());

        // Measure full evaluation table generation
        group.bench_with_input(
            BenchmarkId::new("evaluations", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    black_box(eq_poly.evaluations())
                })
            },
        );

        // Measure single point evaluation
        let eval_point: Vec<TestField> = (0..num_vars)
            .map(|_| TestField::random(&mut rng))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("evaluate_single", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    black_box(eq_poly.evaluate(black_box(&eval_point)))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_polynomial_evaluation,
    bench_polynomial_binding,
    bench_eq_polynomial
);
criterion_main!(benches);
```

### 2. jolt-sumcheck Benchmarks

**benches/sumcheck_protocol.rs:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use jolt_sumcheck::*;
use jolt_poly::{DensePolynomial, MultilinearPolynomial, UnivariatePoly};
use jolt_field::ark_bn254::Fr as TestField;
use jolt_transcript::Blake2bTranscript;

struct BenchmarkWitness {
    poly: DensePolynomial<TestField>,
}

impl SumcheckInstanceProver<TestField> for BenchmarkWitness {
    fn round_polynomial(&self) -> UnivariatePoly<TestField> {
        // Simplified for benchmarking
        let mut coeffs = vec![TestField::zero(); 2];
        for i in 0..self.poly.len() / 2 {
            coeffs[0] += self.poly.evaluations()[2 * i];
            coeffs[1] += self.poly.evaluations()[2 * i + 1] - self.poly.evaluations()[2 * i];
        }
        UnivariatePoly::new(coeffs)
    }

    fn bind(&mut self, challenge: TestField) {
        self.poly.bind_in_place(challenge);
    }
}

fn bench_sumcheck_proving(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_proving");
    let mut rng = thread_rng();

    for num_vars in [4, 8, 12, 16] {
        let poly = DensePolynomial::<TestField>::random(num_vars, &mut rng);
        let sum: TestField = poly.evaluations().iter().sum();

        let claim = SumcheckClaim {
            num_vars,
            degree: 1,
            claimed_sum: sum,
        };

        group.bench_with_input(
            BenchmarkId::new("prove", num_vars),
            &num_vars,
            |b, _| {
                b.iter_batched(
                    || (BenchmarkWitness { poly: poly.clone() }, Blake2bTranscript::new(b"bench")),
                    |(mut witness, mut transcript)| {
                        black_box(SumcheckProver::prove(&claim, &mut witness, &mut transcript))
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

fn bench_sumcheck_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_verification");
    let mut rng = thread_rng();

    for num_vars in [4, 8, 12, 16] {
        // Generate a proof to verify
        let poly = DensePolynomial::<TestField>::random(num_vars, &mut rng);
        let sum: TestField = poly.evaluations().iter().sum();

        let claim = SumcheckClaim {
            num_vars,
            degree: 1,
            claimed_sum: sum,
        };

        let mut witness = BenchmarkWitness { poly };
        let mut transcript = Blake2bTranscript::new(b"bench");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut transcript);

        group.bench_with_input(
            BenchmarkId::new("verify", num_vars),
            &num_vars,
            |b, _| {
                b.iter_batched(
                    || Blake2bTranscript::new(b"bench"),
                    |mut transcript| {
                        black_box(SumcheckVerifier::verify(&claim, &proof, &mut transcript))
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

fn bench_batched_sumcheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("batched_sumcheck");
    let mut rng = thread_rng();

    for batch_size in [2, 4, 8, 16] {
        let num_vars = 10; // Fixed size, vary batch

        let mut claims = Vec::new();
        let mut witnesses: Vec<Box<dyn SumcheckInstanceProver<TestField>>> = Vec::new();

        for _ in 0..batch_size {
            let poly = DensePolynomial::<TestField>::random(num_vars, &mut rng);
            let sum: TestField = poly.evaluations().iter().sum();

            claims.push(SumcheckClaim {
                num_vars,
                degree: 1,
                claimed_sum: sum,
            });

            witnesses.push(Box::new(BenchmarkWitness { poly }));
        }

        group.bench_with_input(
            BenchmarkId::new("prove_batched", batch_size),
            &batch_size,
            |b, _| {
                b.iter_batched(
                    || {
                        let mut w: Vec<Box<dyn SumcheckInstanceProver<TestField>>> = Vec::new();
                        for witness in &witnesses {
                            w.push(Box::new(BenchmarkWitness {
                                poly: witness.as_any().downcast_ref::<BenchmarkWitness>()
                                    .unwrap().poly.clone()
                            }));
                        }
                        (w, Blake2bTranscript::new(b"bench"))
                    },
                    |(mut w, mut transcript)| {
                        black_box(SumcheckProver::prove_batched(&claims, &mut w, &mut transcript))
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sumcheck_proving,
    bench_sumcheck_verification,
    bench_batched_sumcheck
);
criterion_main!(benches);
```

### 3. jolt-openings Benchmarks

**benches/commitment_ops.rs:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use jolt_openings::*;
use jolt_poly::{DensePolynomial, MultilinearPolynomial};
use jolt_field::ark_bn254::Fr as TestField;

fn bench_accumulator_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("opening_accumulator");
    let mut rng = thread_rng();

    for num_openings in [10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::new("accumulate", num_openings),
            &num_openings,
            |b, &n| {
                // Pre-generate polynomials and points
                let polys: Vec<DensePolynomial<TestField>> = (0..n)
                    .map(|_| DensePolynomial::random(10, &mut rng))
                    .collect();

                let points: Vec<Vec<TestField>> = (0..n)
                    .map(|_| (0..10).map(|_| TestField::random(&mut rng)).collect())
                    .collect();

                b.iter_batched(
                    || ProverOpeningAccumulator::<TestField>::new(),
                    |mut acc| {
                        for i in 0..n {
                            let eval = polys[i].evaluate(&points[i]);
                            acc.accumulate(&polys[i], points[i].clone(), eval);
                        }
                        black_box(acc)
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_accumulator_operations);
criterion_main!(benches);
```

### 4. jolt-spartan Benchmarks

**benches/spartan_r1cs.rs:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use jolt_spartan::*;
use jolt_field::ark_bn254::Fr as TestField;

fn bench_spartan_key_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("spartan_key_generation");

    for num_constraints in [10, 100, 1000] {
        let r1cs = create_random_r1cs(num_constraints, num_constraints + 1);

        group.bench_with_input(
            BenchmarkId::new("from_r1cs", num_constraints),
            &num_constraints,
            |b, _| {
                b.iter(|| {
                    black_box(SpartanKey::from_r1cs(&r1cs))
                })
            },
        );
    }

    group.finish();
}

fn bench_uniform_r1cs(c: &mut Criterion) {
    let mut group = c.benchmark_group("uniform_r1cs");

    for num_blocks in [10, 50, 100, 500] {
        let block_size = 4;
        let uniform_r1cs = create_uniform_r1cs(block_size, num_blocks);
        let key = SpartanKey::from_r1cs(&uniform_r1cs);
        let witness = create_satisfying_witness(&uniform_r1cs);

        group.bench_with_input(
            BenchmarkId::new("multiply_witness", num_blocks),
            &num_blocks,
            |b, _| {
                b.iter(|| {
                    black_box(uniform_r1cs.multiply_witness(&witness))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_spartan_key_generation, bench_uniform_r1cs);
criterion_main!(benches);
```

### 5. jolt-instructions Benchmarks

**benches/instruction_lookups.rs:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use jolt_instructions::*;
use jolt_field::ark_bn254::Fr as TestField;

fn bench_instruction_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("instruction_execution");
    let instruction_set = JoltInstructionSet::new();

    let test_cases = vec![
        ("ADD", vec![0x123456789ABCDEF0u64, 0x0FEDCBA987654321u64]),
        ("MUL", vec![0x123456789u64, 0x987654321u64]),
        ("SLL", vec![0x1u64, 10u64]),
        ("AND", vec![0xFFFFFFFFu64, 0x0F0F0F0Fu64]),
    ];

    for (name, operands) in test_cases {
        if let Some(instruction) = instruction_set.instruction_by_name(name) {
            group.bench_function(name, |b| {
                b.iter(|| {
                    black_box(instruction.execute(black_box(&operands)))
                })
            });
        }
    }

    group.finish();
}

fn bench_instruction_lookup_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("instruction_lookups");
    let instruction_set = JoltInstructionSet::new();

    let test_cases = vec![
        ("ADD", vec![0x123456789ABCDEF0u64, 0x0FEDCBA987654321u64]),
        ("MUL", vec![0x123456789u64, 0x987654321u64]),
        ("SLL", vec![0x1u64, 10u64]),
    ];

    for (name, operands) in test_cases {
        if let Some(instruction) = instruction_set.instruction_by_name(name) {
            group.bench_function(format!("{}_lookups", name), |b| {
                b.iter(|| {
                    black_box(instruction.lookups(black_box(&operands)))
                })
            });
        }
    }

    group.finish();
}

fn bench_lookup_table_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_table_evaluation");
    let instruction_set = JoltInstructionSet::new();

    // Benchmark different table sizes
    for table in instruction_set.tables() {
        if table.size() <= 65536 { // Skip very large tables for benchmarking
            group.bench_function(
                format!("{}_evaluate", table.name()),
                |b| {
                    let input = (table.size() / 2) as u64;
                    b.iter(|| {
                        black_box(table.evaluate(black_box(input)))
                    })
                },
            );

            group.bench_function(
                format!("{}_materialize", table.name()),
                |b| {
                    b.iter(|| {
                        black_box(table.materialize())
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_instruction_execution,
    bench_instruction_lookup_decomposition,
    bench_lookup_table_evaluation
);
criterion_main!(benches);
```

### 6. End-to-End Benchmarks

**benches/e2e_proving.rs:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use jolt_zkvm::*;
use jolt_dory::*;

fn bench_program_proving(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_proving");
    group.sample_size(10); // Reduce sample size for expensive operations

    let programs = vec![
        ("fibonacci_10", generate_fibonacci_program(10)),
        ("fibonacci_20", generate_fibonacci_program(20)),
        ("loop_100", generate_loop_program(100)),
        ("loop_1000", generate_loop_program(1000)),
    ];

    for (name, program) in programs {
        let trace = execute_program(&program);

        group.bench_function(format!("prove_{}", name), |b| {
            b.iter_batched(
                || create_prover_with_trace(trace.clone()),
                |(mut prover, trace, mut transcript)| {
                    black_box(prover.prove(trace, &mut transcript).unwrap())
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_proof_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_verification");

    let programs = vec![
        ("fibonacci_10", generate_fibonacci_program(10)),
        ("loop_100", generate_loop_program(100)),
    ];

    for (name, program) in programs {
        let trace = execute_program(&program);
        let proof = generate_proof(trace);

        group.bench_function(format!("verify_{}", name), |b| {
            b.iter_batched(
                || (create_verifier(), Blake2bTranscript::new(b"bench")),
                |(verifier, mut transcript)| {
                    black_box(verifier.verify(&proof, &mut transcript).unwrap())
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(std::time::Duration::from_secs(30));
    targets = bench_program_proving, bench_proof_verification
);
criterion_main!(benches);
```

### 7. Benchmark Runner Script

Create `scripts/run-benchmarks.sh`:

```bash
#!/bin/bash
set -e

echo "Running Jolt Performance Benchmarks"
echo "==================================="
echo

# Set consistent CPU frequency if available (Linux)
if command -v cpupower &> /dev/null; then
    echo "Setting CPU to performance mode..."
    sudo cpupower frequency-set -g performance
fi

# Create results directory
RESULTS_DIR="benchmark-results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Run benchmarks for each crate
CRATES=(
    "jolt-poly"
    "jolt-sumcheck"
    "jolt-openings"
    "jolt-spartan"
    "jolt-instructions"
)

for crate in "${CRATES[@]}"; do
    echo "Benchmarking $crate..."

    # Run benchmark and save results
    cargo bench -p "$crate" --bench '*' -- \
        --save-baseline "$crate-baseline" \
        --output-format bencher \
        | tee "$RESULTS_DIR/$crate.txt"

    echo "Completed $crate"
    echo
done

# Run end-to-end benchmarks
echo "Running end-to-end benchmarks..."
cargo bench --bench e2e_proving -- \
    --save-baseline e2e-baseline \
    --output-format bencher \
    | tee "$RESULTS_DIR/e2e.txt"

# Generate comparison report if baseline exists
if [ -d "target/criterion/e2e-baseline" ]; then
    echo
    echo "Generating comparison report..."
    cargo bench --bench e2e_proving -- --baseline e2e-baseline > "$RESULTS_DIR/comparison.txt"
fi

echo
echo "Benchmark complete! Results saved to $RESULTS_DIR"

# Reset CPU frequency
if command -v cpupower &> /dev/null; then
    sudo cpupower frequency-set -g ondemand
fi
```

### 8. Performance Tracking

Create `benchmarks/track-performance.py`:

```python
#!/usr/bin/env python3
"""Track performance over time and detect regressions."""

import json
import sys
from pathlib import Path
from datetime import datetime

def parse_criterion_output(filepath):
    """Parse criterion benchmark output."""
    results = {}
    with open(filepath, 'r') as f:
        # Parse criterion output format
        # Extract benchmark name, time, throughput
    return results

def check_regression(current, baseline, threshold=0.10):
    """Check if current performance regressed vs baseline."""
    regressions = []

    for bench_name, current_time in current.items():
        if bench_name in baseline:
            baseline_time = baseline[bench_name]
            regression_pct = (current_time - baseline_time) / baseline_time

            if regression_pct > threshold:
                regressions.append({
                    'benchmark': bench_name,
                    'baseline': baseline_time,
                    'current': current_time,
                    'regression': f"{regression_pct * 100:.1f}%"
                })

    return regressions

def main():
    # Load current and baseline results
    # Check for regressions
    # Generate report
    pass

if __name__ == "__main__":
    main()
```

### Benchmark Guidelines

Add to each crate's README:

```markdown
## Benchmarking

To run benchmarks for this crate:

```bash
cargo bench -p jolt-poly
```

To compare against a baseline:

```bash
# Save current performance as baseline
cargo bench -p jolt-poly -- --save-baseline my-baseline

# Later, compare against baseline
cargo bench -p jolt-poly -- --baseline my-baseline
```

### Key Performance Metrics

- **Polynomial evaluation**: Should scale linearly with polynomial size
- **Binding operations**: Critical for sumcheck performance
- **Proof generation**: Should remain sub-linear in program size
- **Memory usage**: Compact polynomials should use 8-32x less memory

### Optimization Opportunities

When optimizing, focus on:

1. Hot paths in sumcheck rounds
2. Polynomial binding operations
3. Memory access patterns
4. Parallelization opportunities
```

### Current Progress

The following crates already have Criterion benchmarks in place:

| Crate | Bench file | Status |
|-------|-----------|--------|
| `jolt-field` | `benches/field_arith.rs` | Done |
| `jolt-poly` | `benches/poly_ops.rs` | Done |
| `jolt-sumcheck` | `benches/sumcheck_prove.rs` | Done |
| `jolt-openings` | `benches/rlc.rs` | Done |
| `jolt-transcript` | `benches/transcript_ops.rs` | Done |
| `jolt-spartan` | — | Not started |
| `jolt-instructions` | — | Not started |

**Remaining work:**
- Add benchmarks for `jolt-spartan` (key generation, witness multiplication)
- Add benchmarks for `jolt-instructions` (instruction execution, lookup decomposition, table evaluation)
- End-to-end benchmarks (blocked on `jolt-zkvm`)
- Benchmark runner script (`scripts/run-benchmarks.sh`)
- Performance tracking tooling

**Note:** The code samples in sections 1–3 above are illustrative but do not match the actual API signatures. The real benchmarks use `Polynomial<T>` (not `DensePolynomial`/`CompactPolynomial` as separate types), and `SumcheckCompute` (not `SumcheckInstanceProver`). Update samples if using this task as a reference.

### Acceptance Criteria

- Comprehensive benchmarks for all performance-critical operations
- Each crate has its own benchmark suite
- End-to-end benchmarks for full proving/verification
- Benchmark runner script for consistent testing
- Performance tracking over time
- Clear documentation on running benchmarks
- Baseline comparison capability
- Performance regression detection
- All benchmarks run successfully
- Results show expected performance characteristics