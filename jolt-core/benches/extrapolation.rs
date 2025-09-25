#![allow(clippy::uninlined_format_args)]

use ark_bn254::Fr;
use ark_std::rand::{rngs::StdRng, RngCore, SeedableRng};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use jolt_core::utils::interpolation::{
    ex16_consecutive, ex4_consecutive, ex8_consecutive, extend_consecutive_symmetric_const,
};
use jolt_core::utils::interpolation::binomial::choose as binom_choose;
use jolt_core::field::JoltField;

fn gen_base<const N: usize>(rng: &mut StdRng) -> [Fr; N] {
    let mut arr = [Fr::from_u64(0); N];
    for i in 0..N {
        arr[i] = Fr::from_u64(rng.next_u64());
    }
    arr
}

// Build forward-difference diagonal at i=0: diffs[k] = Δ^k f(0), k=1..N-1
fn build_diffs<const N: usize>(base: &[Fr; N]) -> [Fr; N] {
    let mut work = *base;
    let mut diffs = [Fr::from_u64(0); N];
    for k in 1..N {
        for i in 0..(N - k) {
            work[i] = work[i + 1] - work[i];
        }
        diffs[k] = work[0];
    }
    diffs
}

// Produce N forward values p(N),...,p(2N-1) using the difference table only
fn forward_by_diffs_sum<const N: usize>(base: &[Fr; N]) -> Fr {
    let mut diffs = build_diffs::<N>(base);
    // advance diffs to right boundary i=N-1
    for _ in 0..(N - 1) {
        for k in (1..(N - 1)).rev() {
            diffs[k] += diffs[k + 1];
        }
    }
    let mut anchor = base[N - 1];
    let mut acc = Fr::from_u64(0);
    for _ in 0..N {
        anchor += diffs[1];
        acc += anchor;
        for k in (1..(N - 1)).rev() {
            diffs[k] += diffs[k + 1];
        }
    }
    acc
}

// Produce N forward values by sliding window and calling exN each step
fn forward_by_exn_sum<const N: usize>(base: &[Fr; N]) -> Fr {
    let mut win = *base;
    let mut acc = Fr::from_u64(0);
    for _ in 0..N {
        let next = match N {
            4 => ex4_consecutive::<Fr>(unsafe { &*(win.as_ptr() as *const [Fr; 4]) }),
            8 => ex8_consecutive::<Fr>(unsafe { &*(win.as_ptr() as *const [Fr; 8]) }),
            16 => ex16_consecutive::<Fr>(unsafe { &*(win.as_ptr() as *const [Fr; 16]) }),
            _ => unreachable!(),
        };
        acc += next;
        // slide window: drop first, push next
        for i in 0..(N - 1) {
            win[i] = win[i + 1];
        }
        win[N - 1] = next;
    }
    acc
}

fn backward_step_exn<const N: usize>(w: &[Fr; N]) -> Fr {
    // Compute p(x-1) from [p(x), ..., p(x+N-1)] using recurrence coefficients
    let mut acc = Fr::from_u64(0);
    for j in 0..N {
        let c = binom_choose(N, j + 1);
        let coef = Fr::from_u64(c as u64);
        if ((N + j) & 1) == 1 {
            acc -= w[j] * coef;
        } else {
            acc += w[j] * coef;
        }
    }
    acc
}

fn symmetric_by_diffs_sum<const N: usize>(base: &[Fr; N]) -> Fr {
    let out = extend_consecutive_symmetric_const::<Fr, N>(base);
    out.iter().copied().fold(Fr::from_u64(0), |a, b| a + b)
}

fn symmetric_by_exn_sum<const N: usize>(base: &[Fr; N]) -> Fr {
    // Left side: compute (-1, -2, ...) using backward recurrence sliding windows
    let left_cnt = (N + 1) / 2;
    let mut window_left = *base; // [p(0)..p(N-1)] initially
    let mut acc = Fr::from_u64(0);
    for _ in 0..left_cnt {
        let prev = backward_step_exn::<N>(&window_left);
        acc += prev;
        // slide window left: insert prev at front, drop last
        for i in (1..N).rev() {
            window_left[i] = window_left[i - 1];
        }
        window_left[0] = prev;
    }

    // Right side: compute (N, N+1, ...) using forward exN sliding windows
    let right_cnt = N / 2;
    let mut window_right = *base;
    for _ in 0..right_cnt {
        let next = match N {
            4 => ex4_consecutive::<Fr>(unsafe { &*(window_right.as_ptr() as *const [Fr; 4]) }),
            8 => ex8_consecutive::<Fr>(unsafe { &*(window_right.as_ptr() as *const [Fr; 8]) }),
            16 => ex16_consecutive::<Fr>(unsafe { &*(window_right.as_ptr() as *const [Fr; 16]) }),
            _ => unreachable!(),
        };
        acc += next;
        for i in 0..(N - 1) {
            window_right[i] = window_right[i + 1];
        }
        window_right[N - 1] = next;
    }
    acc
}

fn bench_extrapolation(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);

    for &n in &[4usize, 8, 16] {
        let mut group = c.benchmark_group(format!("extrapolation_n{n}"));

        match n {
            4 => {
                group.bench_function("forward_diff", |b| {
                    b.iter_batched(
                        || gen_base::<4>(&mut rng),
                        |base| black_box(forward_by_diffs_sum::<4>(black_box(&base))),
                        BatchSize::SmallInput,
                    )
                });
                group.bench_function("ex4_sliding", |b| {
                    b.iter_batched(
                        || gen_base::<4>(&mut rng),
                        |base| black_box(forward_by_exn_sum::<4>(black_box(&base))),
                        BatchSize::SmallInput,
                    )
                });
                group.bench_function("sym_diff", |b| {
                    b.iter_batched(
                        || gen_base::<4>(&mut rng),
                        |base| black_box(symmetric_by_diffs_sum::<4>(black_box(&base))),
                        BatchSize::SmallInput,
                    )
                });
                group.bench_function("sym_exn", |b| {
                    b.iter_batched(
                        || gen_base::<4>(&mut rng),
                        |base| black_box(symmetric_by_exn_sum::<4>(black_box(&base))),
                        BatchSize::SmallInput,
                    )
                });
            }
            8 => {
                group.bench_function("forward_diff", |b| {
                    b.iter_batched(
                        || gen_base::<8>(&mut rng),
                        |base| black_box(forward_by_diffs_sum::<8>(black_box(&base))),
                        BatchSize::SmallInput,
                    )
                });
                group.bench_function("ex8_sliding", |b| {
                    b.iter_batched(
                        || gen_base::<8>(&mut rng),
                        |base| black_box(forward_by_exn_sum::<8>(black_box(&base))),
                        BatchSize::SmallInput,
                    )
                });
                group.bench_function("sym_diff", |b| {
                    b.iter_batched(
                        || gen_base::<8>(&mut rng),
                        |base| black_box(symmetric_by_diffs_sum::<8>(black_box(&base))),
                        BatchSize::SmallInput,
                    )
                });
                group.bench_function("sym_exn", |b| {
                    b.iter_batched(
                        || gen_base::<8>(&mut rng),
                        |base| black_box(symmetric_by_exn_sum::<8>(black_box(&base))),
                        BatchSize::SmallInput,
                    )
                });
            }
            16 => {
                group.bench_function("forward_diff", |b| {
                    b.iter_batched(
                        || gen_base::<16>(&mut rng),
                        |base| black_box(forward_by_diffs_sum::<16>(black_box(&base))),
                        BatchSize::SmallInput,
                    )
                });
                group.bench_function("ex16_sliding", |b| {
                    b.iter_batched(
                        || gen_base::<16>(&mut rng),
                        |base| black_box(forward_by_exn_sum::<16>(black_box(&base))),
                        BatchSize::SmallInput,
                    )
                });
                group.bench_function("sym_diff", |b| {
                    b.iter_batched(
                        || gen_base::<16>(&mut rng),
                        |base| black_box(symmetric_by_diffs_sum::<16>(black_box(&base))),
                        BatchSize::SmallInput,
                    )
                });
                group.bench_function("sym_exn", |b| {
                    b.iter_batched(
                        || gen_base::<16>(&mut rng),
                        |base| black_box(symmetric_by_exn_sum::<16>(black_box(&base))),
                        BatchSize::SmallInput,
                    )
                });
            }
            _ => unreachable!(),
        }

        group.finish();
    }
}

criterion_group!(benches, bench_extrapolation);
criterion_main!(benches);
