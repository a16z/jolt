#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests: fail loudly"
)]

use ark_bn254::G1Affine;
use ark_ff::{Field, UniformRand};
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_witness::RowSource;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use super::super::g1::g1_seg_sum_dispatch;
use super::super::testing::gpu_lock;
use super::*;
use crate::optimized::testing::{with_ram_fixture, FixtureShape, RamOp};

/// Interesting signed-recoding inputs: carry chains, magnitude-128
/// boundaries, and the extreme i128s (`unsigned_abs` of `i128::MIN`).
const EDGE_VALUES: [i128; 20] = [
    0,
    1,
    -1,
    127,
    128,
    129,
    -127,
    -128,
    -129,
    255,
    256,
    257,
    -255,
    -256,
    65535,
    -65536,
    i64::MAX as i128,
    i64::MIN as i128,
    i128::MAX,
    i128::MIN,
];

/// The recoding is a correct signed-digit decomposition: digits stay in
/// range and reconstruct the value in the scalar field.
#[test]
fn signed_digits_reconstruct() {
    let check = |value: i128| {
        let mut sum = ark_bn254::Fr::from(0u64);
        let mut digits = 0usize;
        for_each_signed_digit(value, |slot, magnitude, negate| {
            assert!((1..=INC_MAGNITUDES as u32).contains(&magnitude), "{value}");
            assert!((slot as usize) < INC_SLOTS, "{value}");
            let term = ark_bn254::Fr::from(u64::from(magnitude))
                * ark_bn254::Fr::from(256u64).pow([u64::from(slot)]);
            if negate {
                sum -= term;
            } else {
                sum += term;
            }
            digits += 1;
        });
        assert_eq!(sum, ark_bn254::Fr::from(value), "value {value}");
        if value == 0 {
            assert_eq!(digits, 0);
        }
    };
    for value in EDGE_VALUES {
        check(value);
    }
    let mut rng = ChaCha20Rng::seed_from_u64(23);
    for _ in 0..500 {
        let value = ((u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64())) as i128;
        check(value >> (rng.next_u32() % 128));
    }
}

/// Full increment path against arkworks: recode + bucket, device segment
/// sums (split at a tiny cap), weighted reduction — every window's row
/// equals the direct per-scalar MSM, at a nonzero window base.
#[test]
fn inc_rows_match_direct_msm() {
    let _lock = gpu_lock();
    let ctx = MetalContext::global().expect("metal context");
    let mut rng = ChaCha20Rng::seed_from_u64(29);
    let row_width = 8usize;
    let n_windows = 3usize;
    let window_base = 5usize;
    let bases: Vec<G1Affine> = (0..row_width).map(|_| G1Affine::rand(&mut rng)).collect();

    // Column 0: the edge values (padded with a repeat digit-collision
    // value to force bucket splits). Column 1: random full-range i128s.
    let mut columns = vec![Vec::new(), Vec::new()];
    columns[0].extend_from_slice(&EDGE_VALUES[..row_width * n_windows - 4]);
    columns[0].extend([42i128; 4]);
    for _ in 0..row_width * n_windows {
        columns[1].push(((u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64())) as i128);
    }

    let inc = build_inc_job(
        &columns,
        row_width,
        window_base,
        2,
        &mut DriverScratch::new(row_width),
        &mut SlabPool::detached(),
    )
    .expect("nonzero scalars");
    let n_segs = inc.segs.len();
    let bases_buf = ctx.wrap_slice(bases_as_u32s(&bases)).unwrap();
    let indices_buf = inc.indices.device_buffer(ctx).unwrap();
    let bounds_buf = inc.seg_bounds.device_buffer(ctx).unwrap();
    let out_buf = ctx.alloc_u32s(n_segs * JAC_U32S).unwrap();
    g1_seg_sum_dispatch(ctx, &bases_buf, &indices_buf, &bounds_buf, &out_buf, n_segs)
        .expect("dispatch");
    let mut jac = vec![0u32; n_segs * JAC_U32S];
    out_buf.copy_to_u32s(&mut jac);

    let mut inc_rows: Vec<Vec<Bn254G1>> =
        vec![vec![Default::default(); window_base + n_windows]; columns.len()];
    reduce_inc_superchunk(&inc.segs, &jac, &mut inc_rows);

    for (column, values) in columns.iter().enumerate() {
        for window in 0..n_windows {
            let expected = values[window * row_width..(window + 1) * row_width]
                .iter()
                .zip(&bases)
                .fold(G1Projective::zero(), |acc, (&value, base)| {
                    acc + *base * ark_bn254::Fr::from(value)
                });
            assert_eq!(
                inc_rows[column][window_base + window],
                Bn254G1::from(expected),
                "column {column} window {window}"
            );
        }
        for row in &inc_rows[column][..window_base] {
            assert_eq!(*row, Bn254G1::default(), "untouched row");
        }
    }
}

fn assert_same(
    cpu: &[WitnessCommitment<DoryScheme>],
    device: &[(DoryCommitment, DoryHint)],
    label: &str,
) {
    assert_eq!(cpu.len(), device.len());
    for (cpu, (commitment, hint)) in cpu.iter().zip(device) {
        assert_eq!(
            &cpu.commitment, commitment,
            "{label}: {:?} commitment diverged",
            cpu.id
        );
        assert_eq!(&cpu.hint, hint, "{label}: {:?} hint diverged", cpu.id);
    }
}

/// The device pipeline must reproduce the optimized kernel's commitments
/// and hints exactly: whole-trace superchunks with production segment
/// caps, and single-window superchunks with a 1-entry segment cap (every
/// addition its own device thread — the deepest multi-segment reduction
/// and multi-delivery sequencing). The Miller gate is forced open, so
/// every arm also exercises the hybrid tier-2 absorb at the default CPU
/// share; dedicated arms pin the all-device and all-CPU extremes
/// (partition invariance makes every split byte-identical).
#[test]
fn metal_commit_matches_optimized() {
    let _lock = gpu_lock();
    // nextest runs one process per test, so the env writes cannot race
    // another test. The tiny flush threshold forces mid-stream batch
    // flushes (production only reaches them at deep geometries).
    std::env::set_var("JOLT_METAL_MIN_TERMS_MILLER", "1");
    std::env::set_var("JOLT_METAL_MILLER_FLUSH_PAIRS", "8");
    let shape = FixtureShape {
        log_t: 6,
        ram_k: 16,
    };
    let ops = vec![
        RamOp::Write { word: 2, post: 17 },
        RamOp::Read { word: 2 },
        RamOp::None,
        RamOp::Write { word: 5, post: 3 },
        RamOp::Read { word: 5 },
        RamOp::Write { word: 2, post: 9 },
        RamOp::Read { word: 3 },
    ];
    with_ram_fixture(shape, ops, |witness| {
        let ids: Vec<JoltCommittedPolynomial> = witness
            .committed_order()
            .unwrap()
            .into_iter()
            .filter(|id| {
                !matches!(
                    id,
                    JoltCommittedPolynomial::TrustedAdvice
                        | JoltCommittedPolynomial::UntrustedAdvice
                )
            })
            .collect();
        let grid = CommitmentGrid {
            total_vars: 4 + shape.log_t,
            log_t: shape.log_t,
            log_k_chunk: 4,
            order: TracePolynomialOrder::CycleMajor,
        };
        let setup = DoryScheme::setup_prover(grid.total_vars);
        let source: &dyn RowSource = witness;
        let kinds = column_kinds::<Fr>(&ids, grid).unwrap();
        let ctx = MetalContext::global().expect("metal context");

        let optimized = <OptimizedBackend as CommitWitness<Fr, DoryScheme>>::commit_witness(
            &OptimizedBackend,
            &mut ProofSession::default(),
            source,
            &ids,
            grid,
            &setup,
        )
        .unwrap();

        let miller_dispatches = super::super::testing::miller_dispatch_count();
        let whole_trace = commit_streaming_metal(
            ctx,
            source,
            &kinds,
            grid,
            &setup,
            1 << shape.log_t,
            MAX_SEGMENT_LEN,
            true,
        )
        .expect("whole-trace metal commit");
        assert_same(&optimized, &whole_trace, "whole-trace superchunk");
        assert!(
            super::super::testing::miller_dispatch_count() > miller_dispatches,
            "the hybrid absorb never dispatched a device Miller batch"
        );

        let single_window = commit_streaming_metal(
            ctx,
            source,
            &kinds,
            grid,
            &setup,
            grid.num_columns(),
            1,
            true,
        )
        .expect("single-window metal commit");
        assert_same(&optimized, &single_window, "single-window superchunk");

        let inc_on_cpu = commit_streaming_metal(
            ctx,
            source,
            &kinds,
            grid,
            &setup,
            1 << shape.log_t,
            MAX_SEGMENT_LEN,
            false,
        )
        .expect("cpu-increment metal commit");
        assert_same(&optimized, &inc_on_cpu, "cpu-increment fallback");

        // The split extremes: all pairs on device (fly kill-switch
        // arm), then all on CPU (which skips the table build).
        std::env::set_var("JOLT_METAL_MILLER_CPU_FRACTION", "0");
        std::env::set_var("JOLT_METAL_MILLER_COMMIT_FLY", "1");
        let all_device = commit_streaming_metal(
            ctx,
            source,
            &kinds,
            grid,
            &setup,
            1 << shape.log_t,
            MAX_SEGMENT_LEN,
            true,
        )
        .expect("all-device miller commit");
        assert_same(&optimized, &all_device, "all-device miller split");

        std::env::set_var("JOLT_METAL_MILLER_CPU_FRACTION", "1");
        let all_cpu = commit_streaming_metal(
            ctx,
            source,
            &kinds,
            grid,
            &setup,
            1 << shape.log_t,
            MAX_SEGMENT_LEN,
            true,
        )
        .expect("all-cpu miller commit");
        assert_same(&optimized, &all_cpu, "all-cpu miller split");
        std::env::remove_var("JOLT_METAL_MILLER_CPU_FRACTION");
        std::env::remove_var("JOLT_METAL_MILLER_COMMIT_FLY");
    });
}

/// The full slot path: with the gate forced open, `MetalCommitWitness`
/// routes through the device and matches the optimized kernel; advice
/// stays delegated.
#[test]
fn metal_slot_matches_optimized_through_gate() {
    let _lock = gpu_lock();
    // nextest runs one process per test, so the env writes cannot race
    // another test.
    std::env::set_var("JOLT_METAL_MIN_TERMS_COMMIT", "1");
    std::env::set_var("JOLT_METAL_MIN_TERMS_COMMIT_INC", "1");
    let shape = FixtureShape {
        log_t: 6,
        ram_k: 16,
    };
    let ops = vec![
        RamOp::Write { word: 1, post: 5 },
        RamOp::Read { word: 1 },
        RamOp::Write { word: 7, post: 2 },
    ];
    with_ram_fixture(shape, ops, |witness| {
        let ids: Vec<JoltCommittedPolynomial> = witness
            .committed_order()
            .unwrap()
            .into_iter()
            .filter(|id| {
                !matches!(
                    id,
                    JoltCommittedPolynomial::TrustedAdvice
                        | JoltCommittedPolynomial::UntrustedAdvice
                )
            })
            .collect();
        let grid = CommitmentGrid {
            total_vars: 4 + shape.log_t,
            log_t: shape.log_t,
            log_k_chunk: 4,
            order: TracePolynomialOrder::CycleMajor,
        };
        let setup = DoryScheme::setup_prover(grid.total_vars);
        let source: &dyn RowSource = witness;

        let optimized = <OptimizedBackend as CommitWitness<Fr, DoryScheme>>::commit_witness(
            &OptimizedBackend,
            &mut ProofSession::default(),
            source,
            &ids,
            grid,
            &setup,
        )
        .unwrap();
        let metal = MetalCommitWitness
            .commit_witness(&mut ProofSession::default(), source, &ids, grid, &setup)
            .unwrap();

        assert_eq!(optimized.len(), metal.len());
        for (cpu, device) in optimized.iter().zip(&metal) {
            assert_eq!(cpu.id, device.id);
            assert_eq!(cpu.commitment, device.commitment, "{:?}", cpu.id);
            assert_eq!(cpu.hint, device.hint, "{:?}", cpu.id);
        }
    });
}
