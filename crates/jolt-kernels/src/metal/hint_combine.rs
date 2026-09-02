//! Device `combine_hints`: the stage-8 batch opening's hint combination
//! (`combined[r] = Σ_p γ^p · hint_p[r]` over ~40 ragged hints) as one
//! `jk_g1_combine_rows` dispatch — the largest single ALU term of the stage
//! on the CPU (plain per-row double-and-add scalar muls).
//!
//! The workload's shape beats a generic bucketed MSM here: the SCALARS are
//! shared by every row (one γ-power per hint), so a per-row thread can run a
//! single high-to-low double-and-add across all of the row's hints — the
//! ~254 doublings amortize over the whole row and every bit branch is
//! warp-uniform — with none of a bucket method's threadgroup-memory or
//! register pressure. Raggedness is prefix-structured by sorting hints by
//! row count (descending): row `r`'s live hints are exactly the sorted
//! prefix `{p : len_p > r}`, one uniform `break` on the device.
//!
//! The host builds the gather schedule: hints sorted and flattened
//! hint-major (lane-adjacent rows load adjacent points), all points batch-
//! normalized to affine so the kernel reuses the commit lane's parity-tested
//! `g1_madd`, identity rows lowered to a `(0, 0)` sentinel (not on the
//! curve), and the γ-powers converted to CANONICAL integer limbs for bit
//! tests. Results return as Jacobian points — group-equal to the CPU
//! combination by commutativity; every downstream consumer (Dory's reduce
//! rounds, transcript absorbs) operates on group elements or normalized
//! serializations, so proof bytes are unchanged. Pinned by this module's
//! normalized-coordinate parity tests and byte_diff's metal arms.
//!
//! Installed per proof through [`jolt_dory::install_combine_hints_hook`] by
//! the metal joint-opening slot (the guard parks in the `ProofSession`);
//! declines calls below the gate or on any device failure, falling back to
//! the CPU arithmetic of record.

use ark_bn254::{Fq as ArkFq, G1Affine, G1Projective};
use ark_ec::CurveGroup;
use ark_ff::{BigInteger, PrimeField, Zero};
use jolt_crypto::Bn254G1;
use jolt_dory::DoryHint;
use jolt_field::Fr;
use rayon::prelude::*;

use super::field::FR_U32_LIMBS;
use super::g1::{bases_as_u32s, jac_from_device_limbs, JAC_U32S};
use super::runtime::{KernelId, MetalContext};
use super::{metal_gate, testing, MetalError};

const KIND: &str = "combine_hints";

/// Work-item scaling for the gate: one hint point costs ~2^8 group
/// operations (its share of the 254 amortized doublings plus ~127 mixed
/// adds), where the gate's threshold calibration is one stream element.
const WORK_PER_POINT_LOG2: usize = 8;

/// Signed-digit slots per hint scalar in the NAF encoding (a 254-bit scalar's
/// NAF spans ≤ 255 digits; padded to a u32 boundary, four `i8` per word).
const NAF_DIGIT_SLOTS: usize = 256;
const NAF_U32S: usize = NAF_DIGIT_SLOTS / 4;

/// NAF digits default on (~1/3 nonzero density vs the canonical ladder's
/// 1/2 — one third fewer mixed adds for the same group value). Kill switch:
/// `JOLT_METAL_COMBINE_NAF=0` restores the canonical bit ladder (identical
/// proof bytes either way; both paths land the same group elements).
fn naf_enabled() -> bool {
    std::env::var("JOLT_METAL_COMBINE_NAF").map_or(true, |v| v.trim() != "0")
}

/// The `combine_hints` hook: `Some(combined)` when the device served the
/// call, `None` (undersized, empty, or failed) for the CPU path.
pub(super) fn combine_hints_device(hints: &[DoryHint], scalars: &[Fr]) -> Option<DoryHint> {
    let total_points: usize = hints.iter().map(|hint| hint.row_commitments.len()).sum();
    let num_rows = hints
        .iter()
        .map(|hint| hint.row_commitments.len())
        .max()
        .unwrap_or(0);
    if num_rows == 0 || !metal_gate(KIND, total_points << WORK_PER_POINT_LOG2) {
        return None;
    }
    let context = MetalContext::global().ok()?;
    match combine_rows_device(context, hints, scalars, num_rows) {
        Ok(row_commitments) => {
            // The blind combination is the CPU path's own formula.
            let commit_blind = hints
                .iter()
                .zip(scalars.iter())
                .map(|(hint, &scalar)| scalar * hint.commit_blind)
                .sum();
            Some(DoryHint::new(row_commitments, commit_blind))
        }
        Err(error) => {
            tracing::warn!(
                slot = KIND,
                %error,
                "device hint combination failed; combining on the CPU"
            );
            None
        }
    }
}

/// One `jk_g1_combine_rows` dispatch over the sorted, flattened, normalized
/// hint matrix.
fn combine_rows_device(
    context: &'static MetalContext,
    hints: &[DoryHint],
    scalars: &[Fr],
    num_rows: usize,
) -> Result<Vec<Bn254G1>, MetalError> {
    // Hints sorted by row count descending (empties dropped): per-row live
    // sets become prefixes of the order. The sort is a permutation of a
    // commutative sum — group-equal to the CPU's original-order fold.
    let prep_span = tracing::info_span!("combine_hints_prep").entered();
    let mut order: Vec<usize> = (0..hints.len())
        .filter(|&i| !hints[i].row_commitments.is_empty())
        .collect();
    order.sort_by_key(|&i| std::cmp::Reverse(hints[i].row_commitments.len()));
    let num_hints = order.len();

    let lens: Vec<u32> = order
        .iter()
        .map(|&i| hints[i].row_commitments.len() as u32)
        .collect();
    let mut offsets: Vec<u32> = Vec::with_capacity(num_hints);
    let mut total = 0u32;
    for &len in &lens {
        offsets.push(total);
        total = total
            .checked_add(len)
            .ok_or_else(|| MetalError::Execution("hint point count overflows u32".to_owned()))?;
    }

    // The kernel's shared sweep digits: signed NAF (default) or canonical
    // integer limbs, plus the highest nonzero digit across the live scalars
    // (the sweep's start).
    let use_naf = naf_enabled();
    let mut scalar_limbs: Vec<u32> =
        vec![0; num_hints * if use_naf { NAF_U32S } else { FR_U32_LIMBS }];
    let mut start_bit = 0u32;
    for (slot, &i) in order.iter().enumerate() {
        // SAFETY: jolt_field::Fr is #[repr(transparent)] over ark_bn254::Fr
        // (the layout contract `metal::field` const-asserts and jolt-dory's
        // adapters rely on).
        let ark: ark_bn254::Fr = unsafe { std::mem::transmute_copy(&scalars[i]) };
        let big = ark.into_bigint();
        if use_naf {
            let words = &mut scalar_limbs[slot * NAF_U32S..(slot + 1) * NAF_U32S];
            #[expect(clippy::expect_used, reason = "w = 2 is always a valid wNAF width")]
            let digits = big.find_wnaf(2).expect("w = 2 is a valid wNAF width");
            debug_assert!(
                digits.len() <= NAF_DIGIT_SLOTS,
                "NAF span exceeds its slots"
            );
            for (index, &digit) in digits.iter().enumerate() {
                debug_assert!((-1..=1).contains(&digit), "w = 2 digits are 0 or ±1");
                words[index / 4] |= u32::from(digit as i8 as u8) << ((index % 4) * 8);
                if digit != 0 {
                    start_bit = start_bit.max(index as u32);
                }
            }
        } else {
            let words = &mut scalar_limbs[slot * FR_U32_LIMBS..(slot + 1) * FR_U32_LIMBS];
            for (word_index, word) in big.0.iter().enumerate() {
                let lo = *word as u32;
                let hi = (*word >> 32) as u32;
                words[2 * word_index] = lo;
                words[2 * word_index + 1] = hi;
                for (half, limb) in [(0u32, lo), (1u32, hi)] {
                    if limb != 0 {
                        let bit = (word_index as u32) * 64 + half * 32 + limb.ilog2();
                        start_bit = start_bit.max(bit);
                    }
                }
            }
        }
    }

    drop(prep_span);

    // Normalize the ragged hint matrix straight into the hint-major affine
    // stream — no flattened intermediates. Segments (per hint) are disjoint
    // output ranges; chunks within a segment batch-invert independently
    // (affine coordinates are exact quotients, so batching structure cannot
    // change a value). Identity rows lower to the (0, 0) sentinel the kernel
    // skips; only nonzero points reach `normalize_batch`, so its inversion
    // never sees a zero Z.
    let normalize_span = tracing::info_span!("combine_hints_normalize").entered();
    let total = total as usize;
    let mut points: Vec<G1Affine> = Vec::with_capacity(total);
    let mut segments: Vec<(&mut [std::mem::MaybeUninit<G1Affine>], &[Bn254G1])> = Vec::new();
    let mut spare = points.spare_capacity_mut();
    for &i in &order {
        let rows = hints[i].row_commitments.as_slice();
        let (segment, rest) = spare.split_at_mut(rows.len());
        segments.push((segment, rows));
        spare = rest;
    }
    segments.into_par_iter().for_each(|(segment, rows)| {
        segment
            .par_chunks_mut(4096)
            .zip(rows.par_chunks(4096))
            .for_each(|(out_chunk, row_chunk)| {
                let nonzero: Vec<G1Projective> = row_chunk
                    .iter()
                    .map(|p| p.into_inner())
                    .filter(|p| !p.is_zero())
                    .collect();
                let normalized = G1Projective::normalize_batch(&nonzero);
                let mut normalized_iter = normalized.into_iter();
                for (out, row) in out_chunk.iter_mut().zip(row_chunk) {
                    let affine = if row.into_inner().is_zero() {
                        G1Affine::new_unchecked(ArkFq::zero(), ArkFq::zero())
                    } else {
                        #[expect(
                            clippy::expect_used,
                            reason = "normalize_batch returns one point per nonzero input"
                        )]
                        normalized_iter
                            .next()
                            .expect("normalized point per nonzero input")
                    };
                    let _ = out.write(affine);
                }
            });
    });
    // SAFETY: the segments partition `0..total` and every chunk writes each
    // of its slots exactly once above.
    unsafe { points.set_len(total) };
    drop(normalize_span);

    let buffers_span = tracing::info_span!("combine_hints_buffers").entered();
    let points_buffer = context.wrap_slice(bases_as_u32s(&points))?;
    let scalars_buffer = context.wrap_slice(&scalar_limbs)?;
    let lens_buffer = context.wrap_slice(&lens)?;
    let offsets_buffer = context.wrap_slice(&offsets)?;
    let out_buffer = context.alloc_u32s(num_rows * JAC_U32S)?;
    testing::note_copied_buffers(
        u64::from(points_buffer.was_copied())
            + u64::from(scalars_buffer.was_copied())
            + u64::from(lens_buffer.was_copied())
            + u64::from(offsets_buffer.was_copied()),
    );
    drop(buffers_span);

    let params = [
        num_rows as u32,
        num_hints as u32,
        start_bit,
        u32::from(use_naf),
    ];
    tracing::info_span!("combine_hints_kernel").in_scope(|| {
        context.run_once(
            KernelId::G1CombineRows,
            &params,
            &[
                &points_buffer,
                &scalars_buffer,
                &lens_buffer,
                &offsets_buffer,
                &out_buffer,
            ],
            num_rows,
        )
    })?;
    testing::note_device_round();

    let _readback = tracing::info_span!("combine_hints_readback").entered();
    let mut jac = vec![0u32; num_rows * JAC_U32S];
    out_buffer.copy_to_u32s(&mut jac);
    Ok(jac
        .chunks_exact(JAC_U32S)
        .map(|limbs| Bn254G1::from(jac_from_device_limbs(limbs)))
        .collect())
}

/// Normalized-coordinate parity against the CPU `combine_hints` on ragged,
/// non-normalized, identity-planted hint sets, plus the hook's install /
/// serve / uninstall scoping — device-forced and probed.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use ark_ff::UniformRand;
    use jolt_dory::{install_combine_hints_hook, DoryScheme};
    use jolt_field::Ring;
    use jolt_openings::AdditivelyHomomorphic;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::super::testing::{device_probe_count, gpu_lock};
    use super::*;

    fn force_device_gate() {
        // nextest runs one process per test, so env mutation is safe.
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");
    }

    /// A hint of `len` rows: sums of random points (Z ≠ 1, exercising the
    /// normalization) with identities planted every seventh row.
    fn synthetic_hint(len: usize, seed: u64) -> DoryHint {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let rows: Vec<Bn254G1> = (0..len)
            .map(|row| {
                if row % 7 == 3 {
                    Bn254G1::default()
                } else {
                    Bn254G1::from(G1Projective::rand(&mut rng) + G1Projective::rand(&mut rng))
                }
            })
            .collect();
        DoryHint::new(rows, Fr::from_u64(seed))
    }

    fn synthetic_batch() -> (Vec<DoryHint>, Vec<Fr>) {
        let hints = vec![
            synthetic_hint(64, 1),
            synthetic_hint(64, 2),
            synthetic_hint(33, 3),
            synthetic_hint(5, 4),
            synthetic_hint(1, 5),
            synthetic_hint(0, 6),
        ];
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let mut scalars: Vec<Fr> = (0..hints.len())
            .map(|_| {
                // SAFETY: Fr is #[repr(transparent)] over ark_bn254::Fr.
                unsafe { std::mem::transmute_copy(&ark_bn254::Fr::rand(&mut rng)) }
            })
            .collect();
        scalars[0] = Fr::from_u64(1); // γ⁰ — the batch's real first power
        scalars[3] = Fr::from_u64(3); // near-empty scalar: sparse bit sweep
        (hints, scalars)
    }

    #[test]
    fn combine_rows_matches_cpu() {
        let _lock = gpu_lock();
        force_device_gate();
        let (hints, scalars) = synthetic_batch();

        let cpu = DoryScheme::combine_hints(hints.clone(), &scalars);
        let probes_before = device_probe_count();
        let device = combine_hints_device(&hints, &scalars).unwrap();
        assert_eq!(
            device_probe_count() - probes_before,
            1,
            "the combination must run as one device dispatch"
        );

        assert_eq!(device.commit_blind, cpu.commit_blind);
        assert_eq!(device.row_commitments.len(), cpu.row_commitments.len());
        for (row, (device_row, cpu_row)) in device
            .row_commitments
            .iter()
            .zip(&cpu.row_commitments)
            .enumerate()
        {
            // Group equality AND normalized-coordinate identity.
            assert_eq!(
                device_row.into_inner(),
                cpu_row.into_inner(),
                "row {row} diverged"
            );
            assert_eq!(
                device_row.into_inner().into_affine(),
                cpu_row.into_inner().into_affine(),
                "row {row} normalized coordinates diverged"
            );
        }
    }

    /// The `JOLT_METAL_COMBINE_NAF=0` kill switch: the canonical bit ladder
    /// must produce the same normalized rows as the CPU path (and therefore
    /// as the default NAF sweep).
    #[test]
    fn combine_rows_canonical_ladder_matches_cpu() {
        let _lock = gpu_lock();
        force_device_gate();
        std::env::set_var("JOLT_METAL_COMBINE_NAF", "0");
        let (hints, scalars) = synthetic_batch();

        let cpu = DoryScheme::combine_hints(hints.clone(), &scalars);
        let device = combine_hints_device(&hints, &scalars).unwrap();
        std::env::remove_var("JOLT_METAL_COMBINE_NAF");

        assert_eq!(device.commit_blind, cpu.commit_blind);
        for (row, (device_row, cpu_row)) in device
            .row_commitments
            .iter()
            .zip(&cpu.row_commitments)
            .enumerate()
        {
            assert_eq!(
                device_row.into_inner().into_affine(),
                cpu_row.into_inner().into_affine(),
                "row {row} normalized coordinates diverged"
            );
        }
    }

    #[test]
    fn hook_scopes_to_its_guard() {
        let _lock = gpu_lock();
        force_device_gate();
        let (hints, scalars) = synthetic_batch();
        let reference = DoryScheme::combine_hints(hints.clone(), &scalars);

        let guard = install_combine_hints_hook(combine_hints_device);
        let probes_before = device_probe_count();
        let hooked = DoryScheme::combine_hints(hints.clone(), &scalars);
        assert!(
            device_probe_count() > probes_before,
            "the installed hook never dispatched"
        );
        drop(guard);

        let probes_after = device_probe_count();
        let unhooked = DoryScheme::combine_hints(hints, &scalars);
        assert_eq!(
            device_probe_count(),
            probes_after,
            "a dropped guard must uninstall the hook"
        );

        for ((hooked_row, reference_row), unhooked_row) in hooked
            .row_commitments
            .iter()
            .zip(&reference.row_commitments)
            .zip(&unhooked.row_commitments)
        {
            assert_eq!(hooked_row.into_inner(), reference_row.into_inner());
            assert_eq!(unhooked_row.into_inner(), reference_row.into_inner());
        }
        assert_eq!(hooked.commit_blind, reference.commit_blind);
    }
}
