//! Shared tier-2 pairing state for a streaming-commit pass.
//!
//! Every column finish pairs its row commitments against the SAME prefix of
//! the setup's fixed G2 generators, but `multi_pair_g2_setup` re-runs the G2
//! Miller-loop preparation on every call — for a 42-column witness commit
//! that is 42× redundant preparation of identical points (~40% of the tier-2
//! wall). [`DoryTier2Prep`] borrows the setup-owned prepared table (built
//! once at `setup_prover` time; ~0.47 s of per-proof Miller precompute moved
//! out of the prove wall) and the `*_prepared` finish variants pair against
//! it; a pass needing more rows than the table covers prepares its prefix
//! itself.
//!
//! Unlike dory-pcs's global prepared-point cache (deliberately not primed —
//! see `DoryScheme::setup_prover`), this state is scoped to one setup object
//! (one setup = one URS), so two setup sizes in one process cannot
//! cross-contaminate.
//!
//! Chunked Miller loops multiplied in Fp12 and final-exponentiated once give
//! the exact `multi_pair_g2_setup` value for any chunking: a multi-Miller
//! loop over a pair set equals the Fp12 product of the per-pair Miller
//! values, and final exponentiation is applied to the full product either
//! way. Commitments and hints are therefore byte-identical to the unprepared
//! path (pinned by parity tests in `streaming`).

#![expect(
    clippy::indexing_slicing,
    reason = "prover-only pairing tail; row/generator counts are validated \
              against the setup before any indexed walk"
)]

use std::sync::Arc;

use ark_bn254::{Bn254, Fq12, Fq2, Fq6, G1Affine, G1Projective};
use ark_ec::bn::BnConfig;
use ark_ec::pairing::{MillerLoopOutput, Pairing};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{Field, Fp6Config, One};
use dory::Mode;
use jolt_crypto::Bn254G1;
use rayon::prelude::*;

use crate::scheme::{ark_to_jolt_fr, ark_to_jolt_gt, ArkFr, ArkG1, ArkGT};
use crate::types::{DoryCommitment, DoryHint, DoryProverSetup};

/// One prepared line's coefficients (arkworks D-twist order).
type EllCoeff = ark_ec::bn::g2::EllCoeff<ark_bn254::Config>;

/// Setup-owned Miller-loop-prepared `g2_vec` prefix, shared by every proof
/// over one setup.
pub(crate) type PreparedG2Table = Arc<Vec<<Bn254 as Pairing>::G2Prepared>>;

/// Miller-prepare the first `rows` setup `g2_vec` generators — the eager
/// setup-owned table.
pub(crate) fn prepare_g2_table(
    setup: &dory::backends::arkworks::ArkworksProverSetup,
    rows: usize,
) -> PreparedG2Table {
    Arc::new(prepare_g2_points(&setup.g2_vec[..rows]))
}

fn prepare_g2_points(
    points: &[dory::backends::arkworks::ArkG2],
) -> Vec<<Bn254 as Pairing>::G2Prepared> {
    points
        .par_iter()
        .map(|q| q.0.into_affine().into())
        .collect()
}

/// Miller-loop-prepared `g2_vec[..max_rows]` setup generators, shared
/// read-only across every column finish of one commit pass.
pub struct DoryTier2Prep {
    table: PreparedG2Table,
    rows: usize,
}

/// Pairs per parallel Miller-loop chunk. Value-invariant (see module docs);
/// small enough that a single column finish yields useful parallelism now
/// that each chunk folds serially through one shared ladder (the ladder
/// costs 64 Fq12 squarings per chunk ≈ +0.4% of its line-fold work).
const MILLER_CHUNK: usize = 128;

impl DoryTier2Prep {
    pub(crate) fn new(setup: &DoryProverSetup, max_rows: usize) -> Self {
        assert!(
            max_rows <= setup.0.g2_vec.len(),
            "tier-2 prep: row count ({}) exceeds Dory SRS size ({})",
            max_rows,
            setup.0.g2_vec.len(),
        );
        let table = if setup.1.len() >= max_rows {
            Arc::clone(&setup.1)
        } else {
            Arc::new(prepare_g2_points(&setup.0.g2_vec[..max_rows]))
        };
        Self {
            table,
            rows: max_rows,
        }
    }

    /// The prepared generator prefix — the raw line-coefficient vectors a
    /// device Miller lane stages into its own layout.
    pub fn prepared(&self) -> &[<Bn254 as Pairing>::G2Prepared] {
        &self.table[..self.rows]
    }
}

// --- prepared-coefficient multi-Miller -----------------------------------------
//
// `Bn254::multi_miller_loop`'s value, computed cheaper for the fixed-G2
// absorb shape: ONE squaring ladder shared by the whole pair set (arkworks
// re-chunks pairs by 4, paying 16 Fq12 squarings per pair), coefficients
// borrowed from the prep (arkworks' by-value API forces a per-pair
// G2Prepared clone), and line values folded pairwise (Scott, eprint
// 2019/077 — the same combining `fq12.metal` uses on the device lane).
// Every regrouping multiplies the same Fp12 factors in a different order,
// so the output limbs are bit-identical; `fresh_multi_miller_matches_arkworks`
// and the streaming parity suites pin it.

/// (c0 + c1·v + c2·v²)·v with v³ = ξ — arkworks `mul_by_nonresidue` on the
/// Fq6 tower slot.
#[inline]
fn mul_fq6_by_v(x: &Fq6) -> Fq6 {
    Fq6::new(
        x.c2 * <ark_bn254::Fq6Config as Fp6Config>::NONRESIDUE,
        x.c0,
        x.c1,
    )
}

/// The product of two evaluated D-twist lines, collected over w² = v:
/// even part a full Fq6, odd part two populated slots (v² identically
/// zero) — 6 Fq2 muls with Karatsuba cross terms.
struct LinePair {
    c0: Fq6,
    c1a: Fq2,
    c1b: Fq2,
}

#[inline]
fn combine_lines(l1: &(Fq2, Fq2, Fq2), l2: &(Fq2, Fq2, Fq2)) -> LinePair {
    let (a1, b1, c1) = l1;
    let (a2, b2, c2) = l2;
    let aa = *a1 * a2;
    let bb = *b1 * b2;
    let cc = *c1 * c2;
    let ab = (*a1 + b1) * (*a2 + b2) - aa - bb;
    let ac = (*a1 + c1) * (*a2 + c2) - aa - cc;
    let bc = (*b1 + c1) * (*b2 + c2) - bb - cc;
    LinePair {
        c0: Fq6::new(
            aa + cc * <ark_bn254::Fq6Config as Fp6Config>::NONRESIDUE,
            bb,
            bc,
        ),
        c1a: ab,
        c1b: ac,
    }
}

/// f ← f·(l.c0 + (l.c1a + l.c1b·v)·w): quadratic Karatsuba with the odd
/// part's sparse `mul_by_01` leg — 17 Fq2 muls against 26 for two
/// `mul_by_034`s.
#[inline]
fn mul_by_line_pair(f: &mut Fq12, l: &LinePair) {
    let v0 = f.c0 * l.c0;
    let mut v1 = f.c1;
    v1.mul_by_01(&l.c1a, &l.c1b);
    let sum = Fq6::new(l.c0.c0 + l.c1a, l.c0.c1 + l.c1b, l.c0.c2);
    let mut c1 = f.c0 + f.c1;
    c1 *= &sum;
    c1 -= &v0;
    c1 -= &v1;
    f.c0 = v0 + mul_fq6_by_v(&v1);
    f.c1 = c1;
}

/// Evaluate coefficient `step` of every live pair at its G1 point and fold
/// the line values into `f`, combined pairwise (an odd tail falls back to
/// the single sparse mul).
#[inline]
fn fold_step(f: &mut Fq12, pairs: &[(G1Affine, &[EllCoeff])], step: usize) {
    let mut held: Option<(Fq2, Fq2, Fq2)> = None;
    for (point, coeffs) in pairs {
        let c = &coeffs[step];
        let mut c0 = c.0;
        c0.mul_assign_by_fp(&point.y);
        let mut c1 = c.1;
        c1.mul_assign_by_fp(&point.x);
        let line = (c0, c1, c.2);
        match held.take() {
            Some(h) => mul_by_line_pair(f, &combine_lines(&h, &line)),
            None => held = Some(line),
        }
    }
    if let Some((c0, c3, c4)) = held {
        f.mul_by_034(&c0, &c3, &c4);
    }
}

/// The multi-Miller Fp12 value of the given (non-identity) pairs over one
/// shared squaring ladder. Coefficient slices must be full arkworks
/// preparations (one per ate iteration, one more per nonzero digit, two
/// Frobenius steps) — asserted per pair by `fold_step` indexing.
fn multi_miller_prepared(pairs: &[(G1Affine, &[EllCoeff])]) -> Fq12 {
    let mut f = Fq12::one();
    if pairs.is_empty() {
        return f;
    }
    let ate = <ark_bn254::Config as BnConfig>::ATE_LOOP_COUNT;
    let iters = ate.len() - 1;
    let mut step = 0usize;
    for k in 0..iters {
        if k != 0 {
            let _ = f.square_in_place();
        }
        fold_step(&mut f, pairs, step);
        step += 1;
        if ate[iters - 1 - k] != 0 {
            fold_step(&mut f, pairs, step);
            step += 1;
        }
    }
    fold_step(&mut f, pairs, step);
    fold_step(&mut f, pairs, step + 1);
    f
}

/// The multi-Miller Fp12 value of affine pairs — the deterministic CPU
/// twin of the Metal fly kernel (its co-execution arm). Chunks on the
/// rayon pool; each chunk prepares its own G2 points once and runs ONE
/// shared squaring ladder ([`multi_miller_prepared`]), so per-pair cost is
/// the preparation + prepared-Miller rate with no dependence on arkworks'
/// internal 4-pair re-chunking (nondeterministically 3-4× slower on a
/// saturated pool). Identity pairs are filtered
/// exactly as arkworks' pair filter; any partition multiplies the same
/// Fp12 factors, so the value is bit-identical to `multi_miller_loop`.
pub fn multi_miller_affine(ps: &[G1Affine], qs: &[ark_bn254::G2Affine]) -> Fq12 {
    assert_eq!(ps.len(), qs.len());
    ps.par_chunks(MILLER_CHUNK)
        .zip(qs.par_chunks(MILLER_CHUNK))
        .map(|(ps_chunk, qs_chunk)| {
            let live: Vec<(G1Affine, <Bn254 as Pairing>::G2Prepared)> = ps_chunk
                .iter()
                .zip(qs_chunk)
                .filter(|(p, q)| !p.is_zero() && !q.is_zero())
                .map(|(p, q)| (*p, (*q).into()))
                .collect();
            let pairs: Vec<(G1Affine, &[EllCoeff])> = live
                .iter()
                .map(|(p, prep)| (*p, prep.ell_coeffs.as_slice()))
                .collect();
            multi_miller_prepared(&pairs)
        })
        .reduce(Fq12::one, |a, b| a * b)
}

/// The `multi_pair_g2_setup` Fp12 product of `ps` against the prepared
/// generator prefix, without the per-call G2 re-preparation.
pub(crate) fn multi_pair_g2_prepared(ps: &[ArkG1], prep: &DoryTier2Prep) -> ArkGT {
    assert!(
        ps.len() <= prep.rows,
        "tier-2 prep covers {} rows, finish needs {}",
        prep.rows,
        ps.len(),
    );
    let qs = &prep.prepared()[..ps.len()];
    let combined = ps
        .par_chunks(MILLER_CHUNK)
        .zip(qs.par_chunks(MILLER_CHUNK))
        .map(|(ps_chunk, qs_chunk)| {
            // Batch normalization is one inversion per chunk instead of one
            // per point; affine coordinates are the same field values.
            let affine =
                G1Projective::normalize_batch(&ps_chunk.iter().map(|p| p.0).collect::<Vec<_>>());
            let pairs: Vec<(G1Affine, &[EllCoeff])> = affine
                .iter()
                .zip(qs_chunk)
                .filter(|(p, q)| !p.is_zero() && !q.infinity)
                .map(|(p, q)| (*p, q.ell_coeffs.as_slice()))
                .collect();
            multi_miller_prepared(&pairs)
        })
        .reduce(Fq12::one, |a, b| a * b);
    let combined = MillerLoopOutput(combined);
    #[expect(
        clippy::expect_used,
        reason = "final exponentiation only fails on a zero Miller product, which no pair set produces"
    )]
    let result =
        Bn254::final_exponentiation(combined).expect("final exponentiation should not fail");
    dory::backends::arkworks::ArkGT(result)
}

/// [`commit_rows_tier_2`](crate::scheme::commit_rows_tier_2) against
/// pre-prepared generators — identical output.
pub(crate) fn commit_rows_tier_2_prepared<M: Mode>(
    row_commitments: &[ArkG1],
    setup: &DoryProverSetup,
    prep: &DoryTier2Prep,
) -> (ArkGT, ArkFr) {
    let tier_2 = multi_pair_g2_prepared(row_commitments, prep);
    let commit_blind = M::sample::<ArkFr>();
    let tier_2 = M::mask(tier_2, &setup.0.ht, &commit_blind);
    (tier_2, commit_blind)
}

/// Incremental transparent-mode tier-2: the Fp12 Miller product accumulated
/// as row commitments become final (any order, any partition — the product
/// commutes), final-exponentiated once. Produces the exact
/// `multi_pair_g2_setup` GT for the same row set: a multi-Miller loop equals
/// the per-pair Miller product, and identity rows are skipped by both paths
/// (`e(∞, ·) = 1`). This is the overlap seam for the Metal commit slot —
/// the CPU pairs each superchunk's finished rows while the device crunches
/// the next one.
pub struct Tier2Accumulator {
    miller: <Bn254 as Pairing>::TargetField,
}

impl Default for Tier2Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Tier2Accumulator {
    pub fn new() -> Self {
        Self {
            miller: <Bn254 as Pairing>::TargetField::one(),
        }
    }

    /// Multiply in the Miller loops of `points[i]` paired with prepared
    /// setup generator `row_indices[i]`.
    pub fn absorb(&mut self, prep: &DoryTier2Prep, points: &[G1Affine], row_indices: &[u32]) {
        assert_eq!(
            points.len(),
            row_indices.len(),
            "tier-2 absorb: one row index per point",
        );
        let prepared = prep.prepared();
        let pairs: Vec<(G1Affine, &[EllCoeff])> = points
            .iter()
            .zip(row_indices)
            .map(|(p, &row)| (p, &prepared[row as usize]))
            .filter(|(p, q)| !p.is_zero() && !q.infinity)
            .map(|(p, q)| (*p, q.ell_coeffs.as_slice()))
            .collect();
        self.miller *= multi_miller_prepared(&pairs);
    }

    /// Fold another accumulator in (parallel absorb lanes).
    pub fn merge(&mut self, other: Self) {
        self.miller *= other.miller;
    }

    /// Fold an externally computed partial Miller product in — the device
    /// Miller lane's absorb. Exact for the same reason `absorb` shards are:
    /// the multi-Miller value of a pair set is the Fp12 product of the
    /// values over any partition of it.
    pub fn merge_miller(&mut self, partial: <Bn254 as Pairing>::TargetField) {
        self.miller *= partial;
    }

    /// Final exponentiation — the transparent-mode tier-2 commitment.
    pub fn finish(self) -> DoryCommitment {
        #[expect(
            clippy::expect_used,
            reason = "final exponentiation only fails on a zero Miller product, which no pair set produces"
        )]
        let result = Bn254::final_exponentiation(MillerLoopOutput(self.miller))
            .expect("final exponentiation should not fail");
        DoryCommitment(ark_to_jolt_gt(&dory::backends::arkworks::ArkGT(result)))
    }
}

/// Assemble a one-hot column's output from externally computed row
/// commitments (tier-2 row order: `k · chunk_count + window`) and its
/// accumulated tier-2 — the transparent-mode counterpart of
/// `finish_one_hot_column_major_chunks` for rows that never passed through
/// chunk commitments.
pub fn one_hot_output_from_rows(
    setup: &DoryProverSetup,
    rows: Vec<Bn254G1>,
    tier2: Tier2Accumulator,
) -> (DoryCommitment, DoryHint) {
    crate::streaming::validate_row_count(rows.len(), setup);
    let commit_blind = dory::Transparent::sample::<ArkFr>();
    (
        tier2.finish(),
        DoryHint::new(rows, ark_to_jolt_fr(&commit_blind)),
    )
}

#[cfg(test)]
mod tests {
    use ark_ff::{UniformRand, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::scheme::commit_rows_tier_2;
    use crate::DoryScheme;

    /// The eager setup table covers exactly the consumer bound —
    /// 2^floor(max_num_vars/2) rows of the even-padded SRS (half the
    /// generators when max_num_vars is odd), the most rows any
    /// balanced-layout commit pairs against. Oversizing is silent RSS
    /// bloat (~16.7 kB per prepared point); undersizing silently degrades
    /// to per-pass preparation.
    #[test]
    fn setup_table_sized_to_consumer_bound() {
        let odd = DoryScheme::setup_prover(5);
        assert_eq!(odd.0.g2_vec.len(), 8);
        assert_eq!(odd.1.len(), 4);
        let even = DoryScheme::setup_prover(6);
        assert_eq!(even.0.g2_vec.len(), 8);
        assert_eq!(even.1.len(), 8);
    }

    /// The fresh shared-ladder, pairwise-combined loop must reproduce
    /// arkworks' multi-Miller value bit for bit at every pair-count shape
    /// (empty, single, odd tail, even) with identities mixed in.
    #[test]
    fn fresh_multi_miller_matches_arkworks() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x1077);
        for n in [0usize, 1, 2, 3, 7, 8, 64] {
            let mut points: Vec<G1Affine> = (0..n)
                .map(|_| ark_bn254::G1Projective::rand(&mut rng).into_affine())
                .collect();
            if n > 4 {
                points[1] = G1Affine::zero();
                points[4] = G1Affine::zero();
            }
            let preps: Vec<<Bn254 as Pairing>::G2Prepared> = (0..n)
                .map(|_| ark_bn254::G2Projective::rand(&mut rng).into_affine().into())
                .collect();

            let expected =
                Bn254::multi_miller_loop(points.iter().copied(), preps.iter().cloned()).0;
            let pairs: Vec<(G1Affine, &[EllCoeff])> = points
                .iter()
                .zip(&preps)
                .filter(|(p, q)| !p.is_zero() && !q.infinity)
                .map(|(p, q)| (*p, q.ell_coeffs.as_slice()))
                .collect();
            assert_eq!(
                multi_miller_prepared(&pairs),
                expected,
                "fresh loop diverged at {n} pairs"
            );
        }
    }

    /// The affine co-execution entry must reproduce arkworks'
    /// multi-Miller value bit for bit across chunk boundaries (the
    /// MILLER_CHUNK partition), with identities on both sides.
    #[test]
    fn multi_miller_affine_matches_arkworks() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x0aff);
        for n in [
            1usize,
            MILLER_CHUNK - 1,
            MILLER_CHUNK,
            2 * MILLER_CHUNK + 17,
        ] {
            let mut ps: Vec<G1Affine> = (0..n)
                .map(|_| ark_bn254::G1Projective::rand(&mut rng).into_affine())
                .collect();
            let mut qs: Vec<ark_bn254::G2Affine> = (0..n)
                .map(|_| ark_bn254::G2Projective::rand(&mut rng).into_affine())
                .collect();
            if n > 4 {
                ps[2] = G1Affine::zero();
                qs[4] = ark_bn254::G2Affine::zero();
                qs[n - 1] = ark_bn254::G2Affine::zero();
            }
            let expected = Bn254::multi_miller_loop(
                ps.iter().copied(),
                qs.iter().map(|q| <Bn254 as Pairing>::G2Prepared::from(*q)),
            )
            .0;
            assert_eq!(
                multi_miller_affine(&ps, &qs),
                expected,
                "affine loop diverged at {n} pairs"
            );
        }
    }

    /// Absorbing rows in shards, out of order, across merged accumulators
    /// must reproduce `commit_rows_tier_2`'s GT exactly — including skipped
    /// identity rows.
    #[test]
    fn accumulator_matches_whole_row_tier_2() {
        let num_rows = 16usize;
        let setup = DoryScheme::setup_prover(8);
        let mut rng = ChaCha20Rng::seed_from_u64(77);
        let mut rows: Vec<ArkG1> = (0..num_rows)
            .map(|_| dory::backends::arkworks::ArkG1(ark_bn254::G1Projective::rand(&mut rng)))
            .collect();
        rows[3] = dory::backends::arkworks::ArkG1(ark_bn254::G1Projective::zero());
        rows[10] = dory::backends::arkworks::ArkG1(ark_bn254::G1Projective::zero());

        let (expected_gt, _) = commit_rows_tier_2::<dory::Transparent>(&rows, &setup);

        let prep = DoryTier2Prep::new(&setup, num_rows);
        let affine: Vec<G1Affine> = rows.iter().map(|p| p.0.into_affine()).collect();
        // Shard the rows into interleaved index sets absorbed out of order,
        // one accumulator per parity, merged at the end.
        let mut even = Tier2Accumulator::new();
        let mut odd = Tier2Accumulator::new();
        let evens: Vec<u32> = (0..num_rows as u32).filter(|i| i % 2 == 0).rev().collect();
        let odds: Vec<u32> = (0..num_rows as u32).filter(|i| i % 2 == 1).collect();
        odd.absorb(
            &prep,
            &odds.iter().map(|&i| affine[i as usize]).collect::<Vec<_>>(),
            &odds,
        );
        for &i in &evens {
            even.absorb(&prep, &[affine[i as usize]], &[i]);
        }
        even.merge(odd);
        let device_gt = even.finish();

        assert_eq!(device_gt.0, crate::scheme::ark_to_jolt_gt(&expected_gt));

        // The assembled output: same GT, the given rows, zero blind.
        let jolt_rows: Vec<Bn254G1> = affine
            .iter()
            .map(|a| {
                let point: ark_bn254::G1Projective = (*a).into();
                Bn254G1::from(point)
            })
            .collect();
        let mut acc = Tier2Accumulator::new();
        let all: Vec<u32> = (0..num_rows as u32).collect();
        acc.absorb(&prep, &affine, &all);
        let (commitment, hint) = one_hot_output_from_rows(&setup, jolt_rows.clone(), acc);
        assert_eq!(commitment.0, crate::scheme::ark_to_jolt_gt(&expected_gt));
        let zero_blind = ark_to_jolt_fr(&dory::Transparent::sample::<ArkFr>());
        assert_eq!(hint, DoryHint::new(jolt_rows, zero_blind));
    }
}
