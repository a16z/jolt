//! Shared tier-2 pairing state for a streaming-commit pass.
//!
//! Every column finish pairs its row commitments against the SAME prefix of
//! the setup's fixed G2 generators, but `multi_pair_g2_setup` re-runs the G2
//! Miller-loop preparation on every call — for a 42-column witness commit
//! that is 42× redundant preparation of identical points (~40% of the tier-2
//! wall). [`DoryTier2Prep`] prepares the prefix once per pass and the
//! `*_prepared` finish variants pair against it.
//!
//! Unlike dory-pcs's global prepared-point cache (deliberately not primed —
//! see `DoryScheme::setup_prover`), this state is scoped to one pass over one
//! setup, so two setup sizes in one process cannot cross-contaminate.
//!
//! Chunked Miller loops multiplied in Fp12 and final-exponentiated once give
//! the exact `multi_pair_g2_setup` value for any chunking: a multi-Miller
//! loop over a pair set equals the Fp12 product of the per-pair Miller
//! values, and final exponentiation is applied to the full product either
//! way. Commitments and hints are therefore byte-identical to the unprepared
//! path (pinned by parity tests in `streaming`).

use ark_bn254::{Bn254, G1Affine};
use ark_ec::pairing::{MillerLoopOutput, Pairing};
use ark_ec::CurveGroup;
use ark_ff::One;
use dory::Mode;
use jolt_crypto::Bn254G1;
use rayon::prelude::*;

use crate::scheme::{ark_to_jolt_fr, ark_to_jolt_gt, ArkFr, ArkG1, ArkGT};
use crate::types::{DoryCommitment, DoryHint, DoryProverSetup};

/// Miller-loop-prepared `g2_vec[..max_rows]` setup generators, shared
/// read-only across every column finish of one commit pass.
pub struct DoryTier2Prep {
    prepared: Vec<<Bn254 as Pairing>::G2Prepared>,
}

/// Pairs per parallel Miller-loop chunk. Value-invariant (see module docs);
/// sized so a chunk's G1 preparation stays cache-resident while giving rayon
/// enough tasks per column finish.
const MILLER_CHUNK: usize = 512;

impl DoryTier2Prep {
    pub(crate) fn new(setup: &DoryProverSetup, max_rows: usize) -> Self {
        assert!(
            max_rows <= setup.0.g2_vec.len(),
            "tier-2 prep: row count ({}) exceeds Dory SRS size ({})",
            max_rows,
            setup.0.g2_vec.len(),
        );
        let prepared = setup.0.g2_vec[..max_rows]
            .par_iter()
            .map(|q| q.0.into_affine().into())
            .collect();
        Self { prepared }
    }

    /// The prepared generator prefix — the raw line-coefficient vectors a
    /// device Miller lane stages into its own layout.
    pub fn prepared(&self) -> &[<Bn254 as Pairing>::G2Prepared] {
        &self.prepared
    }
}

/// The `multi_pair_g2_setup` Fp12 product of `ps` against the prepared
/// generator prefix, without the per-call G2 re-preparation.
pub(crate) fn multi_pair_g2_prepared(ps: &[ArkG1], prep: &DoryTier2Prep) -> ArkGT {
    assert!(
        ps.len() <= prep.prepared.len(),
        "tier-2 prep covers {} rows, finish needs {}",
        prep.prepared.len(),
        ps.len(),
    );
    let qs = &prep.prepared[..ps.len()];
    let combined = ps
        .par_chunks(MILLER_CHUNK)
        .zip(qs.par_chunks(MILLER_CHUNK))
        .map(|(ps_chunk, qs_chunk)| {
            let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> =
                ps_chunk.iter().map(|p| p.0.into_affine().into()).collect();
            Bn254::multi_miller_loop(ps_prep, qs_chunk.iter().cloned())
        })
        .reduce(
            || MillerLoopOutput(<Bn254 as Pairing>::TargetField::one()),
            |a, b| MillerLoopOutput(a.0 * b.0),
        );
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
        if points.is_empty() {
            return;
        }
        let qs: Vec<<Bn254 as Pairing>::G2Prepared> = row_indices
            .iter()
            .map(|&row| prep.prepared[row as usize].clone())
            .collect();
        let out = Bn254::multi_miller_loop(points.iter().copied(), qs);
        self.miller *= out.0;
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
