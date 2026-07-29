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

use ark_bn254::Bn254;
use ark_ec::pairing::{MillerLoopOutput, Pairing};
use ark_ec::CurveGroup;
use ark_ff::One;
use dory::Mode;
use rayon::prelude::*;

use crate::scheme::{ArkFr, ArkG1, ArkGT};
use crate::types::DoryProverSetup;

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
