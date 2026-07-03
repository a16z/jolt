//! Equivalence of the sparse one-hot packed-reduction prover with the dense
//! oracle path: identical round polynomials, transcripts, and proofs, without
//! the sparse prover ever materializing the packed evaluation table.

#![expect(clippy::expect_used, reason = "tests assert successful proof paths")]
#![expect(clippy::unwrap_used, reason = "tests unwrap successful PCS operations")]

use jolt_crypto::Commitment;
use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, PackedBatch, PrefixPackedProverSetup,
    PrefixPackedStatement, PrefixPackedVerifierSetup, PrefixPacking,
};
use jolt_poly::{MultilinearPoly, OneHotPolynomial};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

#[path = "support/common.rs"]
pub mod common;
#[path = "support/packed.rs"]
pub mod packed_support;

use common::fr;
use packed_support::{
    independent_claims, materialize_packed, one_hot_packed_fixture, one_hot_packed_polynomials,
    packed_claims, MaterializedPackedWitness, PackedId,
};

type PackedDoryBatch = PackedBatch<DoryScheme, PackedId>;
type DoryOutput = <DoryScheme as Commitment>::Output;
type PackedStatement = PrefixPackedStatement<Fr, PackedId, DoryOutput>;

fn packed_setup(
    packing: PrefixPacking<PackedId>,
) -> (
    PrefixPackedProverSetup<DoryScheme, PackedId>,
    PrefixPackedVerifierSetup<DoryScheme, PackedId>,
) {
    let prover_pcs = DoryScheme::setup_prover(packing.packed_num_vars);
    let verifier_pcs = DoryScheme::setup_verifier(packing.packed_num_vars);
    (
        PrefixPackedProverSetup {
            pcs: prover_pcs,
            packing: packing.clone(),
        },
        PrefixPackedVerifierSetup {
            pcs: verifier_pcs,
            packing,
        },
    )
}

/// Proves the same statement through the dense oracle path (materialized
/// `Polynomial`) and the sparse path (`OneHotPolynomial`), asserting equal
/// commitments, proofs, and prover transcript states, then verifies the
/// sparse proof and asserts verifier/prover transcript state equality.
fn assert_sparse_matches_dense_oracle(
    witness: &OneHotPolynomial,
    packed: &MaterializedPackedWitness<PackedId, Fr>,
    claims: Vec<(PackedId, EvaluationClaim<Fr>)>,
    label: &'static [u8],
) {
    assert_eq!(
        MultilinearPoly::<Fr>::to_dense(witness).as_ref(),
        packed.polynomial.to_dense().as_ref(),
        "one-hot fixture must scatter to the materialized packed witness"
    );

    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let (dense_commitment, dense_hint) =
        DoryScheme::commit(&packed.polynomial, &prover_setup.pcs).unwrap();
    let (sparse_commitment, sparse_hint) = DoryScheme::commit(witness, &prover_setup.pcs).unwrap();
    assert_eq!(dense_commitment, sparse_commitment);

    let statement = PrefixPackedStatement::new(dense_commitment, claims);

    let mut dense_transcript = Blake2bTranscript::new(label);
    let dense_proof = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        &packed.polynomial,
        dense_hint,
        &mut dense_transcript,
    )
    .expect("dense oracle proof should be produced");

    let mut sparse_transcript = Blake2bTranscript::new(label);
    let sparse_proof = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        witness,
        sparse_hint,
        &mut sparse_transcript,
    )
    .expect("sparse one-hot proof should be produced");

    assert_eq!(
        dense_proof.round_polynomials, sparse_proof.round_polynomials,
        "sparse round polynomials must be field-identical to the dense path"
    );
    assert_eq!(dense_proof.opening_eval, sparse_proof.opening_eval);
    assert_eq!(dense_proof, sparse_proof);
    assert_eq!(
        dense_transcript.state(),
        sparse_transcript.state(),
        "sparse prover transcript must be byte-identical to the dense path"
    );

    let mut verifier_transcript = Blake2bTranscript::new(label);
    <PackedDoryBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        statement,
        &sparse_proof,
        &mut verifier_transcript,
    )
    .expect("sparse one-hot proof should verify");
    assert_eq!(sparse_transcript.state(), verifier_transcript.state());
}

#[test]
fn dory_sparse_one_hot_matches_dense_oracle_at_independent_points() {
    let (witness, polynomials) = one_hot_packed_polynomials(4, 0x5eed);
    let packed = materialize_packed(&polynomials).expect("packed witness should build");
    let mut rng = ChaCha20Rng::seed_from_u64(0xabad_1dea);
    let claims = independent_claims(&polynomials, &mut rng);
    assert_sparse_matches_dense_oracle(&witness, &packed, claims, b"dory-sparse-independent");
}

/// `k = 1` makes every packed cell its own row-group, giving an arbitrary 0/1
/// pattern with enough one-positions to trigger the mid-protocol switch to
/// the dense tail rounds.
#[test]
fn dory_sparse_dense_pattern_switchover_matches_dense_oracle() {
    let (witness, polynomials) = one_hot_packed_polynomials(1, 0x0a11_ce55);
    assert!(
        MultilinearPoly::<Fr>::to_dense(&witness)
            .iter()
            .filter(|value| **value == fr(1))
            .count()
            > 8,
        "fixture must have enough ones to reach the dense tail"
    );
    let packed = materialize_packed(&polynomials).expect("packed witness should build");
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    assert_sparse_matches_dense_oracle(&witness, &packed, claims, b"dory-sparse-switchover");
}

/// A single one-position never triggers the dense tail, exercising the fully
/// sparse rounds and the sparse final opening evaluation.
#[test]
fn dory_sparse_single_one_witness_stays_sparse_all_rounds() {
    let indices = vec![None, Some(2), None, None, None, None, None, None];
    let (witness, polynomials) = one_hot_packed_fixture(4, indices);
    let packed = materialize_packed(&polynomials).expect("packed witness should build");
    let mut rng = ChaCha20Rng::seed_from_u64(0x51_46_1e);
    let claims = independent_claims(&polynomials, &mut rng);
    assert_sparse_matches_dense_oracle(&witness, &packed, claims, b"dory-sparse-single-one");
}

#[test]
fn dory_sparse_all_zero_witness_roundtrip() {
    let (witness, polynomials) = one_hot_packed_fixture(4, vec![None; 8]);
    let packed = materialize_packed(&polynomials).expect("packed witness should build");
    let mut rng = ChaCha20Rng::seed_from_u64(0x0);
    let claims = independent_claims(&polynomials, &mut rng);
    assert!(claims.iter().all(|(_, claim)| claim.value == fr(0)));
    assert_sparse_matches_dense_oracle(&witness, &packed, claims, b"dory-sparse-all-zero");
}

#[test]
fn dory_sparse_rejects_tampered_value() {
    let (witness, polynomials) = one_hot_packed_polynomials(4, 0x7a3b);
    let packed = materialize_packed(&polynomials).expect("packed witness should build");
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&witness, &prover_setup.pcs).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(0xbad);
    let mut claims = independent_claims(&polynomials, &mut rng);
    claims[0].1.value += fr(1);
    let statement = PrefixPackedStatement::new(commitment, claims);

    let mut prover_transcript = Blake2bTranscript::new(b"dory-sparse-tampered");
    let proof = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        &witness,
        hint,
        &mut prover_transcript,
    )
    .expect("prover does not validate claim values");

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-sparse-tampered");
    let result = <PackedDoryBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        statement,
        &proof,
        &mut verifier_transcript,
    );
    assert!(
        result.is_err(),
        "tampered value should fail on the sparse path"
    );
}

/// Forwards only the sparse interface; any dense-row access panics. Proving
/// through this wrapper fails loudly if the sparse path (or Dory's one-hot
/// commit/open) ever materializes the packed evaluation table.
struct SparseOnly<'a>(&'a OneHotPolynomial);

impl MultilinearPoly<Fr> for SparseOnly<'_> {
    fn num_vars(&self) -> usize {
        MultilinearPoly::<Fr>::num_vars(self.0)
    }

    fn evaluate(&self, point: &[Fr]) -> Fr {
        self.0.evaluate(point)
    }

    #[expect(clippy::panic, reason = "guard against dense materialization")]
    fn for_each_row(&self, _sigma: usize, _f: &mut dyn FnMut(usize, &[Fr])) {
        panic!("sparse packed prover must not materialize dense rows");
    }

    fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
        self.0.fold_rows(left, sigma)
    }

    fn is_one_hot(&self) -> bool {
        true
    }

    fn for_each_one(&self, f: &mut dyn FnMut(usize)) {
        MultilinearPoly::<Fr>::for_each_one(self.0, f);
    }
}

#[test]
fn dory_sparse_prover_never_materializes_dense_witness() {
    let (witness, polynomials) = one_hot_packed_polynomials(4, 0xdead_beef);
    let packed = materialize_packed(&polynomials).expect("packed witness should build");
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let guarded = SparseOnly(&witness);
    let (commitment, hint) = DoryScheme::commit(&guarded, &prover_setup.pcs).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(0x90_0d);
    let claims = independent_claims(&polynomials, &mut rng);
    let statement: PackedStatement = PrefixPackedStatement::new(commitment, claims);

    let mut prover_transcript = Blake2bTranscript::new(b"dory-sparse-no-densify");
    let proof = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        &guarded,
        hint,
        &mut prover_transcript,
    )
    .expect("sparse proof should be produced without dense access");

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-sparse-no-densify");
    <PackedDoryBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        statement,
        &proof,
        &mut verifier_transcript,
    )
    .expect("sparse proof should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}
