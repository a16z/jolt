//! Integration tests for the Dory commitment scheme.
//!
//! Public-API-only tests — no `pub(crate)` imports. Exercises commit, open,
//! verify, source-batch commitment, combine, and negative cases across transcript backends.

#![expect(clippy::expect_used, reason = "tests may panic on assertion failures")]

use std::sync::atomic::{AtomicUsize, Ordering};

use dory::backends::arkworks::ArkG1;
use jolt_dory::DoryScheme;
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_openings::{
    AdditivelyHomomorphic, AdditivelyHomomorphicVerifier, BatchCommitmentSource, CommitmentScheme,
    CommitmentSchemeVerifier, CommitmentSource, OneHotEntries, OneHotIndex, OneHotRow, SourceRow,
    ZkOpeningScheme, ZkOpeningSchemeVerifier,
};
use jolt_poly::{MultilinearPoly, OneHotPolynomial, Polynomial};
use jolt_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn round_trip<T: Transcript<Challenge = Fr>>(num_vars: usize, seed: u64, label: &'static [u8]) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

    // With hint
    let mut pt = T::new(label);
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt);

    let mut vt = T::new(label);
    DoryScheme::verify(&commitment, &point, eval, &proof, &verifier_setup, &mut vt)
        .expect("round-trip verification (with hint) must succeed");

    // Without hint
    let mut pt2 = T::new(label);
    let proof2 = DoryScheme::open(&poly, &point, eval, &prover_setup, None, &mut pt2);

    let mut vt2 = T::new(label);
    DoryScheme::verify(
        &commitment,
        &point,
        eval,
        &proof2,
        &verifier_setup,
        &mut vt2,
    )
    .expect("round-trip verification (without hint) must succeed");
}

#[test]
fn commit_open_verify_various_sizes() {
    for num_vars in [2, 3, 4, 6] {
        round_trip::<Blake2bTranscript>(num_vars, 100 + num_vars as u64, b"cov-sizes");
    }
}

#[test]
fn commit_open_verify_both_transcripts() {
    let num_vars = 4;
    round_trip::<Blake2bTranscript>(num_vars, 200, b"blake2b-rt");
    round_trip::<KeccakTranscript>(num_vars, 200, b"keccak-rt");
}

#[test]
fn one_hot_commitment_matches_dense() {
    let num_vars = 4;
    let k = 4;
    let indices = vec![Some(2), None, Some(0), Some(3)];
    let mut evals = vec![Fr::from_u64(0); 1 << num_vars];
    for (row, col) in indices.iter().enumerate() {
        if let Some(col) = col {
            evals[row * k + *col as usize] = Fr::from_u64(1);
        }
    }

    let one_hot = OneHotPolynomial::new(k, indices);
    let dense = Polynomial::new(evals);
    let prover_setup = DoryScheme::setup_prover(num_vars);

    let (one_hot_commitment, _) = DoryScheme::commit(&one_hot, &prover_setup);
    let (dense_commitment, _) = DoryScheme::commit(dense.evaluations(), &prover_setup);

    assert_eq!(
        one_hot_commitment, dense_commitment,
        "one-hot commitment must match the equivalent dense table"
    );
}

struct DenseSource<'a> {
    evaluations: &'a [Fr],
}

impl CommitmentSource<Fr> for DenseSource<'_> {
    fn num_vars(&self) -> usize {
        self.evaluations.len().ilog2() as usize
    }

    fn evaluate(&self, point: &[Fr]) -> Fr {
        Polynomial::new(self.evaluations.to_vec()).evaluate(point)
    }

    fn for_each_row<V>(&self, chunk_len: usize, mut visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, Fr>),
    {
        for (row_index, row) in self.evaluations.chunks(chunk_len).enumerate() {
            visit(row_index, SourceRow::FieldElements(row));
        }
    }

    fn fold_rows(&self, left: &[Fr], chunk_len: usize) -> Vec<Fr> {
        let sigma = chunk_len.trailing_zeros() as usize;
        let poly = Polynomial::new(self.evaluations.to_vec());
        MultilinearPoly::fold_rows(&poly, left, sigma)
    }
}

struct DenseBatch {
    ids: Vec<usize>,
    evaluations: Vec<Vec<Fr>>,
    map_rows_calls: AtomicUsize,
}

impl DenseBatch {
    fn new(evaluations: Vec<Vec<Fr>>) -> Self {
        Self {
            ids: (0..evaluations.len()).collect(),
            evaluations,
            map_rows_calls: AtomicUsize::new(0),
        }
    }
}

impl BatchCommitmentSource<Fr> for DenseBatch {
    type Id = usize;

    type Source<'a>
        = DenseSource<'a>
    where
        Self: 'a;

    fn source_ids(&self) -> &[Self::Id] {
        &self.ids
    }

    fn num_vars(&self, id: Self::Id) -> usize {
        self.evaluations[id].len().ilog2() as usize
    }

    fn source(&self, id: Self::Id) -> Self::Source<'_> {
        DenseSource {
            evaluations: &self.evaluations[id],
        }
    }

    fn map_rows<R, V>(&self, chunk_len: usize, ids: &[Self::Id], visit: V) -> Vec<Vec<R>>
    where
        R: Send,
        V: for<'row> Fn(Self::Id, SourceRow<'row, Fr>) -> R + Send + Sync,
    {
        let _ = self.map_rows_calls.fetch_add(1, Ordering::SeqCst);
        let num_rows = self.evaluations[ids[0]].len() / chunk_len;

        (0..num_rows)
            .map(|row_index| {
                ids.iter()
                    .map(|&id| {
                        let start = row_index * chunk_len;
                        let end = start + chunk_len;
                        visit(
                            id,
                            SourceRow::FieldElements(&self.evaluations[id][start..end]),
                        )
                    })
                    .collect()
            })
            .collect()
    }
}

struct I128Batch {
    ids: Vec<usize>,
    rows: Vec<Vec<i128>>,
    dense: Vec<Vec<Fr>>,
    map_rows_calls: AtomicUsize,
}

impl I128Batch {
    fn new(rows: Vec<Vec<i128>>) -> Self {
        let dense = rows
            .iter()
            .map(|row| row.iter().map(|&value| Fr::from_i128(value)).collect())
            .collect();
        Self {
            ids: (0..rows.len()).collect(),
            rows,
            dense,
            map_rows_calls: AtomicUsize::new(0),
        }
    }
}

impl BatchCommitmentSource<Fr> for I128Batch {
    type Id = usize;

    type Source<'a>
        = DenseSource<'a>
    where
        Self: 'a;

    fn source_ids(&self) -> &[Self::Id] {
        &self.ids
    }

    fn num_vars(&self, id: Self::Id) -> usize {
        self.rows[id].len().ilog2() as usize
    }

    fn source(&self, id: Self::Id) -> Self::Source<'_> {
        DenseSource {
            evaluations: &self.dense[id],
        }
    }

    fn map_rows<R, V>(&self, chunk_len: usize, ids: &[Self::Id], visit: V) -> Vec<Vec<R>>
    where
        R: Send,
        V: for<'row> Fn(Self::Id, SourceRow<'row, Fr>) -> R + Send + Sync,
    {
        let _ = self.map_rows_calls.fetch_add(1, Ordering::SeqCst);
        let num_rows = self.rows[ids[0]].len() / chunk_len;

        (0..num_rows)
            .map(|row_index| {
                ids.iter()
                    .map(|&id| {
                        let start = row_index * chunk_len;
                        let end = start + chunk_len;
                        visit(id, SourceRow::I128(&self.rows[id][start..end]))
                    })
                    .collect()
            })
            .collect()
    }
}

#[test]
fn commit_batch_dense_matches_direct_and_uses_shared_rows() {
    let num_vars = 4;
    let mut rng = ChaCha20Rng::seed_from_u64(375);
    let prover_setup = DoryScheme::setup_prover(num_vars);

    let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);
    let batch = DenseBatch::new(vec![
        poly_a.evaluations().to_vec(),
        poly_b.evaluations().to_vec(),
    ]);

    let results = DoryScheme::commit_batch(&batch, batch.source_ids(), &prover_setup);
    assert_eq!(
        batch.map_rows_calls.load(Ordering::SeqCst),
        1,
        "Dory batch commitment should use one shared row traversal",
    );

    for (id, poly) in [poly_a, poly_b].into_iter().enumerate() {
        let (direct, _) = DoryScheme::commit(poly.evaluations(), &prover_setup);
        assert_eq!(results[id].0, direct);

        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);
        let mut pt = Blake2bTranscript::new(b"batch-dense");
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(results[id].1.clone()),
            &mut pt,
        );
        let mut vt = Blake2bTranscript::new(b"batch-dense");
        let verifier_setup = DoryScheme::setup_verifier(num_vars);
        DoryScheme::verify(
            &results[id].0,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut vt,
        )
        .expect("batch dense commitment hint should open and verify");
    }
}

#[test]
fn commit_batch_i128_matches_dense_commitment() {
    let num_vars = 4;
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let batch = I128Batch::new(vec![
        vec![0, 1, -1, 7, -3, 0, 12, -8, 4, 5, -6, 0, 9, -2, 3, 1],
        vec![2, -4, 0, 0, 11, -9, 5, 6, -1, 8, 0, -7, 3, 3, -2, 10],
    ]);

    let results = DoryScheme::commit_batch(&batch, batch.source_ids(), &prover_setup);
    assert_eq!(batch.map_rows_calls.load(Ordering::SeqCst), 1);

    for (id, dense) in batch.dense.iter().enumerate() {
        let (direct, _) = DoryScheme::commit(dense, &prover_setup);
        assert_eq!(results[id].0, direct);
    }
}

#[test]
fn commit_batch_zk_dense_outputs_openable_hints() {
    let num_vars = 4;
    let mut rng = ChaCha20Rng::seed_from_u64(376);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);
    let batch = DenseBatch::new(vec![
        poly_a.evaluations().to_vec(),
        poly_b.evaluations().to_vec(),
    ]);

    let results = DoryScheme::commit_batch_zk(&batch, batch.source_ids(), &prover_setup);
    assert_eq!(batch.map_rows_calls.load(Ordering::SeqCst), 1);

    for (id, poly) in [poly_a, poly_b].into_iter().enumerate() {
        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);
        let mut pt = Blake2bTranscript::new(b"batch-zk-dense");
        let (proof, _, _) = DoryScheme::open_zk(
            &poly,
            &point,
            eval,
            &prover_setup,
            results[id].1.clone(),
            &mut pt,
        );
        let mut vt = Blake2bTranscript::new(b"batch-zk-dense");
        DoryScheme::verify_zk(&results[id].0, &point, &proof, &verifier_setup, &mut vt)
            .expect("batch ZK dense commitment hint should open and verify");
    }
}

struct OneHotBatch {
    ids: Vec<usize>,
    log_domain_size: u8,
    chunks: Vec<Vec<Vec<Option<OneHotIndex>>>>,
    dense: Vec<Vec<Fr>>,
    map_rows_calls: AtomicUsize,
}

impl OneHotBatch {
    fn new(log_domain_size: u8, chunks: Vec<Vec<Vec<Option<OneHotIndex>>>>) -> Self {
        let domain_size = 1usize << log_domain_size;
        let dense = chunks
            .iter()
            .map(|source_chunks| {
                let row_len = source_chunks[0].len();
                let trace_len = source_chunks.len() * row_len;
                let mut evals = vec![Fr::from_u64(0); trace_len * domain_size];
                for (chunk_index, chunk) in source_chunks.iter().enumerate() {
                    assert_eq!(chunk.len(), row_len);
                    for (column, hot_index) in chunk.iter().enumerate() {
                        if let Some(hot_index) = hot_index {
                            evals[hot_index.get() * trace_len + chunk_index * row_len + column] =
                                Fr::from_u64(1);
                        }
                    }
                }
                evals
            })
            .collect();
        Self {
            ids: (0..chunks.len()).collect(),
            log_domain_size,
            chunks,
            dense,
            map_rows_calls: AtomicUsize::new(0),
        }
    }
}

impl BatchCommitmentSource<Fr> for OneHotBatch {
    type Id = usize;

    type Source<'a>
        = DenseSource<'a>
    where
        Self: 'a;

    fn source_ids(&self) -> &[Self::Id] {
        &self.ids
    }

    fn num_vars(&self, id: Self::Id) -> usize {
        self.dense[id].len().ilog2() as usize
    }

    fn source(&self, id: Self::Id) -> Self::Source<'_> {
        DenseSource {
            evaluations: &self.dense[id],
        }
    }

    fn natural_chunk_len(&self, ids: &[Self::Id]) -> Option<usize> {
        ids.first()
            .and_then(|&id| self.chunks[id].first().map(Vec::len))
    }

    fn map_rows<R, V>(&self, _chunk_len: usize, ids: &[Self::Id], visit: V) -> Vec<Vec<R>>
    where
        R: Send,
        V: for<'row> Fn(Self::Id, SourceRow<'row, Fr>) -> R + Send + Sync,
    {
        let _ = self.map_rows_calls.fetch_add(1, Ordering::SeqCst);
        (0..self.chunks[ids[0]].len())
            .map(|chunk_index| {
                ids.iter()
                    .map(|&id| {
                        visit(
                            id,
                            SourceRow::OneHot(OneHotRow {
                                log_domain_size: self.log_domain_size,
                                entries: OneHotEntries::MaybeZero(&self.chunks[id][chunk_index]),
                            }),
                        )
                    })
                    .collect()
            })
            .collect()
    }
}

#[test]
fn commit_batch_one_hot_matches_streaming_dense_layout() {
    let num_vars = 4;
    let log_domain_size = 2;
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let rows = vec![
        vec![vec![
            Some(OneHotIndex::new(2, log_domain_size).expect("valid index")),
            None,
            Some(OneHotIndex::new(0, log_domain_size).expect("valid index")),
            Some(OneHotIndex::new(3, log_domain_size).expect("valid index")),
        ]],
        vec![vec![
            Some(OneHotIndex::new(1, log_domain_size).expect("valid index")),
            Some(OneHotIndex::new(0, log_domain_size).expect("valid index")),
            None,
            Some(OneHotIndex::new(2, log_domain_size).expect("valid index")),
        ]],
    ];
    let batch = OneHotBatch::new(log_domain_size, rows);

    let results = DoryScheme::commit_batch(&batch, batch.source_ids(), &prover_setup);
    assert_eq!(batch.map_rows_calls.load(Ordering::SeqCst), 1);

    for (id, dense) in batch.dense.iter().enumerate() {
        let (direct, _) = DoryScheme::commit(dense, &prover_setup);
        assert_eq!(results[id].0, direct);
    }
}

#[test]
fn commit_batch_one_hot_matches_multi_chunk_streaming_layout() {
    let num_vars = 6;
    let log_domain_size = 2;
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let source_chunks = vec![vec![
        vec![
            Some(OneHotIndex::new(0, log_domain_size).expect("valid index")),
            Some(OneHotIndex::new(1, log_domain_size).expect("valid index")),
            None,
            Some(OneHotIndex::new(3, log_domain_size).expect("valid index")),
            Some(OneHotIndex::new(2, log_domain_size).expect("valid index")),
            None,
            Some(OneHotIndex::new(1, log_domain_size).expect("valid index")),
            Some(OneHotIndex::new(0, log_domain_size).expect("valid index")),
        ],
        vec![
            None,
            Some(OneHotIndex::new(2, log_domain_size).expect("valid index")),
            Some(OneHotIndex::new(3, log_domain_size).expect("valid index")),
            Some(OneHotIndex::new(0, log_domain_size).expect("valid index")),
            None,
            Some(OneHotIndex::new(1, log_domain_size).expect("valid index")),
            Some(OneHotIndex::new(2, log_domain_size).expect("valid index")),
            Some(OneHotIndex::new(3, log_domain_size).expect("valid index")),
        ],
    ]];
    let batch = OneHotBatch::new(log_domain_size, source_chunks);

    let results = DoryScheme::commit_batch(&batch, batch.source_ids(), &prover_setup);
    assert_eq!(batch.map_rows_calls.load(Ordering::SeqCst), 1);

    let (direct, _) = DoryScheme::commit(&batch.dense[0], &prover_setup);
    assert_eq!(results[0].0, direct);
}

#[test]
fn wrong_eval_rejected() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(400);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

    let mut pt = Blake2bTranscript::new(b"wrong-eval");
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt);

    let tampered_eval = eval + Fr::from_u64(1);
    let mut vt = Blake2bTranscript::new(b"wrong-eval");
    let result = DoryScheme::verify(
        &commitment,
        &point,
        tampered_eval,
        &proof,
        &verifier_setup,
        &mut vt,
    );
    assert!(result.is_err(), "tampered eval must be rejected");
}

#[test]
fn wrong_point_rejected() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(500);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

    let mut pt = Blake2bTranscript::new(b"wrong-point");
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt);

    let mut tampered_point = point.clone();
    tampered_point[0] += Fr::from_u64(1);
    let mut vt = Blake2bTranscript::new(b"wrong-point");
    let result = DoryScheme::verify(
        &commitment,
        &tampered_point,
        eval,
        &proof,
        &verifier_setup,
        &mut vt,
    );
    assert!(result.is_err(), "tampered point must be rejected");
}

#[test]
fn combine_linear_combination() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(600);

    let prover_setup = DoryScheme::setup_prover(num_vars);

    let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);

    let (commit_a, _) = DoryScheme::commit(poly_a.evaluations(), &prover_setup);
    let (commit_b, _) = DoryScheme::commit(poly_b.evaluations(), &prover_setup);

    let c1 = <Fr as RandomSampling>::random(&mut rng);
    let c2 = <Fr as RandomSampling>::random(&mut rng);

    let combined = DoryScheme::combine(&[commit_a, commit_b], &[c1, c2]);

    let weighted_evals: Vec<Fr> = poly_a
        .evaluations()
        .iter()
        .zip(poly_b.evaluations().iter())
        .map(|(a, b)| c1 * *a + c2 * *b)
        .collect();
    let (commit_weighted, _) = DoryScheme::commit(&weighted_evals, &prover_setup);

    assert_eq!(
        combined, commit_weighted,
        "combine must match commitment of weighted sum"
    );
}

#[test]
fn deterministic_commitment() {
    let num_vars = 4;
    let prover_setup = DoryScheme::setup_prover(num_vars);

    let mut rng = ChaCha20Rng::seed_from_u64(700);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);

    let (c1, _) = DoryScheme::commit(poly.evaluations(), &prover_setup);
    let (c2, _) = DoryScheme::commit(poly.evaluations(), &prover_setup);

    assert_eq!(c1, c2, "same poly + setup must yield identical commitment");
}

#[test]
fn zk_commitment_is_blinded() {
    let num_vars = 4;
    let prover_setup = DoryScheme::setup_prover(num_vars);

    let mut rng = ChaCha20Rng::seed_from_u64(750);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);

    let (c1, _) = <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);
    let (c2, _) = <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);

    assert_ne!(c1, c2, "ZK Dory commitments must use fresh blinding");
}

#[test]
fn wrong_commitment_rejected() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(900);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

    let mut pt = Blake2bTranscript::new(b"wrong-commit");
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt);

    // Commit to a different polynomial
    let wrong_poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let (wrong_commitment, _) = DoryScheme::commit(wrong_poly.evaluations(), &prover_setup);
    assert_ne!(commitment, wrong_commitment);

    let mut vt = Blake2bTranscript::new(b"wrong-commit");
    let result = DoryScheme::verify(
        &wrong_commitment,
        &point,
        eval,
        &proof,
        &verifier_setup,
        &mut vt,
    );
    assert!(result.is_err(), "wrong commitment must be rejected");
}

#[test]
fn wrong_transcript_domain_rejected() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(1000);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

    let mut pt = Blake2bTranscript::new(b"correct-domain");
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt);

    let mut vt = Blake2bTranscript::new(b"wrong-domain");
    let result = DoryScheme::verify(&commitment, &point, eval, &proof, &verifier_setup, &mut vt);
    assert!(result.is_err(), "wrong transcript domain must be rejected");
}

#[test]
fn property_based_round_trip() {
    for seed in 0..10u64 {
        let num_vars = 2 + (seed as usize % 4); // 2..5
        round_trip::<Blake2bTranscript>(num_vars, 800 + seed, b"prop-rt");
    }
}

fn zk_round_trip<T: Transcript<Challenge = Fr>>(num_vars: usize, seed: u64, label: &'static [u8]) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);

    let mut pt = T::new(label);
    let (proof, _eval_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut pt);

    let mut vt = T::new(label);
    DoryScheme::verify_zk(&commitment, &point, &proof, &verifier_setup, &mut vt)
        .expect("ZK round-trip verification must succeed");
}

#[test]
fn zk_round_trip_various_sizes() {
    for num_vars in [2, 3, 4, 6] {
        zk_round_trip::<Blake2bTranscript>(num_vars, 1100 + num_vars as u64, b"zk-cov-sizes");
    }
}

#[test]
fn zk_round_trip_both_transcripts() {
    let num_vars = 4;
    zk_round_trip::<Blake2bTranscript>(num_vars, 1200, b"zk-blake2b-rt");
    zk_round_trip::<KeccakTranscript>(num_vars, 1200, b"zk-keccak-rt");
}

#[test]
fn zk_wrong_commitment_rejected() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(1400);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);

    let mut pt = Blake2bTranscript::new(b"zk-wrong-commit");
    let (proof, _eval_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut pt);

    let wrong_poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let (wrong_commitment, _) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(wrong_poly.evaluations(), &prover_setup);
    assert_ne!(commitment, wrong_commitment);

    let mut vt = Blake2bTranscript::new(b"zk-wrong-commit");
    let result = DoryScheme::verify_zk(&wrong_commitment, &point, &proof, &verifier_setup, &mut vt);
    assert!(result.is_err(), "ZK: wrong commitment must be rejected");
}

#[test]
fn transparent_commitment_rejected_for_zk_blinded_proof() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(1450);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (transparent_commitment, _) = DoryScheme::commit(poly.evaluations(), &prover_setup);
    let (_zk_commitment, hint) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);

    let mut pt = Blake2bTranscript::new(b"zk-transparent-reject");
    let (proof, _eval_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut pt);

    let mut vt = Blake2bTranscript::new(b"zk-transparent-reject");
    let result = DoryScheme::verify_zk(
        &transparent_commitment,
        &point,
        &proof,
        &verifier_setup,
        &mut vt,
    );
    assert!(
        result.is_err(),
        "transparent commitment must not verify against a proof using a ZK commit blind"
    );
}

#[test]
fn zk_combined_commitment_and_hint_verify() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(1475);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);
    let (commit_a, hint_a) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly_a.evaluations(), &prover_setup);
    let (commit_b, hint_b) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly_b.evaluations(), &prover_setup);

    let c1 = <Fr as RandomSampling>::random(&mut rng);
    let c2 = <Fr as RandomSampling>::random(&mut rng);
    let combined_commitment = DoryScheme::combine(&[commit_a, commit_b], &[c1, c2]);
    let combined_hint = DoryScheme::combine_hints(vec![hint_a, hint_b], &[c1, c2]);

    let weighted_evals: Vec<Fr> = poly_a
        .evaluations()
        .iter()
        .zip(poly_b.evaluations().iter())
        .map(|(a, b)| c1 * *a + c2 * *b)
        .collect();
    let weighted_poly = Polynomial::new(weighted_evals);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = weighted_poly.evaluate(&point);

    let mut pt = Blake2bTranscript::new(b"zk-combined");
    let (proof, _eval_com, _blind) = DoryScheme::open_zk(
        &weighted_poly,
        &point,
        eval,
        &prover_setup,
        combined_hint,
        &mut pt,
    );

    let mut vt = Blake2bTranscript::new(b"zk-combined");
    DoryScheme::verify_zk(
        &combined_commitment,
        &point,
        &proof,
        &verifier_setup,
        &mut vt,
    )
    .expect("combined ZK commitment and hint must verify");
}

#[test]
fn wrong_eval_commitment_rejected_zk() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(1600);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);

    let mut pt = Blake2bTranscript::new(b"zk-tampered-y-com");
    let (mut proof, _eval_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut pt);

    // Replace proof.y_com (the hiding commitment to the evaluation) with a
    // different valid G1. dory::verify must reject because the Σ₁/Σ₂ sub-proofs
    // bind y_com cryptographically to the rest of the proof.
    proof.0.y_com = Some(ArkG1::default());

    let mut vt = Blake2bTranscript::new(b"zk-tampered-y-com");
    let result = DoryScheme::verify_zk(&commitment, &point, &proof, &verifier_setup, &mut vt);
    assert!(result.is_err(), "tampered proof.y_com must be rejected");
}

#[test]
fn zk_wrong_transcript_domain_rejected() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(1500);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);

    let mut pt = Blake2bTranscript::new(b"zk-correct-domain");
    let (proof, _eval_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut pt);

    let mut vt = Blake2bTranscript::new(b"zk-wrong-domain");
    let result = DoryScheme::verify_zk(&commitment, &point, &proof, &verifier_setup, &mut vt);
    assert!(
        result.is_err(),
        "ZK: wrong transcript domain must be rejected"
    );
}
