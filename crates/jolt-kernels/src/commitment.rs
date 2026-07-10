//! The witness-commitment kernel: streaming PCS commitment of committed
//! witness polynomials over the proof's shared embedding grid.
//!
//! Every witness polynomial is committed as a matrix in one common grid shape
//! (`2^⌈total_vars/2⌉` columns), not per-polynomial squares: the stage-8 joint
//! opening combines the commitments homomorphically, which is only meaningful
//! when they share row geometry. Dense polynomials stream row-by-row through
//! [`StreamingCommitment::feed`]-family calls; one-hot polynomials stream
//! per-cycle hot indices through the column-major one-hot path (a `(K × T)`
//! matrix whose rows interleave address and cycle-chunk exactly as the legacy
//! prover's tiered commit did).

use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_field::Field;
use jolt_openings::{CommitmentScheme, StreamingCommitment};
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::{OracleRef, PolynomialChunk, PolynomialEncoding, WitnessProvider};

use crate::{KernelError, ProofSession, ReferenceBackend};

/// The shared embedding grid every witness polynomial is committed in:
/// `2^⌈total_vars/2⌉` columns, where `total_vars` is the maximum over the
/// one-hot main matrix (`log_k_chunk + log_t`) and any precommitted-candidate
/// shapes (advice, committed program).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentGrid {
    pub total_vars: usize,
    pub log_t: usize,
}

impl CommitmentGrid {
    pub const fn num_columns(&self) -> usize {
        1 << self.total_vars.div_ceil(2)
    }
}

/// One committed witness polynomial: its id, commitment, and the opening hint
/// the stage-8 joint opening consumes.
pub struct WitnessCommitment<PCS: CommitmentScheme> {
    pub id: JoltCommittedPolynomial,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
}

/// The witness-commitment slot: commit every polynomial in `ids` out of the
/// witness oracle over the shared embedding grid. Results are returned in
/// `ids` order; execution order, batching, and streaming strategy are the
/// implementation's business (the trait deliberately does not require
/// [`StreamingCommitment`] — that is the reference implementation's
/// strategy). Transcript-free: the caller absorbs the returned commitments.
pub trait CommitWitness<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    fn commit_witness(
        &self,
        session: &mut ProofSession,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
        ids: &[JoltCommittedPolynomial],
        grid: CommitmentGrid,
        setup: &PCS::ProverSetup,
    ) -> Result<Vec<WitnessCommitment<PCS>>, KernelError<F>>;
}

impl<F, PCS> CommitWitness<F, PCS> for ReferenceBackend
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment,
{
    fn commit_witness(
        &self,
        _session: &mut ProofSession,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
        ids: &[JoltCommittedPolynomial],
        grid: CommitmentGrid,
        setup: &PCS::ProverSetup,
    ) -> Result<Vec<WitnessCommitment<PCS>>, KernelError<F>> {
        commit_witness::<F, PCS>(witness, ids, grid, setup)
    }
}

/// Commit each witness polynomial in `ids` (in order) by streaming it out of
/// the witness oracle and into the PCS over `grid` — the reference
/// implementation behind [`CommitWitness`].
pub fn commit_witness<F, PCS>(
    witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ids: &[JoltCommittedPolynomial],
    grid: CommitmentGrid,
    setup: &PCS::ProverSetup,
) -> Result<Vec<WitnessCommitment<PCS>>, KernelError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment,
{
    ids.iter()
        .map(|&id| {
            let (commitment, hint) = commit_one::<F, PCS>(witness, id, grid, setup)?;
            Ok(WitnessCommitment {
                id,
                commitment,
                hint,
            })
        })
        .collect()
}

fn commit_one<F, PCS>(
    witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    id: JoltCommittedPolynomial,
    grid: CommitmentGrid,
    setup: &PCS::ProverSetup,
) -> Result<(PCS::Output, PCS::OpeningHint), KernelError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment,
{
    let descriptor = witness.describe_oracle(OracleRef::<JoltVmNamespace>::Committed(id))?;
    let row_width = grid.num_columns();
    let mut stream = witness.committed_stream(id, row_width)?;

    match descriptor.encoding {
        PolynomialEncoding::OneHot => {
            // The one-hot `(K × T)` matrix: `K = 2^(log_rows − log_t)` and the
            // stream yields `row_width` cycles of hot addresses per chunk.
            let one_hot_k = 1usize
                .checked_shl(
                    descriptor
                        .dimensions
                        .log_rows
                        .checked_sub(grid.log_t)
                        .ok_or_else(|| KernelError::InvalidGeometry {
                            reason: format!(
                                "one-hot polynomial {id:?} has fewer variables than log_t"
                            ),
                        })? as u32,
                )
                .ok_or_else(|| KernelError::InvalidGeometry {
                    reason: format!("one-hot K overflow for {id:?}"),
                })?;
            let mut context = PCS::begin_one_hot_column_major_stream(setup, row_width);
            let mut chunk_commitments = Vec::new();
            while let Some(chunk) = stream.next_chunk()? {
                let PolynomialChunk::OneHot(indices) = chunk else {
                    return Err(KernelError::UnsupportedChunk {
                        reason: format!("one-hot polynomial {id:?} streamed a non-one-hot chunk"),
                    });
                };
                chunk_commitments.push(PCS::process_one_hot_chunk(
                    &mut context,
                    setup,
                    one_hot_k,
                    &indices,
                ));
            }
            Ok(PCS::finish_one_hot_column_major_chunks(
                setup,
                one_hot_k,
                &chunk_commitments,
            ))
        }
        PolynomialEncoding::Dense | PolynomialEncoding::Compact => {
            let mut partial = PCS::begin(setup);
            while let Some(chunk) = stream.next_chunk()? {
                feed_dense_chunk::<F, PCS>(&mut partial, chunk, row_width, id, setup)?;
            }
            Ok(PCS::finish_with_hint(partial, setup))
        }
    }
}

fn feed_dense_chunk<F, PCS>(
    partial: &mut PCS::PartialCommitment,
    chunk: PolynomialChunk<F>,
    row_width: usize,
    id: JoltCommittedPolynomial,
    setup: &PCS::ProverSetup,
) -> Result<(), KernelError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment,
{
    match chunk {
        PolynomialChunk::Dense(values) => PCS::feed(partial, &values, setup),
        PolynomialChunk::U64(values) => PCS::feed_u64(partial, &values, setup),
        PolynomialChunk::I128(values) => PCS::feed_i128(partial, &values, setup),
        PolynomialChunk::U8(values) => {
            let values: Vec<u64> = values.into_iter().map(u64::from).collect();
            PCS::feed_u64(partial, &values, setup);
        }
        PolynomialChunk::U16(values) => {
            let values: Vec<u64> = values.into_iter().map(u64::from).collect();
            PCS::feed_u64(partial, &values, setup);
        }
        PolynomialChunk::U32(values) => {
            let values: Vec<u64> = values.into_iter().map(u64::from).collect();
            PCS::feed_u64(partial, &values, setup);
        }
        PolynomialChunk::I64(values) => {
            let values: Vec<i128> = values.into_iter().map(i128::from).collect();
            PCS::feed_i128(partial, &values, setup);
        }
        PolynomialChunk::Zeros(len) => {
            if len % row_width != 0 {
                return Err(KernelError::UnsupportedChunk {
                    reason: format!(
                        "zero run of {len} for {id:?} is not a whole number of {row_width}-wide rows"
                    ),
                });
            }
            PCS::feed_zeros(partial, row_width, len / row_width, setup);
        }
        PolynomialChunk::OneHot(_) => {
            return Err(KernelError::UnsupportedChunk {
                reason: format!("dense polynomial {id:?} streamed a one-hot chunk"),
            });
        }
    }
    Ok(())
}
