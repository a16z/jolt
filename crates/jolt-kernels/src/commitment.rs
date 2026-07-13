//! The witness-commitment kernel: streaming PCS commitment of committed
//! witness polynomials over the proof's shared embedding grid.
//!
//! Every witness polynomial is committed as a matrix in one common grid shape
//! (`2^⌈total_vars/2⌉` columns), not per-polynomial squares: the stage-8 joint
//! opening combines the commitments homomorphically, which is only meaningful
//! when they share row geometry. Under the cycle-major order, dense
//! polynomials stream row-by-row through [`StreamingCommitment::feed`]-family
//! calls and one-hot polynomials stream per-cycle hot indices through the
//! column-major one-hot path (a `(K × T)` matrix whose rows interleave
//! address and cycle-chunk exactly as the legacy prover's tiered commit did).
//!
//! Under the address-major order every polynomial's coefficients scatter
//! cycle-block-strided across the whole grid (`index = t · cycle_stride +
//! k · one_hot_stride`, dense polynomials at address slot zero), so no
//! cycle-contiguous stream exists; the reference implementation materializes
//! the full grid table and feeds dense rows — the same per-row MSMs legacy's
//! materialized address-major commit runs, full matrix height included (its
//! trailing identity rows are part of the wire hint).

use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, TracePolynomialOrder};
use jolt_field::Field;
use jolt_openings::{CommitmentScheme, StreamingCommitment};
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::{OracleRef, PolynomialChunk, PolynomialEncoding, WitnessProvider};

use crate::{KernelError, ProofSession, ReferenceBackend};

/// The shared embedding grid every witness polynomial is committed in:
/// `2^⌈total_vars/2⌉` columns, where `total_vars` is the maximum over the
/// one-hot main matrix (`log_k_chunk + log_t`) and any precommitted-candidate
/// shapes (advice, committed program). `order` is the proof's
/// coefficient-placement mode; dedicated advice grids are always
/// [`TracePolynomialOrder::CycleMajor`] (their placement is contiguous in
/// both proof layouts — legacy's strides collapse outside the main context)
/// with `log_k_chunk` 0 (no one-hot polynomials).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentGrid {
    pub total_vars: usize,
    pub log_t: usize,
    /// The committed one-hot address width — the main matrix contributes
    /// `log_k_chunk + log_t` of `total_vars`; the rest is the
    /// precommitted-candidate embedding extra.
    pub log_k_chunk: usize,
    pub order: TracePolynomialOrder,
}

impl CommitmentGrid {
    pub const fn num_columns(&self) -> usize {
        1 << self.total_vars.div_ceil(2)
    }

    /// The address-major per-cycle block width: cycle `t`'s coefficients
    /// occupy grid indices `[t · cycle_stride, (t+1) · cycle_stride)` —
    /// legacy's `dense_stride = 2^(e + log_k_chunk)`.
    pub const fn cycle_stride(&self) -> usize {
        debug_assert!(self.total_vars >= self.log_t);
        1 << (self.total_vars - self.log_t)
    }

    /// The address-major within-block address stride — legacy's
    /// `one_hot_stride = 2^e`, where `e` is the embedding extra a
    /// precommitted candidate wider than the main matrix leaves between
    /// addresses (`1` on unwidened grids).
    pub const fn one_hot_stride(&self) -> usize {
        debug_assert!(self.total_vars >= self.log_t + self.log_k_chunk);
        1 << (self.total_vars - self.log_t - self.log_k_chunk)
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

    if grid.order == TracePolynomialOrder::AddressMajor {
        // No cycle-contiguous stream exists in this order (see the module
        // doc): materialize the strided grid table and feed dense rows.
        let table = materialize_address_major::<F>(&mut stream, descriptor.encoding, grid, id)?;
        let mut partial = PCS::begin(setup);
        for row in table.chunks(row_width) {
            PCS::feed(&mut partial, row, setup);
        }
        return Ok(PCS::finish_with_hint(partial, setup));
    }

    match descriptor.encoding {
        PolynomialEncoding::OneHot if row_width > (1usize << grid.log_t) => {
            // A precommitted candidate widened the grid columns past the
            // trace length, so one committed row packs multiple `k`-blocks of
            // the flat `(K × T)` matrix — inexpressible through the
            // column-major one-hot stream. Materialize the flat table and
            // feed dense rows (the same MSM legacy's materialized path runs).
            let flat = materialize_one_hot_flat::<F>(
                &mut stream,
                descriptor.dimensions.rows(),
                1usize << grid.log_t,
            )?;
            let mut partial = PCS::begin(setup);
            for row in flat.chunks(row_width) {
                PCS::feed(&mut partial, row, setup);
            }
            Ok(PCS::finish_with_hint(partial, setup))
        }
        PolynomialEncoding::OneHot => {
            // The one-hot `(K × T)` matrix: `K = 2^(log_rows − log_t)` and the
            // stream yields `row_width` cycles of hot addresses per chunk.
            let log_k = descriptor
                .dimensions
                .log_rows
                .checked_sub(grid.log_t)
                .ok_or_else(|| KernelError::InvalidGeometry {
                    reason: format!("one-hot polynomial {id:?} has fewer variables than log_t"),
                })?;
            let one_hot_k =
                1usize
                    .checked_shl(log_k as u32)
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

/// Drain a committed stream into the full `2^total_vars` address-major grid
/// table: cycle `t`'s coefficients occupy the block at `t · cycle_stride` — a
/// one-hot's hot address lands at `t · cycle_stride + k · one_hot_stride`, a
/// dense coefficient at the block's address slot zero (legacy's
/// `scaled_index = cycle · dense_stride + k · one_hot_stride`). Eager-dense
/// like the stage-8 joint-opening slot: a reference-tier trade-off.
fn materialize_address_major<F: Field>(
    stream: &mut Box<dyn jolt_witness::PolynomialStream<F> + '_>,
    encoding: PolynomialEncoding,
    grid: CommitmentGrid,
    id: JoltCommittedPolynomial,
) -> Result<Vec<F>, KernelError<F>> {
    let cycle_stride = grid.cycle_stride();
    let one_hot_stride = grid.one_hot_stride();
    let one_hot_k = 1usize << grid.log_k_chunk;
    let cycles = 1usize << grid.log_t;
    let mut table = vec![F::zero(); 1usize << grid.total_vars];
    let mut cycle = 0usize;
    let overrun = || KernelError::UnsupportedChunk {
        reason: format!("stream for {id:?} ran past the trace length"),
    };
    while let Some(chunk) = stream.next_chunk()? {
        match chunk {
            PolynomialChunk::OneHot(indices) => {
                if encoding != PolynomialEncoding::OneHot {
                    return Err(KernelError::UnsupportedChunk {
                        reason: format!("dense polynomial {id:?} streamed a one-hot chunk"),
                    });
                }
                if cycle + indices.len() > cycles {
                    return Err(overrun());
                }
                for hot in indices {
                    if let Some(k) = hot {
                        if k >= one_hot_k {
                            return Err(KernelError::UnsupportedChunk {
                                reason: format!(
                                    "one-hot address for {id:?} outside its cycle block"
                                ),
                            });
                        }
                        table[cycle * cycle_stride + k * one_hot_stride] = F::one();
                    }
                    cycle += 1;
                }
            }
            // A zero run's unit is COEFFICIENTS (the dense consumers' reading);
            // one cycle = one dense coefficient here, but a one-hot cycle is a
            // K-wide block — reject like every cycle-major one-hot path.
            PolynomialChunk::Zeros(len) => {
                if encoding == PolynomialEncoding::OneHot {
                    return Err(KernelError::UnsupportedChunk {
                        reason: format!("one-hot polynomial {id:?} streamed a non-one-hot chunk"),
                    });
                }
                if cycle + len > cycles {
                    return Err(overrun());
                }
                cycle += len;
            }
            chunk => {
                if encoding == PolynomialEncoding::OneHot {
                    return Err(KernelError::UnsupportedChunk {
                        reason: format!("one-hot polynomial {id:?} streamed a non-one-hot chunk"),
                    });
                }
                let values = dense_chunk_values(chunk, id)?;
                if cycle + values.len() > cycles {
                    return Err(overrun());
                }
                for value in values {
                    table[cycle * cycle_stride] = value;
                    cycle += 1;
                }
            }
        }
    }
    Ok(table)
}

/// Promote a dense-typed chunk's coefficients to field elements. One-hot and
/// zero-run chunks are the caller's business (its match arms consume them
/// before reaching this).
fn dense_chunk_values<F: Field>(
    chunk: PolynomialChunk<F>,
    id: JoltCommittedPolynomial,
) -> Result<Vec<F>, KernelError<F>> {
    Ok(match chunk {
        PolynomialChunk::Dense(values) => values,
        PolynomialChunk::U64(values) => values.into_iter().map(F::from_u64).collect(),
        PolynomialChunk::U8(values) => values.into_iter().map(F::from_u8).collect(),
        PolynomialChunk::U16(values) => values.into_iter().map(F::from_u16).collect(),
        PolynomialChunk::U32(values) => values.into_iter().map(F::from_u32).collect(),
        PolynomialChunk::I64(values) => values.into_iter().map(F::from_i64).collect(),
        PolynomialChunk::I128(values) => values.into_iter().map(F::from_i128).collect(),
        PolynomialChunk::Zeros(_) | PolynomialChunk::OneHot(_) => {
            return Err(KernelError::UnsupportedChunk {
                reason: format!("non-dense chunk for {id:?} reached the dense promotion"),
            });
        }
    })
}

/// Drain a one-hot stream (per-cycle hot addresses) into the flat `(K × T)`
/// 0/1 table (`flat[k · T + cycle]`), `total = K · T` entries long.
fn materialize_one_hot_flat<F: Field>(
    stream: &mut Box<dyn jolt_witness::PolynomialStream<F> + '_>,
    total: usize,
    cycles: usize,
) -> Result<Vec<F>, KernelError<F>> {
    let mut flat = vec![F::zero(); total];
    let mut cycle = 0usize;
    while let Some(chunk) = stream.next_chunk()? {
        let PolynomialChunk::OneHot(indices) = chunk else {
            return Err(KernelError::UnsupportedChunk {
                reason: "one-hot polynomial streamed a non-one-hot chunk".to_owned(),
            });
        };
        for hot in indices {
            if cycle >= cycles {
                return Err(KernelError::UnsupportedChunk {
                    reason: "one-hot stream ran past the trace length".to_owned(),
                });
            }
            if let Some(k) = hot {
                let index = k * cycles + cycle;
                if index >= total {
                    return Err(KernelError::UnsupportedChunk {
                        reason: "one-hot address outside the (K × T) grid".to_owned(),
                    });
                }
                flat[index] = F::one();
            }
            cycle += 1;
        }
    }
    Ok(flat)
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
