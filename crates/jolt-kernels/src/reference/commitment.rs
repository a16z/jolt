//! The reference witness-commitment kernel: streaming.
//!
//! Under the cycle-major order, dense polynomials stream row-by-row through
//! [`StreamingCommitment::feed`]-family calls and one-hot polynomials stream
//! per-cycle hot indices through the column-major one-hot path (a `(K × T)`
//! matrix whose rows interleave address and cycle-chunk exactly as the
//! legacy prover's tiered commit did).
//!
//! Under the address-major order every polynomial's coefficients scatter
//! cycle-block-strided across the whole grid (`index = t · cycle_stride +
//! k · one_hot_stride`, dense polynomials at address slot zero), so no
//! cycle-contiguous stream exists; this implementation materializes the
//! full grid table and feeds dense rows — the same per-row MSMs legacy's
//! materialized address-major commit runs, full matrix height included (its
//! trailing identity rows are part of the wire hint).

use jolt_claims::protocols::jolt::{
    JoltCommittedPolynomial, JoltPolynomialId, TracePolynomialOrder,
};
use jolt_field::Field;
use jolt_openings::{CommitmentScheme, StreamingCommitment};
use jolt_witness::{CommittedChunk, JoltWitnessOracle, PolynomialEncoding, WitnessError};

use crate::commitment::{CommitWitness, CommitmentGrid, WitnessCommitment};
use crate::{KernelError, ProofSession, ReferenceBackend};

impl<F, PCS> CommitWitness<F, PCS> for ReferenceBackend
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment,
{
    fn commit_witness(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessOracle<F>,
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

/// Walks one committed column, letting the per-chunk work fail with a kernel
/// error: the kernel error is parked across the walk's `WitnessError` abort
/// channel and rethrown.
fn walk_column<F: Field>(
    witness: &dyn JoltWitnessOracle<F>,
    id: JoltCommittedPolynomial,
    chunk_size: usize,
    mut visit: impl FnMut(CommittedChunk<'_, F>) -> Result<(), KernelError<F>>,
) -> Result<(), KernelError<F>> {
    let mut kernel_error = None;
    let result = witness.visit_committed_column(id, chunk_size, &mut |chunk| {
        visit(chunk).map_err(|error| {
            kernel_error = Some(error);
            WitnessError::UnsupportedView {
                view: "commitment kernel aborted the column walk",
            }
        })
    });
    if let Some(error) = kernel_error {
        return Err(error);
    }
    result.map_err(KernelError::from)
}

fn commit_one<F, PCS>(
    witness: &dyn JoltWitnessOracle<F>,
    id: JoltCommittedPolynomial,
    grid: CommitmentGrid,
    setup: &PCS::ProverSetup,
) -> Result<(PCS::Output, PCS::OpeningHint), KernelError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment,
{
    let shape = witness.shape(JoltPolynomialId::Committed(id))?;
    let row_width = grid.num_columns();

    if grid.order == TracePolynomialOrder::AddressMajor {
        // No cycle-contiguous stream exists in this order (see the module
        // doc): materialize the strided grid table and feed dense rows.
        let table = materialize_address_major::<F>(witness, id, shape.encoding, grid, row_width)?;
        let mut partial = PCS::begin(setup);
        for row in table.chunks(row_width) {
            PCS::feed(&mut partial, row, setup);
        }
        return Ok(PCS::finish_with_hint(partial, setup));
    }

    match shape.encoding {
        PolynomialEncoding::OneHot if row_width > (1usize << grid.log_t) => {
            // A precommitted candidate widened the grid columns past the
            // trace length, so one committed row packs multiple `k`-blocks of
            // the flat `(K × T)` matrix — inexpressible through the
            // column-major one-hot stream. Materialize the flat table and
            // feed dense rows (the same MSM legacy's materialized path runs).
            let flat = materialize_one_hot_flat::<F>(
                witness,
                id,
                shape.rows(),
                1usize << grid.log_t,
                row_width,
            )?;
            let mut partial = PCS::begin(setup);
            for row in flat.chunks(row_width) {
                PCS::feed(&mut partial, row, setup);
            }
            Ok(PCS::finish_with_hint(partial, setup))
        }
        PolynomialEncoding::OneHot => {
            // The one-hot `(K × T)` matrix: `K = 2^(log_rows − log_t)` and the
            // walk yields `row_width` cycles of hot addresses per chunk.
            let log_k = shape.log_rows.checked_sub(grid.log_t).ok_or_else(|| {
                KernelError::InvalidGeometry {
                    reason: format!("one-hot polynomial {id:?} has fewer variables than log_t"),
                }
            })?;
            let one_hot_k =
                1usize
                    .checked_shl(log_k as u32)
                    .ok_or_else(|| KernelError::InvalidGeometry {
                        reason: format!("one-hot K overflow for {id:?}"),
                    })?;
            let mut context = PCS::begin_one_hot_column_major_stream(setup, row_width);
            let mut chunk_commitments = Vec::new();
            walk_column(witness, id, row_width, |chunk| {
                let CommittedChunk::HotAddresses(indices) = chunk else {
                    return Err(KernelError::UnsupportedChunk {
                        reason: format!("one-hot polynomial {id:?} streamed a non-one-hot chunk"),
                    });
                };
                chunk_commitments.push(PCS::process_one_hot_chunk(
                    &mut context,
                    setup,
                    one_hot_k,
                    indices,
                ));
                Ok(())
            })?;
            Ok(PCS::finish_one_hot_column_major_chunks(
                setup,
                one_hot_k,
                &chunk_commitments,
            ))
        }
        PolynomialEncoding::Dense | PolynomialEncoding::Compact => {
            let mut partial = PCS::begin(setup);
            walk_column(witness, id, row_width, |chunk| {
                feed_dense_chunk::<F, PCS>(&mut partial, chunk, row_width, id, setup)
            })?;
            Ok(PCS::finish_with_hint(partial, setup))
        }
    }
}

/// Walk a committed column into the full `2^total_vars` address-major grid
/// table: cycle `t`'s coefficients occupy the block at `t · cycle_stride` — a
/// one-hot's hot address lands at `t · cycle_stride + k · one_hot_stride`, a
/// dense coefficient at the block's address slot zero (legacy's
/// `scaled_index = cycle · dense_stride + k · one_hot_stride`). Eager-dense
/// like the stage-8 joint-opening slot: a reference-tier trade-off.
fn materialize_address_major<F: Field>(
    witness: &dyn JoltWitnessOracle<F>,
    id: JoltCommittedPolynomial,
    encoding: PolynomialEncoding,
    grid: CommitmentGrid,
    chunk_size: usize,
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
    walk_column(witness, id, chunk_size, |chunk| {
        match chunk {
            CommittedChunk::HotAddresses(indices) => {
                if encoding != PolynomialEncoding::OneHot {
                    return Err(KernelError::UnsupportedChunk {
                        reason: format!("dense polynomial {id:?} streamed a one-hot chunk"),
                    });
                }
                if cycle + indices.len() > cycles {
                    return Err(overrun());
                }
                for &hot in indices {
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
            CommittedChunk::Zeros(len) => {
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
                if cycle + chunk.len() > cycles {
                    return Err(overrun());
                }
                match chunk {
                    CommittedChunk::Dense(values) => {
                        for &value in values {
                            table[cycle * cycle_stride] = value;
                            cycle += 1;
                        }
                    }
                    CommittedChunk::Words(values) => {
                        for &value in values {
                            table[cycle * cycle_stride] = F::from_u64(value);
                            cycle += 1;
                        }
                    }
                    CommittedChunk::Increments(values) => {
                        for &value in values {
                            table[cycle * cycle_stride] = F::from_i128(value);
                            cycle += 1;
                        }
                    }
                    CommittedChunk::Zeros(_) | CommittedChunk::HotAddresses(_) => {
                        unreachable!("consumed by the outer match arms")
                    }
                }
            }
        }
        Ok(())
    })?;
    Ok(table)
}

/// Walk a one-hot column (per-cycle hot addresses) into the flat `(K × T)`
/// 0/1 table (`flat[k · T + cycle]`), `total = K · T` entries long.
fn materialize_one_hot_flat<F: Field>(
    witness: &dyn JoltWitnessOracle<F>,
    id: JoltCommittedPolynomial,
    total: usize,
    cycles: usize,
    chunk_size: usize,
) -> Result<Vec<F>, KernelError<F>> {
    let mut flat = vec![F::zero(); total];
    let mut cycle = 0usize;
    walk_column(witness, id, chunk_size, |chunk| {
        let CommittedChunk::HotAddresses(indices) = chunk else {
            return Err(KernelError::UnsupportedChunk {
                reason: "one-hot polynomial streamed a non-one-hot chunk".to_owned(),
            });
        };
        for &hot in indices {
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
        Ok(())
    })?;
    Ok(flat)
}

fn feed_dense_chunk<F, PCS>(
    partial: &mut PCS::PartialCommitment,
    chunk: CommittedChunk<'_, F>,
    row_width: usize,
    id: JoltCommittedPolynomial,
    setup: &PCS::ProverSetup,
) -> Result<(), KernelError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment,
{
    match chunk {
        CommittedChunk::Dense(values) => PCS::feed(partial, values, setup),
        CommittedChunk::Words(values) => PCS::feed_u64(partial, values, setup),
        CommittedChunk::Increments(values) => PCS::feed_i128(partial, values, setup),
        CommittedChunk::Zeros(len) => {
            if len % row_width != 0 {
                return Err(KernelError::UnsupportedChunk {
                    reason: format!(
                        "zero run of {len} for {id:?} is not a whole number of {row_width}-wide rows"
                    ),
                });
            }
            PCS::feed_zeros(partial, row_width, len / row_width, setup);
        }
        CommittedChunk::HotAddresses(_) => {
            return Err(KernelError::UnsupportedChunk {
                reason: format!("dense polynomial {id:?} streamed a one-hot chunk"),
            });
        }
    }
    Ok(())
}
