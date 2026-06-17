use crate::{
    OracleDescriptor, OracleRef, OracleViewRequest, PolynomialBatchChunk, PolynomialBatchStream,
    PolynomialStream, PolynomialView, ViewRequirement, WitnessError, WitnessNamespace,
};

pub const RA_FAMILY_MAX_INSTRUCTION_CHUNKS: usize = 32;
pub const RA_FAMILY_MAX_BYTECODE_CHUNKS: usize = 6;
pub const RA_FAMILY_MAX_RAM_CHUNKS: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RaFamilyCycleIndices {
    pub instruction: [u8; RA_FAMILY_MAX_INSTRUCTION_CHUNKS],
    pub bytecode: [u8; RA_FAMILY_MAX_BYTECODE_CHUNKS],
    pub ram: [Option<u8>; RA_FAMILY_MAX_RAM_CHUNKS],
}

pub trait WitnessProvider<F, N: WitnessNamespace> {
    fn namespace(&self) -> crate::NamespaceId {
        N::ID
    }

    fn describe_oracle(&self, oracle: OracleRef<N>) -> Result<OracleDescriptor<N>, WitnessError>;

    fn view_requirements(
        &self,
        oracle: OracleRef<N>,
    ) -> Result<Vec<ViewRequirement<N>>, WitnessError>;

    fn oracle_view(
        &self,
        request: OracleViewRequest<N>,
    ) -> Result<PolynomialView<'_, F, N>, WitnessError>;

    /// Optionally evaluates a requested oracle view without materializing it.
    ///
    /// Providers should return `Ok(None)` when they do not have a direct path
    /// for the requested view, leaving callers to fall back to `oracle_view`.
    fn try_evaluate_oracle_view(
        &self,
        request: OracleViewRequest<N>,
        point: &[F],
    ) -> Result<Option<F>, WitnessError> {
        let _ = request;
        let _ = point;
        Ok(None)
    }

    fn committed_stream<'a>(
        &'a self,
        _id: N::CommittedId,
        _chunk_size: usize,
    ) -> Result<Box<dyn PolynomialStream<F> + 'a>, WitnessError>
    where
        F: 'a,
        N: 'a,
    {
        Err(WitnessError::UnsupportedView {
            view: "committed polynomial streaming",
        })
    }

    fn committed_batch_stream<'a>(
        &'a self,
        ids: &'a [N::CommittedId],
        chunk_size: usize,
    ) -> Result<Box<dyn PolynomialBatchStream<F, N> + 'a>, WitnessError>
    where
        F: 'a,
        N: 'a,
        N::CommittedId: 'a,
    {
        let streams = ids
            .iter()
            .copied()
            .map(|id| self.committed_stream(id, chunk_size))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Box::new(FallbackCommittedBatchStream {
            ids: ids.to_vec(),
            streams,
        }))
    }

    fn try_collect_ra_family_cycle_indices(
        &self,
        _instruction_ids: &[N::CommittedId],
        _bytecode_ids: &[N::CommittedId],
        _ram_ids: &[N::CommittedId],
        _log_k_chunk: usize,
        _log_t: usize,
    ) -> Result<Option<Vec<RaFamilyCycleIndices>>, WitnessError> {
        Ok(None)
    }
}

pub trait CommittedWitnessProvider<F, N: WitnessNamespace>: WitnessProvider<F, N> {
    fn committed_oracle_order(&self) -> Result<Vec<N::CommittedId>, WitnessError>;
}

struct FallbackCommittedBatchStream<'a, F, N: WitnessNamespace> {
    ids: Vec<N::CommittedId>,
    streams: Vec<Box<dyn PolynomialStream<F> + 'a>>,
}

impl<F, N> PolynomialBatchStream<F, N> for FallbackCommittedBatchStream<'_, F, N>
where
    N: WitnessNamespace,
{
    fn next_batch(&mut self) -> Result<Option<PolynomialBatchChunk<N, F>>, WitnessError> {
        let mut chunks = Vec::with_capacity(self.streams.len());
        let mut saw_chunk = false;
        let mut saw_end = false;

        for (id, stream) in self.ids.iter().copied().zip(&mut self.streams) {
            match stream.next_chunk()? {
                Some(chunk) => {
                    saw_chunk = true;
                    chunks.push((id, chunk));
                }
                None => saw_end = true,
            }
        }

        match (saw_chunk, saw_end) {
            (false, true | false) => Ok(None),
            (true, false) => Ok(Some(PolynomialBatchChunk::new(chunks))),
            (true, true) => Err(WitnessError::InvalidWitnessData {
                namespace: N::ID.name,
                reason: "committed batch streams ended at different chunk boundaries".to_owned(),
            }),
        }
    }
}
