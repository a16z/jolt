use crate::{
    OracleDescriptor, OracleRef, PolynomialBatchChunk, PolynomialBatchStream, PolynomialStream,
    WitnessError, WitnessNamespace,
};

pub trait WitnessProvider<F, N: WitnessNamespace> {
    fn namespace(&self) -> crate::NamespaceId {
        N::ID
    }

    fn describe_oracle(&self, oracle: OracleRef<N>) -> Result<OracleDescriptor<N>, WitnessError>;

    /// Materializes the oracle's dense field-element evaluations, row-major
    /// over the domain declared by [`describe_oracle`](Self::describe_oracle).
    fn oracle_table(&self, oracle: OracleRef<N>) -> Result<Vec<F>, WitnessError>;

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
