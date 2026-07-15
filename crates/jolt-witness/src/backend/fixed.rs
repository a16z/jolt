//! Stored-column backend: the second implementor of the oracle seam —
//! kernel unit tests and slot-fixture replay run against it without a trace.

use std::collections::HashMap;

use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
use jolt_field::Field;

use crate::{
    JoltWitnessOracle, PolynomialBatchChunk, PolynomialBatchStream, PolynomialChunk,
    PolynomialStream, Shape, WitnessError,
};

pub(crate) const FIXED_LABEL: &str = "fixed";

/// Serves explicitly stored dense columns by id; every id not inserted is
/// unknown.
#[derive(Clone, Debug, Default)]
pub struct FixedBackend<F> {
    columns: HashMap<JoltPolynomialId, (Shape, Vec<F>)>,
    committed_order: Vec<JoltCommittedPolynomial>,
}

impl<F> FixedBackend<F> {
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
            committed_order: Vec::new(),
        }
    }

    pub fn insert(
        &mut self,
        id: JoltPolynomialId,
        shape: Shape,
        values: Vec<F>,
    ) -> Result<(), WitnessError> {
        if values.len() != shape.rows() {
            return Err(WitnessError::InvalidDimensions {
                label: FIXED_LABEL,
                reason: format!(
                    "column {id:?} has {} values, shape declares {}",
                    values.len(),
                    shape.rows()
                ),
            });
        }
        let _ = self.columns.insert(id, (shape, values));
        Ok(())
    }

    /// The proof-payload order reported by [`JoltWitnessOracle::committed_order`].
    pub fn set_committed_order(&mut self, order: Vec<JoltCommittedPolynomial>) {
        self.committed_order = order;
    }

    fn column(&self, id: JoltPolynomialId) -> Result<&(Shape, Vec<F>), WitnessError> {
        self.columns
            .get(&id)
            .ok_or(WitnessError::UnknownOracle { label: FIXED_LABEL })
    }
}

impl<F: Field> JoltWitnessOracle<F> for FixedBackend<F> {
    fn shape(&self, id: JoltPolynomialId) -> Result<Shape, WitnessError> {
        self.column(id).map(|(shape, _)| *shape)
    }

    fn oracle_table(&self, id: JoltPolynomialId) -> Result<Vec<F>, WitnessError> {
        self.column(id).map(|(_, values)| values.clone())
    }

    fn committed_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
        Ok(self.committed_order.clone())
    }

    fn committed_stream<'a>(
        &'a self,
        id: JoltCommittedPolynomial,
        chunk_size: usize,
    ) -> Result<Box<dyn PolynomialStream<F> + 'a>, WitnessError>
    where
        F: 'a,
    {
        if chunk_size == 0 {
            return Err(WitnessError::InvalidDimensions {
                label: FIXED_LABEL,
                reason: "stream chunk size must be nonzero".to_owned(),
            });
        }
        let (_, values) = self.column(JoltPolynomialId::Committed(id))?;
        Ok(Box::new(FixedColumnStream {
            values,
            position: 0,
            chunk_size,
        }))
    }

    fn committed_batch_stream<'a>(
        &'a self,
        ids: &'a [JoltCommittedPolynomial],
        chunk_size: usize,
    ) -> Result<Box<dyn PolynomialBatchStream<F, JoltCommittedPolynomial> + 'a>, WitnessError>
    where
        F: 'a,
    {
        let streams = ids
            .iter()
            .map(|&id| {
                let (_, values) = self.column(JoltPolynomialId::Committed(id))?;
                Ok((
                    id,
                    FixedColumnStream {
                        values,
                        position: 0,
                        chunk_size,
                    },
                ))
            })
            .collect::<Result<Vec<_>, WitnessError>>()?;
        Ok(Box::new(FixedColumnBatchStream { streams }))
    }
}

struct FixedColumnStream<'a, F> {
    values: &'a [F],
    position: usize,
    chunk_size: usize,
}

impl<F: Field> PolynomialStream<F> for FixedColumnStream<'_, F> {
    fn next_chunk(&mut self) -> Result<Option<PolynomialChunk<F>>, WitnessError> {
        if self.position >= self.values.len() {
            return Ok(None);
        }
        let end = (self.position + self.chunk_size).min(self.values.len());
        let chunk = self.values[self.position..end].to_vec();
        self.position = end;
        Ok(Some(PolynomialChunk::Dense(chunk)))
    }
}

struct FixedColumnBatchStream<'a, F> {
    streams: Vec<(JoltCommittedPolynomial, FixedColumnStream<'a, F>)>,
}

impl<F: Field> PolynomialBatchStream<F, JoltCommittedPolynomial> for FixedColumnBatchStream<'_, F> {
    fn next_batch(
        &mut self,
    ) -> Result<Option<PolynomialBatchChunk<JoltCommittedPolynomial, F>>, WitnessError> {
        let mut chunks = Vec::with_capacity(self.streams.len());
        for (id, stream) in &mut self.streams {
            match stream.next_chunk()? {
                Some(chunk) => chunks.push((*id, chunk)),
                None => return Ok(None),
            }
        }
        if chunks.is_empty() {
            return Ok(None);
        }
        Ok(Some(PolynomialBatchChunk::new(chunks)))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::JoltVirtualPolynomial;
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::*;
    use crate::PolynomialEncoding;

    #[test]
    fn serves_inserted_columns_and_rejects_unknown_ids() {
        let mut backend = FixedBackend::new();
        let id = JoltPolynomialId::Virtual(JoltVirtualPolynomial::LookupOutput);
        let values: Vec<Fr> = (0..4).map(Fr::from_u64).collect();
        backend
            .insert(id, Shape::new(2, PolynomialEncoding::Dense), values.clone())
            .unwrap();

        assert_eq!(
            JoltWitnessOracle::<Fr>::shape(&backend, id).unwrap(),
            Shape::new(2, PolynomialEncoding::Dense)
        );
        assert_eq!(
            JoltWitnessOracle::<Fr>::oracle_table(&backend, id).unwrap(),
            values
        );
        assert_eq!(
            JoltWitnessOracle::<Fr>::shape(
                &backend,
                JoltPolynomialId::Virtual(JoltVirtualPolynomial::PC)
            ),
            Err(WitnessError::UnknownOracle { label: FIXED_LABEL })
        );
    }

    #[test]
    fn insert_rejects_shape_mismatch() {
        let mut backend = FixedBackend::<Fr>::new();
        let id = JoltPolynomialId::Virtual(JoltVirtualPolynomial::LookupOutput);
        assert!(matches!(
            backend.insert(
                id,
                Shape::new(2, PolynomialEncoding::Dense),
                vec![Fr::from_u64(1)]
            ),
            Err(WitnessError::InvalidDimensions { .. })
        ));
    }
}
