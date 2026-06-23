mod stream;

use crate::{
    BackendError, CommitmentBackend, CommitmentMode, CommitmentRequest, CommitmentResult,
    ResolvedWitnessRequirement, TracePolynomialEmbedding,
};
use jolt_claims::protocols::jolt::formulas::dimensions::TracePolynomialOrder;
use jolt_openings::ZkStreamingCommitment;
use jolt_poly::OneHotIndexOrder;
use jolt_witness::{
    MaterializationPolicy, OracleDescriptor, OracleRef, PolynomialEncoding, RetentionHint,
    ViewRequirement, WitnessError, WitnessNamespace, WitnessProvider,
};
use rayon::prelude::*;

use super::CpuBackend;

impl<F, N, PCS> CommitmentBackend<F, N, PCS> for CpuBackend
where
    F: jolt_field::Field,
    N: WitnessNamespace + Send + Sync,
    N::CommittedId: Send + Sync,
    N::VirtualId: Send + Sync,
    PCS: ZkStreamingCommitment<Field = F>,
{
    fn commit<W>(
        &mut self,
        request: &CommitmentRequest<N>,
        witness: &W,
        setup: &PCS::ProverSetup,
    ) -> Result<CommitmentResult<N, PCS>, BackendError>
    where
        W: WitnessProvider<F, N> + Sync + ?Sized,
    {
        let mut resolved = Vec::with_capacity(request.items.len());
        for item in &request.items {
            let descriptor = witness.describe_oracle(item.requirement.oracle)?;
            if descriptor.encoding != item.requirement.encoding {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: N::ID.name,
                    reason: format!(
                        "request asked for {:?} encoding, provider exposes {:?}",
                        item.requirement.encoding, descriptor.encoding,
                    ),
                }
                .into());
            }
            if item.requirement.materialization != MaterializationPolicy::Streaming {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: N::ID.name,
                    reason: "CPU commitment backend currently requires streaming materialization"
                        .to_owned(),
                }
                .into());
            }
            let OracleRef::Committed(_) = item.requirement.oracle else {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: N::ID.name,
                    reason: "commitment requests require committed oracles".to_owned(),
                }
                .into());
            };
            resolved.push(ResolvedCommitmentRequest {
                slot: item.slot,
                requirement: item.requirement,
                mode: item.mode,
                trace_polynomial_order: item.trace_polynomial_order,
                trace_embedding: item.trace_embedding,
                descriptor,
            });
        }

        let core_fast_path_layout = self
            .config
            .preserve_core_fast_path
            .then(|| infer_core_fast_path_layout::<N>(&resolved))
            .flatten();

        let resolved_requirements = resolved
            .iter()
            .map(|item| {
                ResolvedWitnessRequirement::new(item.slot, item.requirement, item.descriptor)
            })
            .collect();

        let mut committed = Vec::with_capacity(resolved.len());
        let mut individual_indices = Vec::new();
        if self.config.preserve_core_fast_path {
            let batch_indices = resolved
                .iter()
                .enumerate()
                .filter_map(|(index, item)| {
                    (item.requirement.retention == RetentionHint::ThroughStage8).then_some(index)
                })
                .collect::<Vec<_>>();
            if batch_indices.len() > 1 {
                match commit_resolved_batch::<F, N, PCS, W>(
                    &resolved,
                    &batch_indices,
                    core_fast_path_layout,
                    witness,
                    setup,
                    self.config.commitment_chunk_size,
                ) {
                    Ok(batch_committed) => committed.extend(batch_committed),
                    Err(BackendError::Witness(WitnessError::UnsupportedView { .. })) => {
                        individual_indices.extend(batch_indices);
                    }
                    Err(error) => return Err(error),
                }
            } else {
                individual_indices.extend(batch_indices);
            }
            individual_indices.extend(resolved.iter().enumerate().filter_map(|(index, item)| {
                (item.requirement.retention != RetentionHint::ThroughStage8).then_some(index)
            }));
        } else {
            individual_indices.extend(0..resolved.len());
        }

        if !individual_indices.is_empty() {
            committed.extend(commit_resolved_individually::<F, N, PCS, W>(
                &resolved,
                &individual_indices,
                core_fast_path_layout,
                witness,
                setup,
                self.config.commitment_chunk_size,
            )?);
        }
        committed.sort_by_key(|(index, _)| *index);

        Ok(CommitmentResult::new(
            resolved_requirements,
            committed
                .iter()
                .map(|(_, item)| item.streamed.clone())
                .collect(),
            committed.into_iter().map(|(_, item)| item.output).collect(),
        ))
    }
}

type IndexedCpuCommitmentResult<N, PCS> = (usize, stream::CpuCommitmentResult<N, PCS>);

fn commit_resolved_batch<F, N, PCS, W>(
    resolved: &[ResolvedCommitmentRequest<N>],
    indices: &[usize],
    core_fast_path_layout: Option<CoreFastPathLayout>,
    witness: &W,
    setup: &PCS::ProverSetup,
    chunk_size: usize,
) -> Result<Vec<IndexedCpuCommitmentResult<N, PCS>>, BackendError>
where
    F: jolt_field::Field,
    N: WitnessNamespace + Send + Sync,
    N::CommittedId: Send + Sync,
    N::VirtualId: Send + Sync,
    PCS: ZkStreamingCommitment<Field = F>,
    W: WitnessProvider<F, N> + Sync + ?Sized,
{
    let chunk_size = core_fast_path_layout.map_or(chunk_size, |layout| layout.row_width);
    let ids = indices
        .iter()
        .map(|&index| committed_id(resolved[index].requirement.oracle))
        .collect::<Vec<_>>();
    let mut stream = witness.committed_batch_stream(&ids, chunk_size)?;
    let items = indices
        .iter()
        .map(|&index| {
            let item = &resolved[index];
            stream::BatchCommitmentPlanItem {
                slot: item.slot,
                oracle: item.requirement.oracle,
                id: committed_id(item.requirement.oracle),
                polynomial_rows: item.descriptor.dimensions.rows(),
                layout: item.commitment_layout(core_fast_path_layout),
                mode: item.mode,
            }
        })
        .collect::<Vec<_>>();
    let results =
        stream::commit_batched_streamed_witness::<F, PCS, N>(&items, stream.as_mut(), setup)?;
    Ok(indices.iter().copied().zip(results).collect())
}

fn commit_resolved_individually<F, N, PCS, W>(
    resolved: &[ResolvedCommitmentRequest<N>],
    indices: &[usize],
    core_fast_path_layout: Option<CoreFastPathLayout>,
    witness: &W,
    setup: &PCS::ProverSetup,
    chunk_size: usize,
) -> Result<Vec<IndexedCpuCommitmentResult<N, PCS>>, BackendError>
where
    F: jolt_field::Field,
    N: WitnessNamespace + Send + Sync,
    N::CommittedId: Send + Sync,
    N::VirtualId: Send + Sync,
    PCS: ZkStreamingCommitment<Field = F>,
    W: WitnessProvider<F, N> + Sync + ?Sized,
{
    let mut committed = Vec::with_capacity(indices.len());
    for batch in indices.chunks(commitment_parallel_batch_size_for_indices(
        resolved, indices,
    )) {
        let batch_committed = batch
            .par_iter()
            .map(|&index| {
                let item = &resolved[index];
                let id = committed_id(item.requirement.oracle);
                let mut stream = witness.committed_stream(id, chunk_size)?;
                let layout = item.commitment_layout(core_fast_path_layout);
                let result = stream::commit_streamed_witness::<F, PCS, N>(
                    item.slot,
                    item.requirement.oracle,
                    item.descriptor.dimensions.rows(),
                    layout,
                    item.mode,
                    stream.as_mut(),
                    setup,
                )?;
                Ok((index, result))
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        committed.extend(batch_committed);
    }
    Ok(committed)
}

fn committed_id<N: WitnessNamespace>(kind: OracleRef<N>) -> N::CommittedId {
    let OracleRef::Committed(id) = kind else {
        unreachable!("committed oracle shape is validated before request resolution")
    };
    id
}

fn commitment_parallel_batch_size_for_indices<N: WitnessNamespace>(
    items: &[ResolvedCommitmentRequest<N>],
    indices: &[usize],
) -> usize {
    if indices
        .iter()
        .all(|&index| items[index].requirement.retention == RetentionHint::ThroughBlindFold)
    {
        1
    } else {
        indices.len().clamp(1, 8)
    }
}

#[derive(Clone, Copy)]
struct ResolvedCommitmentRequest<N: WitnessNamespace> {
    slot: crate::CommitmentSlot,
    requirement: ViewRequirement<N>,
    mode: CommitmentMode,
    trace_polynomial_order: TracePolynomialOrder,
    trace_embedding: Option<TracePolynomialEmbedding>,
    descriptor: OracleDescriptor<N>,
}

impl<N: WitnessNamespace> ResolvedCommitmentRequest<N> {
    fn commitment_layout(
        &self,
        core_fast_path_layout: Option<CoreFastPathLayout>,
    ) -> stream::CommitmentLayout {
        if let Some(embedding) = self.trace_embedding {
            if matches!(
                self.descriptor.encoding,
                PolynomialEncoding::Dense | PolynomialEncoding::Compact
            ) && self.descriptor.dimensions.rows() == embedding.trace_rows
                && matches!(
                    self.requirement.retention,
                    RetentionHint::ThroughStage8 | RetentionHint::ThroughBlindFold
                )
            {
                return stream::CommitmentLayout {
                    dense_layout: stream::DenseCommitmentLayout::TraceEmbedding {
                        row_width: embedded_trace_row_width(embedding),
                        trace_rows: embedding.trace_rows,
                        address_columns: embedding.address_columns,
                        trace_polynomial_order: embedding.trace_polynomial_order,
                    },
                    one_hot_index_order: OneHotIndexOrder::RowMajor,
                    one_hot_streaming: None,
                };
            }
        }
        match self.descriptor.encoding {
            PolynomialEncoding::Dense | PolynomialEncoding::Compact
                if self.requirement.retention == RetentionHint::ThroughStage8 =>
            {
                let Some(core_fast_path_layout) = core_fast_path_layout else {
                    return stream::CommitmentLayout::default_for_request();
                };
                stream::CommitmentLayout {
                    dense_layout: stream::DenseCommitmentLayout::Standard {
                        row_width: Some(core_fast_path_layout.row_width),
                    },
                    one_hot_index_order: OneHotIndexOrder::RowMajor,
                    one_hot_streaming: None,
                }
            }
            PolynomialEncoding::OneHot
                if self.requirement.retention == RetentionHint::ThroughStage8 =>
            {
                let one_hot_index_order = match self.trace_polynomial_order {
                    TracePolynomialOrder::CycleMajor => OneHotIndexOrder::ColumnMajor,
                    TracePolynomialOrder::AddressMajor => OneHotIndexOrder::RowMajor,
                };
                stream::CommitmentLayout {
                    dense_layout: stream::DenseCommitmentLayout::Standard { row_width: None },
                    one_hot_index_order,
                    one_hot_streaming: None,
                }
            }
            _ => stream::CommitmentLayout::default_for_request(),
        }
    }
}

fn embedded_trace_row_width(embedding: TracePolynomialEmbedding) -> usize {
    let full_rows = embedding.trace_rows * embedding.address_columns;
    let total_vars = full_rows.trailing_zeros() as usize;
    1usize << total_vars.div_ceil(2)
}

#[derive(Clone, Copy)]
struct CoreFastPathLayout {
    row_width: usize,
}

fn infer_core_fast_path_layout<N: WitnessNamespace>(
    resolved: &[ResolvedCommitmentRequest<N>],
) -> Option<CoreFastPathLayout> {
    let trace_rows = resolved
        .iter()
        .filter(|item| {
            item.requirement.retention == RetentionHint::ThroughStage8
                && matches!(
                    item.descriptor.encoding,
                    PolynomialEncoding::Dense | PolynomialEncoding::Compact
                )
        })
        .map(|item| item.descriptor.dimensions.rows())
        .min()?;

    let one_hot_rows = resolved
        .iter()
        .filter(|item| {
            item.requirement.retention == RetentionHint::ThroughStage8
                && item.descriptor.encoding == PolynomialEncoding::OneHot
        })
        .map(|item| item.descriptor.dimensions.rows())
        .max()?;

    if trace_rows == 0
        || one_hot_rows <= trace_rows
        || !one_hot_rows.is_power_of_two()
        || !one_hot_rows.is_multiple_of(trace_rows)
    {
        return None;
    }

    let total_vars = one_hot_rows.trailing_zeros() as usize;
    let row_width = 1_usize << total_vars.div_ceil(2);
    if row_width <= trace_rows && trace_rows.is_multiple_of(row_width) {
        Some(CoreFastPathLayout { row_width })
    } else {
        None
    }
}
