use std::slice;
use std::time::Duration;

use jolt_field::AkitaField;
use jolt_poly::TensorEqTable;

use super::{
    invalid_source, InstructionReadRafCompatibilityScatterConfig, INSTRUCTION_READ_RAF_SEGMENTS,
};
use crate::metal::solinas::{
    AddressPhaseSequenceConfig, AddressRafScanRow, BooleanityRow, Fp128, MetalError, SolinasMetal,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionReadRafStage1ProbeResult {
    pub rows: usize,
    pub threads_per_threadgroup: usize,
    pub fused_grouped_phase: bool,
    pub scatter_gpu_active: Duration,
    pub scatter_wall: Duration,
    pub baseline_phase_gpu_active: Duration,
    pub resident_phase_gpu_active: Duration,
    pub exact: bool,
}

pub fn run_instruction_read_raf_stage1_probe(
    context: &SolinasMetal,
    log_rows: usize,
    threads_per_threadgroup: usize,
    fused_grouped_phase: bool,
) -> Result<InstructionReadRafStage1ProbeResult, MetalError> {
    if !(7..=20).contains(&log_rows) {
        return Err(invalid_source("probe log rows must be in 7..=20"));
    }
    let rows = 1usize << log_rows;
    let mut storage = context.prepare_instruction_read_raf_stage1_storage(rows)?;
    storage.with_chunk_writers(|chunks| {
        for (chunk_index, chunk) in chunks.iter_mut().enumerate() {
            let chunk_start = chunk_index * super::INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
            for local in 0..chunk.len() {
                let cycle = chunk_start + local;
                let logical = cycle % INSTRUCTION_READ_RAF_SEGMENTS;
                let table_plus_one = (logical / 2) as u8;
                let raf = !logical.is_multiple_of(2);
                chunk.push(
                    BooleanityRow::new(probe_lookup(cycle), Some(cycle as u64), None, 0)?,
                    table_plus_one,
                    raf,
                )?;
            }
        }
        Ok(())
    })?;
    let owner = storage.seal()?;
    let point: Vec<_> = (0..log_rows)
        .map(|index| AkitaField::from_u64(0x100 + 17 * index as u64))
        .collect();
    let equality = TensorEqTable::new(&point);
    let weights: Vec<_> = (0..rows)
        .map(|cycle| Fp128::from_jolt_field(&equality.evaluate_index(cycle)))
        .collect();
    let source_rows: Vec<_> = (0..rows)
        .map(|cycle| {
            let logical = cycle % INSTRUCTION_READ_RAF_SEGMENTS;
            let table_plus_one = logical / 2;
            AddressRafScanRow::new_with_table(
                probe_lookup(cycle),
                (table_plus_one != 0).then_some(table_plus_one - 1),
                !logical.is_multiple_of(2),
            )
        })
        .collect();

    let planes = context.prepare_instruction_read_raf_compatibility_scatter(
        owner.lease(rows, context.device_registry_id())?,
        &point,
        InstructionReadRafCompatibilityScatterConfig {
            threads_per_threadgroup,
        },
    )?;
    let execution = planes.execution();
    validate_probe_planes(&planes, &weights)?;

    let config = AddressPhaseSequenceConfig {
        fused_grouped_phase,
        ..AddressPhaseSequenceConfig::default()
    };
    let mut baseline = context.prepare_address_phase_sequence(&source_rows, &weights, config)?;
    let mut resident =
        context.prepare_address_phase_sequence_from_resident_grouped(planes, config)?;
    let baseline_sums = baseline.phase(120, None)?;
    let resident_sums = resident.phase(120, None)?;
    let exact = baseline_sums.raf() == resident_sums.raf()
        && baseline_sums.suffix() == resident_sums.suffix();
    if !exact {
        return Err(invalid_source(
            "resident grouped first phase disagrees with the host-prepared sequence",
        ));
    }
    Ok(InstructionReadRafStage1ProbeResult {
        rows,
        threads_per_threadgroup,
        fused_grouped_phase,
        scatter_gpu_active: execution.gpu_active,
        scatter_wall: execution.command_wall,
        baseline_phase_gpu_active: baseline_sums.gpu_active_time(),
        resident_phase_gpu_active: resident_sums.gpu_active_time(),
        exact,
    })
}

fn validate_probe_planes(
    planes: &super::InstructionReadRafDenseGroupedPlanes,
    expected_weights: &[Fp128],
) -> Result<(), MetalError> {
    let receipt = planes.receipt();
    let rows = receipt.rows();
    let [packed_buffer, lookup_buffer, inverse_buffer, weight_buffer] = planes.buffers();
    // SAFETY: the completed receipt proves exact initialized lengths for all
    // four StorageModeShared buffers, which remain alive through this check.
    let packed = unsafe { slice::from_raw_parts(packed_buffer.contents().cast::<u8>(), rows) };
    // SAFETY: see the completed typed receipt argument above.
    let lookups =
        unsafe { slice::from_raw_parts(lookup_buffer.contents().cast::<u64>(), 2 * rows) };
    // SAFETY: see the completed typed receipt argument above.
    let inverse = unsafe { slice::from_raw_parts(inverse_buffer.contents().cast::<u32>(), rows) };
    // SAFETY: see the completed typed receipt argument above.
    let weights = unsafe { slice::from_raw_parts(weight_buffer.contents().cast::<Fp128>(), rows) };
    let mut visited = vec![false; rows];
    for cycle in 0..rows {
        let grouped = inverse[cycle] as usize;
        if grouped >= rows || visited[grouped] {
            return Err(invalid_source("probe inverse is not a permutation"));
        }
        visited[grouped] = true;
        let logical = cycle % INSTRUCTION_READ_RAF_SEGMENTS;
        let table_plus_one = (logical / 2) as u8;
        let expected_claim = table_plus_one | (((logical % 2) as u8) << 7);
        let lookup = probe_lookup(cycle);
        if !receipt.segment_ranges()[logical].contains(&grouped)
            || packed[grouped] != expected_claim
            || lookups[2 * grouped] != lookup as u64
            || lookups[2 * grouped + 1] != (lookup >> 64) as u64
            || weights[grouped] != expected_weights[cycle]
        {
            return Err(invalid_source("probe grouped plane parity failed"));
        }
    }
    Ok(())
}

fn probe_lookup(cycle: usize) -> u128 {
    ((cycle as u128) << 67) ^ (cycle as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}
