use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::ram::ram_val_final;
use jolt_field::{AkitaField, CanonicalU64};
use jolt_witness::JoltWitnessPlane;

use super::solinas::ram_cycle_family::{
    OwnerConfig, RamAccessRecord, RamCycleFamilyOwner, RamIncrementRecord,
};
use crate::optimized::ram_trace::{RamAccessColumns, RamIncrementActivity};
use crate::ram_access::{RamAccessTape, MAX_RETAINED_RAM_ACCESSES};
use crate::reference::views::dense_view;
use crate::{KernelError, ProofSession};

static NEXT_RAM_CYCLE_GENERATION: AtomicU64 = AtomicU64::new(1);

pub(super) fn shared_ram_cycle_family_owner(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    log_t: usize,
    log_k: usize,
) -> Result<Option<Arc<RamCycleFamilyOwner>>, KernelError<AkitaField>> {
    if let Some(owner) = session.state::<Arc<RamCycleFamilyOwner>>() {
        let receipt = owner.receipt();
        if receipt.log_t() != log_t || receipt.log_k() != log_k {
            return Err(KernelError::InvariantViolation {
                reason: "RAM cycle-family owner has stale geometry",
            });
        }
        return Ok(Some(Arc::clone(owner)));
    }

    let address_domain = 1usize
        .checked_shl(u32::try_from(log_k).map_err(|_| KernelError::Unsupported {
            reason: "RAM cycle-family address domain is too large",
        })?)
        .ok_or(KernelError::Unsupported {
            reason: "RAM cycle-family address domain is too large",
        })?;
    let cycles = 1usize
        .checked_shl(u32::try_from(log_t).map_err(|_| KernelError::Unsupported {
            reason: "RAM cycle-family cycle domain is too large",
        })?)
        .ok_or(KernelError::Unsupported {
            reason: "RAM cycle-family cycle domain is too large",
        })?;
    let source_collection_performed = session.state::<Arc<RamAccessColumns>>().is_none();
    let owner_span = tracing::info_span!(
        "MetalRamCycleFamily::owner_prepare",
        enabled = true,
        schema_version = super::solinas::ram_cycle_family::RAM_CYCLE_FAMILY_SCHEMA_VERSION,
        source_kind = "ram_access_tape_v1",
        source_generation = tracing::field::Empty,
        source_fingerprint = tracing::field::Empty,
        log_t,
        log_k,
        cycles,
        address_domain,
        access_count = tracing::field::Empty,
        increment_count = tracing::field::Empty,
        access_records = tracing::field::Empty,
        increment_records = tracing::field::Empty,
        retained_records = tracing::field::Empty,
        increment_compatible = tracing::field::Empty,
        ram_ra_compatible = tracing::field::Empty,
        hamming_exact = tracing::field::Empty,
        rejection_reason = tracing::field::Empty,
        final_memory_elements = address_domain,
        record_bytes = tracing::field::Empty,
        final_memory_bytes = tracing::field::Empty,
        block_topology_nodes = tracing::field::Empty,
        topology_bytes = tracing::field::Empty,
        owner_bytes = tracing::field::Empty,
        source_rows = cycles,
        source_collection_performed,
        shared_source_row_scans = usize::from(source_collection_performed),
        additional_source_row_scans = 0,
        member_upload_bytes = 0,
        complete_publication = tracing::field::Empty,
    );
    let _owner_guard = owner_span.enter();
    let columns = RamAccessColumns::shared(session, witness, log_t)?;
    columns.validate_addresses(address_domain)?;
    let activity =
        session
            .state::<Arc<RamIncrementActivity>>()
            .ok_or(KernelError::InvariantViolation {
                reason: "RAM access collection did not publish increment activity",
            })?;
    let records = {
        let tape = session
            .state::<RamAccessTape>()
            .ok_or(KernelError::InvariantViolation {
                reason: "RAM access collection did not publish its retained tape",
            })?;
        let _ = owner_span.record("access_count", tape.access_count());
        let _ = owner_span.record("increment_count", activity.len());
        let _ = owner_span.record(
            "retained_records",
            tape.records()
                .map_or(0, <[crate::ram_access::RamAccessRecord]>::len),
        );
        let _ = owner_span.record("increment_compatible", tape.increment_compatible());
        let _ = owner_span.record("ram_ra_compatible", tape.ram_ra_compatible());
        let _ = owner_span.record("hamming_exact", tape.hamming_exact());
        if let Err(error) = tape.validate(log_t, address_domain) {
            let _ = owner_span.record("rejection_reason", tracing::field::display(error));
            let _ = owner_span.record("complete_publication", false);
            return Ok(None);
        }
        if !tape.hamming_exact() {
            let _ = owner_span.record("rejection_reason", "hamming_inexact");
            let _ = owner_span.record("complete_publication", false);
            return Ok(None);
        }
        if tape.access_count().max(activity.len()) > MAX_RETAINED_RAM_ACCESSES {
            let _ = owner_span.record("rejection_reason", "retained_access_cap");
            let _ = owner_span.record("complete_publication", false);
            return Ok(None);
        }
        let Some(records) = tape.records() else {
            let _ = owner_span.record("rejection_reason", "records_unretained");
            let _ = owner_span.record("complete_publication", false);
            return Ok(None);
        };
        records
            .iter()
            .map(|record| {
                RamAccessRecord::new(
                    record.cycle,
                    record.address,
                    record.pre_value,
                    record.post_value,
                )
            })
            .collect::<Vec<_>>()
    };
    let increments = activity
        .records()
        .map(|(cycle, increment)| {
            u64::try_from(cycle)
                .map(|cycle| RamIncrementRecord::new(cycle, increment))
                .map_err(|_| owner_error("RAM increment cycle exceeds the sparse owner ABI"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let final_memory = dense_view::<AkitaField>(witness, ram_val_final())?;
    if final_memory.len() != address_domain {
        return Err(KernelError::TableSizeMismatch {
            table: format!("{:?}", ram_val_final()),
            expected: address_domain,
            got: final_memory.len(),
        });
    }
    let final_memory = final_memory
        .iter()
        .map(|value| {
            value.to_canonical_u64_checked().ok_or_else(|| {
                owner_error("RAM final memory is not canonically representable as u64")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let generation = NEXT_RAM_CYCLE_GENERATION.fetch_add(1, Ordering::Relaxed);
    if generation == 0 {
        return Err(KernelError::InvariantViolation {
            reason: "RAM cycle-family owner generation wrapped",
        });
    }
    let config = OwnerConfig::new(
        log_t,
        log_k,
        generation,
        records.len().max(increments.len()).max(1),
    )
    .map_err(|error| owner_error(error.to_string()))?;
    let owner = Arc::new(
        RamCycleFamilyOwner::from_sparse_records(config, records, increments, final_memory)
            .map_err(|error| owner_error(error.to_string()))?,
    );
    let receipt = owner.receipt();
    let block_topology_nodes = receipt
        .block_census()
        .iter()
        .map(|level| level.entries())
        .sum::<u64>();
    let record_bytes = std::mem::size_of::<RamAccessRecord>()
        .checked_mul(receipt.access_count())
        .and_then(|bytes| {
            (std::mem::size_of::<u64>() + std::mem::size_of::<i128>())
                .checked_mul(receipt.increment_count())
                .and_then(|increment_bytes| bytes.checked_add(increment_bytes))
        })
        .ok_or(KernelError::InvariantViolation {
            reason: "RAM cycle-family owner byte ledger overflowed",
        })?;
    let final_memory_bytes = std::mem::size_of::<u64>()
        .checked_mul(receipt.address_domain())
        .ok_or(KernelError::InvariantViolation {
            reason: "RAM cycle-family final-memory byte ledger overflowed",
        })?;
    let owner_bytes = owner.owned_heap_bytes();
    let topology_bytes = owner_bytes
        .checked_sub(record_bytes)
        .and_then(|bytes| bytes.checked_sub(final_memory_bytes))
        .ok_or(KernelError::InvariantViolation {
            reason: "RAM cycle-family topology byte ledger underflowed",
        })?;
    let _ = owner_span.record("source_generation", receipt.source_generation());
    let _ = owner_span.record("source_fingerprint", receipt.fingerprint());
    let _ = owner_span.record("access_records", receipt.access_count());
    let _ = owner_span.record("increment_records", receipt.increment_count());
    let _ = owner_span.record("rejection_reason", "none");
    let _ = owner_span.record("record_bytes", record_bytes);
    let _ = owner_span.record("final_memory_bytes", final_memory_bytes);
    let _ = owner_span.record("block_topology_nodes", block_topology_nodes);
    let _ = owner_span.record("topology_bytes", topology_bytes);
    let _ = owner_span.record("owner_bytes", owner_bytes);
    tracing::info!(
        target: "jolt::metal",
        generation,
        cycles = owner.receipt().cycles(),
        accesses = owner.receipt().access_count(),
        increments = owner.receipt().increment_count(),
        topology_nodes = owner
            .receipt()
            .block_census()
            .iter()
            .map(|level| level.entries())
            .sum::<u64>(),
        "prepared shared RAM cycle-family owner"
    );
    session.park(Arc::clone(&owner));
    let _ = owner_span.record("complete_publication", true);
    Ok(Some(owner))
}

fn owner_error(message: impl Into<String>) -> KernelError<AkitaField> {
    KernelError::Sumcheck(jolt_sumcheck::SumcheckError::ComputeBackend {
        backend: "host-sparse",
        message: message.into(),
    })
}
