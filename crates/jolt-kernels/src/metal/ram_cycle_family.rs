use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::ram::ram_val_final;
use jolt_field::{AkitaField, CanonicalU64};
use jolt_witness::JoltWitnessPlane;

use super::solinas::ram_cycle_family_v3::{
    OwnerConfig, RamAccessRecord, RamCycleFamilyOwner, RamIncrementRecord,
};
use crate::optimized::ram_trace::{RamAccessColumns, RamIncrementActivity};
use crate::ram_access::{RamAccessTape, MAX_RETAINED_RAM_ACCESSES};
use crate::reference::views::dense_view;
use crate::{KernelError, ProofSession};

const RAM_CYCLE_THREADGROUP_WIDTH: usize = 256;
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
        owner
            .verify_integrity()
            .map_err(|error| owner_error(error.to_string()))?;
        return Ok(Some(Arc::clone(owner)));
    }

    let address_domain = 1usize
        .checked_shl(u32::try_from(log_k).map_err(|_| KernelError::Unsupported {
            reason: "RAM cycle-family address domain is too large",
        })?)
        .ok_or(KernelError::Unsupported {
            reason: "RAM cycle-family address domain is too large",
        })?;
    let columns = RamAccessColumns::shared(session, witness, log_t)?;
    columns.validate_addresses(address_domain)?;
    let records = {
        let tape = session
            .state::<RamAccessTape>()
            .ok_or(KernelError::InvariantViolation {
                reason: "RAM access collection did not publish its retained tape",
            })?;
        if tape.validate(log_t, address_domain).is_err() || !tape.hamming_exact() {
            return Ok(None);
        }
        let Some(records) = tape.records() else {
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
    let increments = session
        .state::<RamIncrementActivity>()
        .ok_or(KernelError::InvariantViolation {
            reason: "RAM access collection did not publish increment activity",
        })?
        .records()
        .map(|(cycle, increment)| {
            u64::try_from(cycle)
                .map(|cycle| RamIncrementRecord::new(cycle, increment))
                .map_err(|_| owner_error("RAM increment cycle exceeds the sparse owner ABI"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if records.len().max(increments.len()) > MAX_RETAINED_RAM_ACCESSES {
        return Ok(None);
    }

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
        RAM_CYCLE_THREADGROUP_WIDTH,
        records.len().max(increments.len()).max(1),
    )
    .map_err(|error| owner_error(error.to_string()))?;
    let owner = Arc::new(
        RamCycleFamilyOwner::from_sparse_records(config, records, increments, final_memory)
            .map_err(|error| owner_error(error.to_string()))?,
    );
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
    Ok(Some(owner))
}

fn owner_error(message: impl Into<String>) -> KernelError<AkitaField> {
    KernelError::Sumcheck(jolt_sumcheck::SumcheckError::ComputeBackend {
        backend: "host-sparse",
        message: message.into(),
    })
}
