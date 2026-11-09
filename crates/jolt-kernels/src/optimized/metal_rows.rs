use std::sync::{Arc, Mutex, MutexGuard, Weak};

use jolt_akita::AkitaField;
use jolt_metal::runtime::{Device, DeviceBuffer};
use jolt_metal::MetalError;

use super::constant_suffix::ConstantSuffix;
use super::instruction_read_raf::InstructionCycleRow;
use super::support::map_indices;
use crate::{KernelError, ProofSession};

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct SharedMetalRows(Weak<ResidentInstructionRows>);

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct NextMetalRows(Arc<ResidentInstructionRows>);

/// Proof-owned compact selectors. The session retains only a weak reference;
/// a producer may carry it to the next stage; consumers release it after materialization.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct ResidentInstructionRows {
    source: Weak<ConstantSuffix<InstructionCycleRow>>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    uploaded: Mutex<Option<UploadedRows>>,
}

pub(crate) struct UploadedRows {
    /// Lookup limbs, PC in low 56 bits plus claim byte in high 8 bits,
    /// then RAM-address-plus-one. PC consumers must mask off the claim byte.
    pub(crate) prefix: DeviceBuffer<[u64; 4]>,
    pub(crate) tail: [u64; 4],
}

impl ResidentInstructionRows {
    pub(crate) fn shared(
        session: &mut ProofSession,
        rows: &Arc<ConstantSuffix<InstructionCycleRow>>,
    ) -> Result<Arc<Self>, KernelError<AkitaField>> {
        let source = Arc::downgrade(rows);
        if let Some(carry) = session.take::<NextMetalRows>() {
            if !Weak::ptr_eq(&carry.0.source, &source) {
                return Err(KernelError::InvariantViolation {
                    reason: "Metal next-stage rows changed source owner",
                });
            }
            session.park(SharedMetalRows(Arc::downgrade(&carry.0)));
            return Ok(carry.0);
        }
        if let Some(owner) = session
            .state::<SharedMetalRows>()
            .and_then(|carry| carry.0.upgrade())
        {
            if !Weak::ptr_eq(&owner.source, &source) {
                return Err(KernelError::InvariantViolation {
                    reason: "Metal RA kernels received different instruction-row owners",
                });
            }
            return Ok(owner);
        }
        let owner = Arc::new(Self {
            source,
            uploaded: Mutex::new(None),
        });
        session.park(SharedMetalRows(Arc::downgrade(&owner)));
        Ok(owner)
    }

    pub(crate) fn carry_to_next_stage(self: Arc<Self>, session: &mut ProofSession) {
        session.park(NextMetalRows(self));
    }

    pub(crate) fn resident(
        &self,
        device: &Device,
    ) -> Result<MutexGuard<'_, Option<UploadedRows>>, MetalError> {
        let invalid = |reason: &str| MetalError::InvalidDispatch {
            pipeline: "compact RA rows".into(),
            reason: reason.into(),
        };
        let mut uploaded = self
            .uploaded
            .lock()
            .map_err(|_| invalid("row cache lock poisoned"))?;
        if uploaded.is_none() {
            let _span = tracing::info_span!("metal_ra_rows_upload").entered();
            let rows = self
                .source
                .upgrade()
                .ok_or_else(|| invalid("instruction rows were released before upload"))?;
            let pack = |row: &InstructionCycleRow| {
                let index = row.lookup_index();
                [
                    index as u64,
                    (index >> 64) as u64,
                    row.bytecode_pc() as u64 | (u64::from(row.claim_byte()) << 56),
                    row.remapped_ram_address().map_or(0, |address| address + 1),
                ]
            };
            let tail = rows
                .representatives()
                .last()
                .ok_or_else(|| invalid("empty instruction domain"))?;
            let values = map_indices(rows.prefix_len(), |row| pack(&rows[row]));
            *uploaded = Some(UploadedRows {
                prefix: DeviceBuffer::from_slice(device, &values)?,
                tail: pack(tail),
            });
        }
        Ok(uploaded)
    }
}
