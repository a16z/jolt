//! Checked shard-local CSR for exact InstructionReadRaf address atoms.

mod accounting;
mod error;
mod receipts;
mod topology;

pub use accounting::{
    AddressAtomBufferShape, AddressAtomPartitionPenalty, AddressAtomPlaneRole, AddressAtomShape,
    AddressAtomTraffic, ADDRESS_ATOM_MASS_BYTES, ADDRESS_ATOM_PLANE_ROLES,
};
pub use error::{AddressAtomError, AddressAtomResult};
pub use receipts::{
    AddressAtomMassReceipt, AddressAtomPlaneReceipt, AddressAtomSourceProvenance,
    AddressAtomTopologyBatchReceipt, AddressAtomTopologyReceipt,
};
pub use topology::{
    split_equality_weight, AddressAtomCycleRow, AddressAtomCycleSource, AddressAtomLookup,
    AddressAtomTopology, AddressAtomTopologyParts,
};
