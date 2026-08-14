use super::super::{ProducerGeometry, ProducerShardPlan};
use super::accounting::{AddressAtomPlaneRole, AddressAtomShape, ADDRESS_ATOM_PLANE_ROLES};
use super::error::{AddressAtomError, AddressAtomResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressAtomSourceProvenance {
    shard: ProducerShardPlan,
    device_registry_id: u64,
    source_generation: u64,
    source_completion_serial: u64,
    allocation_identities: [usize; 3],
}

impl AddressAtomSourceProvenance {
    pub fn new(
        shard: ProducerShardPlan,
        device_registry_id: u64,
        source_generation: u64,
        source_completion_serial: u64,
        allocation_identities: [usize; 3],
    ) -> AddressAtomResult<Self> {
        nonzero("device registry", device_registry_id)?;
        nonzero("source generation", source_generation)?;
        nonzero("source completion serial", source_completion_serial)?;
        validate_identities(&allocation_identities)?;
        Ok(Self {
            shard,
            device_registry_id,
            source_generation,
            source_completion_serial,
            allocation_identities,
        })
    }

    pub const fn shard(self) -> ProducerShardPlan {
        self.shard
    }

    pub const fn device_registry_id(self) -> u64 {
        self.device_registry_id
    }

    pub const fn source_generation(self) -> u64 {
        self.source_generation
    }

    pub const fn source_completion_serial(self) -> u64 {
        self.source_completion_serial
    }

    pub const fn allocation_identities(self) -> [usize; 3] {
        self.allocation_identities
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressAtomPlaneReceipt {
    role: AddressAtomPlaneRole,
    elements: usize,
    bytes: u64,
    device_registry_id: u64,
    allocation_identity: usize,
    initialized_generation: u64,
    completion_serial: u64,
}

impl AddressAtomPlaneReceipt {
    pub fn new(
        role: AddressAtomPlaneRole,
        elements: usize,
        bytes: u64,
        device_registry_id: u64,
        allocation_identity: usize,
        initialized_generation: u64,
        completion_serial: u64,
    ) -> AddressAtomResult<Self> {
        nonzero("device registry", device_registry_id)?;
        nonzero_usize("topology allocation", allocation_identity)?;
        nonzero("topology generation", initialized_generation)?;
        nonzero("topology completion serial", completion_serial)?;
        Ok(Self {
            role,
            elements,
            bytes,
            device_registry_id,
            allocation_identity,
            initialized_generation,
            completion_serial,
        })
    }

    pub const fn role(self) -> AddressAtomPlaneRole {
        self.role
    }

    pub const fn elements(self) -> usize {
        self.elements
    }

    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    pub const fn allocation_identity(self) -> usize {
        self.allocation_identity
    }

    pub const fn device_registry_id(self) -> u64 {
        self.device_registry_id
    }

    pub const fn initialized_generation(self) -> u64 {
        self.initialized_generation
    }

    pub const fn completion_serial(self) -> u64 {
        self.completion_serial
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressAtomTopologyReceipt {
    shape: AddressAtomShape,
    source: AddressAtomSourceProvenance,
    completion_serial: u64,
    planes: [AddressAtomPlaneReceipt; 6],
}

impl AddressAtomTopologyReceipt {
    pub fn new(
        shape: AddressAtomShape,
        source: AddressAtomSourceProvenance,
        completion_serial: u64,
        status: u32,
        planes: [AddressAtomPlaneReceipt; 6],
    ) -> AddressAtomResult<Self> {
        nonzero("topology completion serial", completion_serial)?;
        if status != 0 {
            return Err(AddressAtomError::NonzeroStatus {
                shard: shape.shard().shard_index(),
                status,
            });
        }
        if shape.shard() != source.shard() {
            return Err(AddressAtomError::ShardMismatch);
        }
        if completion_serial < source.source_completion_serial() {
            return Err(AddressAtomError::IncompleteTopology {
                minimum: source.source_completion_serial(),
                got: completion_serial,
            });
        }

        let expected = shape.buffer_shapes()?;
        for ((plane, expected), role) in planes.iter().zip(expected).zip(ADDRESS_ATOM_PLANE_ROLES) {
            if plane.role != role
                || expected.role() != role
                || plane.elements != expected.elements()
                || plane.bytes != expected.bytes() as u64
            {
                return Err(AddressAtomError::PlaneShape {
                    role,
                    expected_elements: expected.elements(),
                    got_elements: plane.elements,
                    expected_bytes: expected.bytes() as u64,
                    got_bytes: plane.bytes,
                });
            }
            if plane.device_registry_id != source.device_registry_id() {
                return Err(AddressAtomError::DeviceMismatch { role });
            }
            if plane.initialized_generation != source.source_generation() {
                return Err(AddressAtomError::GenerationMismatch {
                    role,
                    expected: source.source_generation(),
                    got: plane.initialized_generation,
                });
            }
            if plane.completion_serial != completion_serial {
                return Err(AddressAtomError::PlaneCompletionMismatch {
                    role,
                    expected: completion_serial,
                    got: plane.completion_serial,
                });
            }
        }
        let output_ids = planes.map(|plane| plane.allocation_identity);
        validate_identities(&output_ids)?;
        reject_cross_aliases(&source.allocation_identities(), &output_ids)?;
        Ok(Self {
            shape,
            source,
            completion_serial,
            planes,
        })
    }

    pub const fn shape(self) -> AddressAtomShape {
        self.shape
    }

    pub const fn source(self) -> AddressAtomSourceProvenance {
        self.source
    }

    pub const fn completion_serial(self) -> u64 {
        self.completion_serial
    }

    pub const fn planes(self) -> [AddressAtomPlaneReceipt; 6] {
        self.planes
    }

    pub fn allocation_identities(self) -> [usize; 6] {
        self.planes.map(|plane| plane.allocation_identity)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressAtomTopologyBatchReceipt {
    total_rows: usize,
    receipts: Vec<AddressAtomTopologyReceipt>,
}

impl AddressAtomTopologyBatchReceipt {
    pub fn new(
        total_rows: usize,
        receipts: Vec<AddressAtomTopologyReceipt>,
    ) -> AddressAtomResult<Self> {
        let geometry = ProducerGeometry::new(total_rows)?;
        if receipts.len() != geometry.shard_count() {
            return Err(AddressAtomError::ReceiptShardCount {
                expected: geometry.shard_count(),
                got: receipts.len(),
            });
        }
        let first = receipts
            .first()
            .ok_or(AddressAtomError::ReceiptShardCount {
                expected: geometry.shard_count(),
                got: 0,
            })?;
        let mut identities = Vec::with_capacity(receipts.len() * 9);
        for (index, receipt) in receipts.iter().enumerate() {
            let expected = geometry.shard(index)?;
            if receipt.shape.shard() != expected {
                return Err(AddressAtomError::ReceiptShard { index });
            }
            if receipt.source.device_registry_id() != first.source.device_registry_id()
                || receipt.source.source_generation() != first.source.source_generation()
                || receipt.source.source_completion_serial()
                    != first.source.source_completion_serial()
                || receipt.completion_serial != first.completion_serial
            {
                return Err(AddressAtomError::BatchProvenanceMismatch { index });
            }
            identities.extend(receipt.source.allocation_identities());
            identities.extend(receipt.allocation_identities());
        }
        validate_identities(&identities)?;
        Ok(Self {
            total_rows,
            receipts,
        })
    }

    pub const fn total_rows(&self) -> usize {
        self.total_rows
    }

    pub fn receipts(&self) -> &[AddressAtomTopologyReceipt] {
        &self.receipts
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressAtomMassReceipt {
    topology: AddressAtomTopologyReceipt,
    allocation_identity: usize,
    completion_serial: u64,
    challenge_digest: u64,
}

impl AddressAtomMassReceipt {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        topology: AddressAtomTopologyReceipt,
        elements: usize,
        bytes: u64,
        device_registry_id: u64,
        allocation_identity: usize,
        initialized_generation: u64,
        completion_serial: u64,
        challenge_digest: u64,
    ) -> AddressAtomResult<Self> {
        nonzero_usize("atom mass allocation", allocation_identity)?;
        nonzero("atom mass completion serial", completion_serial)?;
        nonzero("atom mass challenge digest", challenge_digest)?;
        let expected_bytes = topology.shape.mass_bytes()?;
        if elements != topology.shape.atoms() || bytes != expected_bytes {
            return Err(AddressAtomError::MassPlaneShape {
                expected_elements: topology.shape.atoms(),
                got_elements: elements,
                expected_bytes,
                got_bytes: bytes,
            });
        }
        if device_registry_id != topology.source.device_registry_id() {
            return Err(AddressAtomError::MassDeviceMismatch);
        }
        if initialized_generation != topology.source.source_generation() {
            return Err(AddressAtomError::MassGenerationMismatch {
                expected: topology.source.source_generation(),
                got: initialized_generation,
            });
        }
        if completion_serial < topology.completion_serial {
            return Err(AddressAtomError::IncompleteMasses {
                minimum: topology.completion_serial,
                got: completion_serial,
            });
        }
        if topology
            .source
            .allocation_identities()
            .contains(&allocation_identity)
            || topology
                .allocation_identities()
                .contains(&allocation_identity)
        {
            return Err(AddressAtomError::AliasedAllocation {
                identity: allocation_identity,
            });
        }
        Ok(Self {
            topology,
            allocation_identity,
            completion_serial,
            challenge_digest,
        })
    }

    pub const fn topology(self) -> AddressAtomTopologyReceipt {
        self.topology
    }

    pub const fn allocation_identity(self) -> usize {
        self.allocation_identity
    }

    pub const fn completion_serial(self) -> u64 {
        self.completion_serial
    }

    pub const fn challenge_digest(self) -> u64 {
        self.challenge_digest
    }
}

fn nonzero(name: &'static str, value: u64) -> AddressAtomResult<()> {
    if value == 0 {
        return Err(AddressAtomError::MissingIdentity { name });
    }
    Ok(())
}

fn nonzero_usize(name: &'static str, value: usize) -> AddressAtomResult<()> {
    if value == 0 {
        return Err(AddressAtomError::MissingIdentity { name });
    }
    Ok(())
}

fn validate_identities(identities: &[usize]) -> AddressAtomResult<()> {
    for (index, &identity) in identities.iter().enumerate() {
        nonzero_usize("allocation", identity)?;
        if identities[..index].contains(&identity) {
            return Err(AddressAtomError::AliasedAllocation { identity });
        }
    }
    Ok(())
}

fn reject_cross_aliases(left: &[usize], right: &[usize]) -> AddressAtomResult<()> {
    if let Some(&identity) = left.iter().find(|&&identity| right.contains(&identity)) {
        return Err(AddressAtomError::AliasedAllocation { identity });
    }
    Ok(())
}
