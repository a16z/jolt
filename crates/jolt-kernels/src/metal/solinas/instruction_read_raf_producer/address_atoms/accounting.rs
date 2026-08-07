use core::mem::size_of;

use super::super::{
    ProducerShardPlan, GROUPED_SEGMENT_OFFSETS, MAX_BUFFER_BYTES, PRODUCER_INPUT_BYTES_PER_ROW,
};
use super::error::{validate_atom_count, AddressAtomError, AddressAtomResult};
use super::topology::AddressAtomLookup;

pub const ADDRESS_ATOM_MASS_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddressAtomPlaneRole {
    AtomLookups,
    AtomClaims,
    AtomCycleOffsets,
    CycleIndices,
    CycleToAtom,
    SegmentAtomOffsets,
}

pub const ADDRESS_ATOM_PLANE_ROLES: [AddressAtomPlaneRole; 6] = [
    AddressAtomPlaneRole::AtomLookups,
    AddressAtomPlaneRole::AtomClaims,
    AddressAtomPlaneRole::AtomCycleOffsets,
    AddressAtomPlaneRole::CycleIndices,
    AddressAtomPlaneRole::CycleToAtom,
    AddressAtomPlaneRole::SegmentAtomOffsets,
];

impl AddressAtomPlaneRole {
    const fn element_bytes(self) -> usize {
        match self {
            Self::AtomLookups => size_of::<AddressAtomLookup>(),
            Self::AtomClaims => size_of::<u8>(),
            Self::AtomCycleOffsets
            | Self::CycleIndices
            | Self::CycleToAtom
            | Self::SegmentAtomOffsets => size_of::<u32>(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressAtomBufferShape {
    role: AddressAtomPlaneRole,
    elements: usize,
    bytes: usize,
}

impl AddressAtomBufferShape {
    pub const fn role(self) -> AddressAtomPlaneRole {
        self.role
    }

    pub const fn elements(self) -> usize {
        self.elements
    }

    pub const fn bytes(self) -> usize {
        self.bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressAtomShape {
    shard: ProducerShardPlan,
    atoms: usize,
}

impl AddressAtomShape {
    pub fn new(shard: ProducerShardPlan, atoms: usize) -> AddressAtomResult<Self> {
        validate_atom_count(shard.rows(), atoms)?;
        let shape = Self { shard, atoms };
        let _shapes = shape.buffer_shapes()?;
        Ok(shape)
    }

    pub const fn shard(self) -> ProducerShardPlan {
        self.shard
    }

    pub const fn rows(self) -> usize {
        self.shard.rows()
    }

    pub const fn atoms(self) -> usize {
        self.atoms
    }

    pub fn buffer_shape(
        self,
        role: AddressAtomPlaneRole,
    ) -> AddressAtomResult<AddressAtomBufferShape> {
        let elements = match role {
            AddressAtomPlaneRole::AtomLookups | AddressAtomPlaneRole::AtomClaims => self.atoms,
            AddressAtomPlaneRole::AtomCycleOffsets => self
                .atoms
                .checked_add(1)
                .ok_or(AddressAtomError::SizeOverflow("atom cycle offsets"))?,
            AddressAtomPlaneRole::CycleIndices | AddressAtomPlaneRole::CycleToAtom => self.rows(),
            AddressAtomPlaneRole::SegmentAtomOffsets => GROUPED_SEGMENT_OFFSETS,
        };
        let bytes = elements
            .checked_mul(role.element_bytes())
            .ok_or(AddressAtomError::SizeOverflow("address atom buffer"))?;
        if bytes > MAX_BUFFER_BYTES {
            return Err(AddressAtomError::BufferTooLarge { role, bytes });
        }
        Ok(AddressAtomBufferShape {
            role,
            elements,
            bytes,
        })
    }

    pub fn buffer_shapes(self) -> AddressAtomResult<[AddressAtomBufferShape; 6]> {
        Ok([
            self.buffer_shape(AddressAtomPlaneRole::AtomLookups)?,
            self.buffer_shape(AddressAtomPlaneRole::AtomClaims)?,
            self.buffer_shape(AddressAtomPlaneRole::AtomCycleOffsets)?,
            self.buffer_shape(AddressAtomPlaneRole::CycleIndices)?,
            self.buffer_shape(AddressAtomPlaneRole::CycleToAtom)?,
            self.buffer_shape(AddressAtomPlaneRole::SegmentAtomOffsets)?,
        ])
    }

    pub fn topology_bytes(self) -> AddressAtomResult<u64> {
        self.buffer_shapes()?
            .into_iter()
            .try_fold(0u64, |sum, shape| {
                sum.checked_add(shape.bytes as u64)
                    .ok_or(AddressAtomError::SizeOverflow("topology resident bytes"))
            })
    }

    pub fn v3_handoff_bytes(self) -> AddressAtomResult<u64> {
        self.topology_bytes()?
            .checked_sub(self.buffer_shape(AddressAtomPlaneRole::CycleToAtom)?.bytes as u64)
            .ok_or(AddressAtomError::SizeOverflow("v3 handoff bytes"))
    }

    pub fn mass_bytes(self) -> AddressAtomResult<u64> {
        (self.atoms as u64)
            .checked_mul(ADDRESS_ATOM_MASS_BYTES as u64)
            .ok_or(AddressAtomError::SizeOverflow("atom mass bytes"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressAtomTraffic {
    rows: u64,
    atoms: u64,
    source_read_bytes: u64,
    topology_write_bytes: u64,
    v3_handoff_bytes: u64,
    mass_floor_bytes: u64,
    topology_and_mass_bytes: u64,
}

impl AddressAtomTraffic {
    /// Logical payload only; sort/radix scratch and cache-line amplification are separate.
    pub fn for_shape(shape: AddressAtomShape) -> AddressAtomResult<Self> {
        let rows = shape.rows() as u64;
        let atoms = shape.atoms() as u64;
        let source_read_bytes = rows
            .checked_mul(PRODUCER_INPUT_BYTES_PER_ROW as u64)
            .ok_or(AddressAtomError::SizeOverflow("atom source read bytes"))?;
        let topology_write_bytes = shape.topology_bytes()?;
        let v3_handoff_bytes = shape.v3_handoff_bytes()?;
        let mass_floor_bytes = rows
            .checked_mul(36)
            .and_then(|bytes| bytes.checked_add(atoms.checked_mul(16)?))
            .ok_or(AddressAtomError::SizeOverflow("atom mass payload floor"))?;
        let topology_and_mass_bytes = topology_write_bytes
            .checked_add(shape.mass_bytes()?)
            .ok_or(AddressAtomError::SizeOverflow("topology and mass bytes"))?;
        Ok(Self {
            rows,
            atoms,
            source_read_bytes,
            topology_write_bytes,
            v3_handoff_bytes,
            mass_floor_bytes,
            topology_and_mass_bytes,
        })
    }

    pub const fn rows(self) -> u64 {
        self.rows
    }

    pub const fn atoms(self) -> u64 {
        self.atoms
    }

    pub const fn source_read_bytes(self) -> u64 {
        self.source_read_bytes
    }

    pub const fn topology_write_bytes(self) -> u64 {
        self.topology_write_bytes
    }

    pub fn standalone_build_bytes(self) -> AddressAtomResult<u64> {
        self.source_read_bytes
            .checked_add(self.topology_write_bytes)
            .ok_or(AddressAtomError::SizeOverflow(
                "standalone atom build bytes",
            ))
    }

    pub const fn co_produced_build_bytes(self) -> u64 {
        self.topology_write_bytes
    }

    pub const fn v3_handoff_bytes(self) -> u64 {
        self.v3_handoff_bytes
    }

    pub const fn mass_floor_bytes(self) -> u64 {
        self.mass_floor_bytes
    }

    pub const fn topology_and_mass_bytes(self) -> u64 {
        self.topology_and_mass_bytes
    }

    pub fn live_with_source_bytes(self) -> AddressAtomResult<u64> {
        self.source_read_bytes
            .checked_add(self.topology_and_mass_bytes)
            .ok_or(AddressAtomError::SizeOverflow("source topology live bytes"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressAtomPartitionPenalty {
    duplicate_atoms: u64,
    topology_bytes: u64,
    mass_bytes: u64,
    address_state_bytes: u64,
}

impl AddressAtomPartitionPenalty {
    /// Minimum penalty before extra split-atom jobs or partials are known.
    pub fn new(shard_atoms: u64, globally_unique_atoms: u64) -> AddressAtomResult<Self> {
        let duplicate_atoms = shard_atoms.checked_sub(globally_unique_atoms).ok_or(
            AddressAtomError::InvalidTopology(
                "global atom count exceeds the sum of shard atom counts",
            ),
        )?;
        Ok(Self {
            duplicate_atoms,
            topology_bytes: duplicate_atoms
                .checked_mul(21)
                .ok_or(AddressAtomError::SizeOverflow("partition topology penalty"))?,
            mass_bytes: duplicate_atoms
                .checked_mul(16)
                .ok_or(AddressAtomError::SizeOverflow("partition mass penalty"))?,
            address_state_bytes: duplicate_atoms
                .checked_mul(736)
                .ok_or(AddressAtomError::SizeOverflow("partition address penalty"))?,
        })
    }

    pub const fn duplicate_atoms(self) -> u64 {
        self.duplicate_atoms
    }

    pub const fn topology_bytes(self) -> u64 {
        self.topology_bytes
    }

    pub const fn mass_bytes(self) -> u64 {
        self.mass_bytes
    }

    pub const fn address_state_bytes(self) -> u64 {
        self.address_state_bytes
    }
}
