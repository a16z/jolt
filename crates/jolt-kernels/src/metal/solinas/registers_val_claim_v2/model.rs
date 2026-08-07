use core::mem::size_of;
use std::sync::Arc;

use jolt_field::Field;
use thiserror::Error;

use super::super::registers::{CertifiedRegisterOwner, RegisterEventCounts, REGISTER_CSR_COLUMNS};

pub const REGISTER_ADDRESS_BITS: usize = 7;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegisterOwnerSourceKind {
    OwnedRandomAccess,
    Streamed,
}

/// Identity shared by every register-family carrier in one proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisterOwnerIdentity {
    proof_generation: u64,
    source_kind: RegisterOwnerSourceKind,
    source_id: u64,
    owner_id: u64,
}

impl RegisterOwnerIdentity {
    pub fn new(
        proof_generation: u64,
        source_kind: RegisterOwnerSourceKind,
        source_id: u64,
        owner_id: u64,
    ) -> Result<Self, RegisterFamilyModelError> {
        for (field, value) in [
            ("proof generation", proof_generation),
            ("source id", source_id),
            ("owner id", owner_id),
        ] {
            if value == 0 {
                return Err(RegisterFamilyModelError::ZeroIdentity { field });
            }
        }
        Ok(Self {
            proof_generation,
            source_kind,
            source_id,
            owner_id,
        })
    }

    pub const fn proof_generation(self) -> u64 {
        self.proof_generation
    }

    pub const fn source_kind(self) -> RegisterOwnerSourceKind {
        self.source_kind
    }

    pub const fn source_id(self) -> u64 {
        self.source_id
    }

    pub const fn owner_id(self) -> u64 {
        self.owner_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisterFamilyGeometry {
    cycles: usize,
    log_t: usize,
    blocks: usize,
    prefix_bits: usize,
    suffix_bits: usize,
    prefix_elements: usize,
    suffix_elements: usize,
    events: RegisterEventCounts,
}

impl RegisterFamilyGeometry {
    fn from_owner(owner: &CertifiedRegisterOwner) -> Result<Self, RegisterFamilyModelError> {
        let cycles = owner.csr().cycles();
        if cycles < 2 || !cycles.is_power_of_two() {
            return Err(RegisterFamilyModelError::InvalidCycleCount { got: cycles });
        }
        if owner.state_flow().cycles() != cycles {
            return Err(RegisterFamilyModelError::CertificateCycleMismatch {
                csr: cycles,
                certificate: owner.state_flow().cycles(),
            });
        }
        let log_t = cycles.ilog2() as usize;
        let suffix_bits = log_t / 2;
        let prefix_bits = log_t - suffix_bits;
        Ok(Self {
            cycles,
            log_t,
            blocks: owner.csr().block_count(),
            prefix_bits,
            suffix_bits,
            prefix_elements: 1usize << prefix_bits,
            suffix_elements: 1usize << suffix_bits,
            events: owner.csr().event_counts(),
        })
    }

    pub const fn cycles(self) -> usize {
        self.cycles
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn blocks(self) -> usize {
        self.blocks
    }

    pub const fn prefix_bits(self) -> usize {
        self.prefix_bits
    }

    pub const fn suffix_bits(self) -> usize {
        self.suffix_bits
    }

    pub const fn prefix_elements(self) -> usize {
        self.prefix_elements
    }

    pub const fn suffix_elements(self) -> usize {
        self.suffix_elements
    }

    pub const fn events(self) -> RegisterEventCounts {
        self.events
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisterOwnerStorage {
    pub full_owner_bytes: u128,
    pub claim_midpoint_bytes: u128,
    pub value_owner_bytes: u128,
}

impl RegisterOwnerStorage {
    fn from_owner(owner: &CertifiedRegisterOwner) -> Self {
        let parts = owner.csr().parts();
        let rd_offsets = slice_bytes(&parts.rd_offsets);
        let rd_positions = slice_bytes(&parts.rd_positions);
        let rd_post_values = slice_bytes(&parts.rd_post_values);
        let claim_midpoint_bytes = rd_offsets + rd_positions + rd_post_values;
        Self {
            full_owner_bytes: owner.csr().storage_bytes(),
            claim_midpoint_bytes,
            value_owner_bytes: slice_bytes(&parts.start_values) + claim_midpoint_bytes,
        }
    }
}

/// Shared owner handle. Cloning the carrier clones only the `Arc` and identity.
#[derive(Clone, Debug)]
pub struct RegisterFamilyCarrier {
    identity: RegisterOwnerIdentity,
    geometry: RegisterFamilyGeometry,
    storage: RegisterOwnerStorage,
    owner: Arc<CertifiedRegisterOwner>,
}

impl RegisterFamilyCarrier {
    pub fn new(
        identity: RegisterOwnerIdentity,
        owner: Arc<CertifiedRegisterOwner>,
    ) -> Result<Self, RegisterFamilyModelError> {
        let geometry = RegisterFamilyGeometry::from_owner(&owner)?;
        let storage = RegisterOwnerStorage::from_owner(&owner);
        Ok(Self {
            identity,
            geometry,
            storage,
            owner,
        })
    }

    pub const fn identity(&self) -> RegisterOwnerIdentity {
        self.identity
    }

    pub const fn geometry(&self) -> RegisterFamilyGeometry {
        self.geometry
    }

    pub const fn storage(&self) -> RegisterOwnerStorage {
        self.storage
    }

    pub fn owner(&self) -> &CertifiedRegisterOwner {
        &self.owner
    }

    pub fn shared_owner(&self) -> Arc<CertifiedRegisterOwner> {
        Arc::clone(&self.owner)
    }
}

/// Challenge-independent Q components produced before stage 3 draws `gamma`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegisterClaimComponents<F: Field> {
    owner_identity: RegisterOwnerIdentity,
    geometry: RegisterFamilyGeometry,
    tau: Vec<F>,
    components: [Vec<F>; 3],
}

impl<F: Field> RegisterClaimComponents<F> {
    pub fn new(
        owner: &RegisterFamilyCarrier,
        tau: Vec<F>,
        components: [Vec<F>; 3],
    ) -> Result<Self, RegisterFamilyModelError> {
        let geometry = owner.geometry();
        if tau.len() != geometry.log_t() {
            return Err(RegisterFamilyModelError::PointLength {
                point: "claim tau",
                expected: geometry.log_t(),
                got: tau.len(),
            });
        }
        for (column, values) in components.iter().enumerate() {
            if values.len() != geometry.prefix_elements() {
                return Err(RegisterFamilyModelError::ClaimComponentLength {
                    column,
                    expected: geometry.prefix_elements(),
                    got: values.len(),
                });
            }
        }
        Ok(Self {
            owner_identity: owner.identity(),
            geometry,
            tau,
            components,
        })
    }

    pub const fn owner_identity(&self) -> RegisterOwnerIdentity {
        self.owner_identity
    }

    pub const fn geometry(&self) -> RegisterFamilyGeometry {
        self.geometry
    }

    pub fn tau(&self) -> &[F] {
        &self.tau
    }

    pub fn components(&self) -> &[Vec<F>; 3] {
        &self.components
    }

    /// Gamma is accepted here, not by the producer, so the carrier cannot cross
    /// the stage-3 challenge boundary with a pre-combined table.
    pub fn combined_q(&self, gamma: F) -> Vec<F> {
        let gamma_sq = gamma * gamma;
        (0..self.geometry.prefix_elements())
            .map(|index| {
                self.components[0][index]
                    + gamma * self.components[1][index]
                    + gamma_sq * self.components[2][index]
            })
            .collect()
    }

    pub fn validate_owner(
        &self,
        owner: &RegisterFamilyCarrier,
    ) -> Result<(), RegisterFamilyModelError> {
        if self.owner_identity != owner.identity() {
            return Err(RegisterFamilyModelError::OwnerIdentityMismatch);
        }
        if self.geometry != owner.geometry() {
            return Err(RegisterFamilyModelError::OwnerGeometryMismatch);
        }
        Ok(())
    }

    pub fn opening_point(&self, challenges: &[F]) -> Result<Vec<F>, RegisterFamilyModelError> {
        if challenges.len() != self.geometry.log_t() {
            return Err(RegisterFamilyModelError::PointLength {
                point: "claim challenges",
                expected: self.geometry.log_t(),
                got: challenges.len(),
            });
        }
        Ok(challenges.iter().rev().copied().collect())
    }
}

/// Stage-4 register point consumed by value evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegisterValuePoint<F: Field> {
    owner_identity: RegisterOwnerIdentity,
    geometry: RegisterFamilyGeometry,
    address: [F; REGISTER_ADDRESS_BITS],
    cycle: Vec<F>,
}

impl<F: Field> RegisterValuePoint<F> {
    pub fn new(
        owner: &RegisterFamilyCarrier,
        address: &[F],
        cycle: Vec<F>,
    ) -> Result<Self, RegisterFamilyModelError> {
        let geometry = owner.geometry();
        let address: [F; REGISTER_ADDRESS_BITS] =
            address
                .try_into()
                .map_err(|_| RegisterFamilyModelError::PointLength {
                    point: "register address",
                    expected: REGISTER_ADDRESS_BITS,
                    got: address.len(),
                })?;
        if cycle.len() != geometry.log_t() {
            return Err(RegisterFamilyModelError::PointLength {
                point: "register cycle",
                expected: geometry.log_t(),
                got: cycle.len(),
            });
        }
        Ok(Self {
            owner_identity: owner.identity(),
            geometry,
            address,
            cycle,
        })
    }

    pub const fn owner_identity(&self) -> RegisterOwnerIdentity {
        self.owner_identity
    }

    pub const fn geometry(&self) -> RegisterFamilyGeometry {
        self.geometry
    }

    pub const fn address(&self) -> &[F; REGISTER_ADDRESS_BITS] {
        &self.address
    }

    pub fn cycle(&self) -> &[F] {
        &self.cycle
    }

    pub fn validate_owner(
        &self,
        owner: &RegisterFamilyCarrier,
    ) -> Result<(), RegisterFamilyModelError> {
        if self.owner_identity != owner.identity() {
            return Err(RegisterFamilyModelError::OwnerIdentityMismatch);
        }
        if self.geometry != owner.geometry() {
            return Err(RegisterFamilyModelError::OwnerGeometryMismatch);
        }
        Ok(())
    }

    pub fn output_point(&self, challenges: &[F]) -> Result<Vec<F>, RegisterFamilyModelError> {
        if challenges.len() != self.geometry.log_t() {
            return Err(RegisterFamilyModelError::PointLength {
                point: "value challenges",
                expected: self.geometry.log_t(),
                got: challenges.len(),
            });
        }
        Ok(self
            .address
            .iter()
            .copied()
            .chain(challenges.iter().rev().copied())
            .collect())
    }
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum RegisterFamilyModelError {
    #[error("register family identity field {field} must be nonzero")]
    ZeroIdentity { field: &'static str },
    #[error("register family cycle count {got} must be a power of two greater than one")]
    InvalidCycleCount { got: usize },
    #[error("register CSR has {csr} cycles but its certificate has {certificate}")]
    CertificateCycleMismatch { csr: usize, certificate: usize },
    #[error("{point} has {got} coordinates, expected {expected}")]
    PointLength {
        point: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("claim component {column} has {got} entries, expected {expected}")]
    ClaimComponentLength {
        column: usize,
        expected: usize,
        got: usize,
    },
    #[error("register claim/value carrier names a different owner")]
    OwnerIdentityMismatch,
    #[error("register claim/value carrier has different owner geometry")]
    OwnerGeometryMismatch,
}

fn slice_bytes<T>(values: &[T]) -> u128 {
    values.len() as u128 * size_of::<T>() as u128
}

const _: () = assert!(REGISTER_CSR_COLUMNS == 1 << REGISTER_ADDRESS_BITS);
