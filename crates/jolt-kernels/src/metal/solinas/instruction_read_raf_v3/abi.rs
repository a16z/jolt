use core::{marker::PhantomData, mem::size_of, num::NonZeroU64, num::NonZeroUsize};

use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};

use super::{
    InstructionReadRafV3Error, ADDRESS_BINS, ADDRESS_BITS, ADDRESS_PHASES, FP128_BYTES,
    INSTRUCTION_ROW_BYTES, PRODUCTION_VIRTUAL_RA,
};

pub(crate) const LOOKUP_TABLES: usize = LookupTableKind::<RISCV_XLEN>::COUNT;
pub(crate) const ADDRESS_SEGMENTS: usize = 2 * (LOOKUP_TABLES + 1);
pub(crate) const ADDRESS_SEGMENT_OFFSETS: usize = ADDRESS_SEGMENTS + 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InstructionReadRafGeometry {
    cycles: usize,
    log_t: usize,
    virtual_ra: usize,
    phases_per_ra: usize,
}

impl InstructionReadRafGeometry {
    pub(crate) fn new(cycles: usize, virtual_ra: usize) -> Result<Self, InstructionReadRafV3Error> {
        if cycles == 0 || !cycles.is_power_of_two() || cycles > u32::MAX as usize {
            return Err(InstructionReadRafV3Error::InvalidCycles(cycles));
        }
        if virtual_ra != PRODUCTION_VIRTUAL_RA {
            return Err(InstructionReadRafV3Error::InvalidVirtualRa(virtual_ra));
        }
        Ok(Self {
            cycles,
            log_t: cycles.trailing_zeros() as usize,
            virtual_ra,
            phases_per_ra: ADDRESS_PHASES / virtual_ra,
        })
    }

    pub(crate) const fn cycles(self) -> usize {
        self.cycles
    }

    pub(crate) const fn log_t(self) -> usize {
        self.log_t
    }

    pub(crate) const fn virtual_ra(self) -> usize {
        self.virtual_ra
    }

    pub(crate) const fn cycle_factors(self) -> usize {
        self.virtual_ra + 1
    }

    pub(crate) const fn phases_per_ra(self) -> usize {
        self.phases_per_ra
    }

    pub(crate) const fn member_rounds(self) -> usize {
        ADDRESS_BITS + self.log_t
    }
}

/// Provenance shared by every borrowed plane from one witness extraction.
///
/// `completion_serial` is the command serial after which the source allocation
/// is initialized.  A member-local upload cannot manufacture this identity;
/// the stage-1/2 resident producer must publish it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProducerIdentity {
    device_registry_id: NonZeroU64,
    source_allocation_identity: NonZeroUsize,
    source_generation: NonZeroU64,
    completion_serial: NonZeroU64,
    geometry: InstructionReadRafGeometry,
}

impl ProducerIdentity {
    pub(crate) fn new(
        device_registry_id: u64,
        source_allocation_identity: usize,
        source_generation: u64,
        completion_serial: u64,
        geometry: InstructionReadRafGeometry,
    ) -> Result<Self, InstructionReadRafV3Error> {
        Ok(Self {
            device_registry_id: nonzero_u64("device registry", device_registry_id)?,
            source_allocation_identity: nonzero_usize(
                "source allocation",
                source_allocation_identity,
            )?,
            source_generation: nonzero_u64("source generation", source_generation)?,
            completion_serial: nonzero_u64("source completion serial", completion_serial)?,
            geometry,
        })
    }

    pub(crate) const fn device_registry_id(self) -> u64 {
        self.device_registry_id.get()
    }

    pub(crate) const fn source_allocation_identity(self) -> usize {
        self.source_allocation_identity.get()
    }

    pub(crate) const fn source_generation(self) -> u64 {
        self.source_generation.get()
    }

    pub(crate) const fn completion_serial(self) -> u64 {
        self.completion_serial.get()
    }

    pub(crate) const fn geometry(self) -> InstructionReadRafGeometry {
        self.geometry
    }
}

/// Runtime metadata for one Metal allocation.  Constructors below turn this
/// untyped descriptor into relation-specific planes only after checking shape,
/// producer generation, and command completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlaneDescriptor {
    producer: ProducerIdentity,
    allocation_identity: NonZeroUsize,
    initialized_generation: NonZeroU64,
    completion_serial: NonZeroU64,
    elements: usize,
    bytes: usize,
}

impl PlaneDescriptor {
    pub(crate) fn new(
        producer: ProducerIdentity,
        allocation_identity: usize,
        initialized_generation: u64,
        completion_serial: u64,
        elements: usize,
        bytes: usize,
        plane: &'static str,
    ) -> Result<Self, InstructionReadRafV3Error> {
        let allocation_identity = nonzero_usize(plane, allocation_identity)?;
        let initialized_generation = nonzero_u64(plane, initialized_generation)?;
        let completion_serial = nonzero_u64(plane, completion_serial)?;
        if initialized_generation.get() != producer.source_generation() {
            return Err(InstructionReadRafV3Error::GenerationMismatch {
                plane,
                expected: producer.source_generation(),
                got: initialized_generation.get(),
            });
        }
        if completion_serial.get() < producer.completion_serial() {
            return Err(InstructionReadRafV3Error::IncompletePlane {
                plane,
                minimum: producer.completion_serial(),
                got: completion_serial.get(),
            });
        }
        Ok(Self {
            producer,
            allocation_identity,
            initialized_generation,
            completion_serial,
            elements,
            bytes,
        })
    }

    pub(crate) fn for_elements<T>(
        producer: ProducerIdentity,
        allocation_identity: usize,
        completion_serial: u64,
        elements: usize,
        plane: &'static str,
    ) -> Result<Self, InstructionReadRafV3Error> {
        let bytes = elements
            .checked_mul(size_of::<T>())
            .ok_or(InstructionReadRafV3Error::SizeOverflow(plane))?;
        Self::new(
            producer,
            allocation_identity,
            producer.source_generation(),
            completion_serial,
            elements,
            bytes,
            plane,
        )
    }

    pub(crate) const fn producer(self) -> ProducerIdentity {
        self.producer
    }

    pub(crate) const fn allocation_identity(self) -> usize {
        self.allocation_identity.get()
    }

    pub(crate) const fn initialized_generation(self) -> u64 {
        self.initialized_generation.get()
    }

    pub(crate) const fn completion_serial(self) -> u64 {
        self.completion_serial.get()
    }

    pub(crate) const fn elements(self) -> usize {
        self.elements
    }

    pub(crate) const fn bytes(self) -> usize {
        self.bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResidentPlane<Tag> {
    descriptor: PlaneDescriptor,
    _tag: PhantomData<Tag>,
}

impl<Tag> ResidentPlane<Tag> {
    fn checked(
        descriptor: PlaneDescriptor,
        expected_elements: usize,
        expected_bytes: usize,
        plane: &'static str,
    ) -> Result<Self, InstructionReadRafV3Error> {
        if descriptor.elements != expected_elements {
            return Err(InstructionReadRafV3Error::PlaneElements {
                plane,
                expected: expected_elements,
                got: descriptor.elements,
            });
        }
        if descriptor.bytes != expected_bytes {
            return Err(InstructionReadRafV3Error::PlaneBytes {
                plane,
                expected: expected_bytes,
                got: descriptor.bytes,
            });
        }
        Ok(Self {
            descriptor,
            _tag: PhantomData,
        })
    }

    pub(crate) const fn descriptor(self) -> PlaneDescriptor {
        self.descriptor
    }

    pub(crate) const fn allocation_identity(self) -> usize {
        self.descriptor.allocation_identity()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InstructionRows {}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ClaimBytes {}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EqIn {}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EqOut {}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AtomLookups {}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AtomClaims {}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AtomOffsets {}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AtomCycles {}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SegmentOffsets {}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MutableMasses {}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PhaseTables {}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CycleFactors {}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OutputClaims {}

/// The exact producer-owned facts used by stage 5 and later instruction
/// stages.  The 40-byte row is the existing Akita `InstructionCycleRow`: raw
/// lookup limbs occupy words 0/1 and table/RAF selectors are decoded from word
/// 4.  The one-byte claim plane is co-produced for the terminal flag opening.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResidentInstructionFacts {
    producer: ProducerIdentity,
    rows: ResidentPlane<InstructionRows>,
    claims: ResidentPlane<ClaimBytes>,
}

impl ResidentInstructionFacts {
    pub(crate) fn new(
        producer: ProducerIdentity,
        rows: PlaneDescriptor,
        claims: PlaneDescriptor,
    ) -> Result<Self, InstructionReadRafV3Error> {
        same_producer(producer, rows, "instruction rows")?;
        same_producer(producer, claims, "claim bytes")?;
        if rows.allocation_identity() != producer.source_allocation_identity() {
            return Err(InstructionReadRafV3Error::SourceAllocationMismatch {
                expected: producer.source_allocation_identity(),
                got: rows.allocation_identity(),
            });
        }
        let cycles = producer.geometry().cycles();
        let rows = ResidentPlane::checked(
            rows,
            cycles,
            checked_bytes(cycles, INSTRUCTION_ROW_BYTES, "instruction row bytes")?,
            "instruction rows",
        )?;
        let claims = ResidentPlane::checked(claims, cycles, cycles, "claim bytes")?;
        reject_aliases(&[rows.allocation_identity(), claims.allocation_identity()])?;
        Ok(Self {
            producer,
            rows,
            claims,
        })
    }

    pub(crate) const fn producer(self) -> ProducerIdentity {
        self.producer
    }

    pub(crate) const fn rows(self) -> ResidentPlane<InstructionRows> {
        self.rows
    }

    pub(crate) const fn claims(self) -> ResidentPlane<ClaimBytes> {
        self.claims
    }
}

/// Split equality factors for `eq(r_reduction, cycle)`.  These are borrowed
/// from the upstream reduction owner or generated once on device; no dense
/// `T`-field equality table is part of this ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReductionEqReceipt {
    producer: ProducerIdentity,
    e_in: ResidentPlane<EqIn>,
    e_out: ResidentPlane<EqOut>,
}

impl ReductionEqReceipt {
    pub(crate) fn new(
        producer: ProducerIdentity,
        e_in: PlaneDescriptor,
        e_out: PlaneDescriptor,
    ) -> Result<Self, InstructionReadRafV3Error> {
        same_producer(producer, e_in, "reduction E_in")?;
        same_producer(producer, e_out, "reduction E_out")?;
        let log_t = producer.geometry().log_t();
        let out_len = 1usize << (log_t / 2);
        let in_len = 1usize << (log_t - log_t / 2);
        let e_in = ResidentPlane::checked(
            e_in,
            in_len,
            checked_bytes(in_len, FP128_BYTES, "E_in bytes")?,
            "reduction E_in",
        )?;
        let e_out = ResidentPlane::checked(
            e_out,
            out_len,
            checked_bytes(out_len, FP128_BYTES, "E_out bytes")?,
            "reduction E_out",
        )?;
        reject_aliases(&[e_in.allocation_identity(), e_out.allocation_identity()])?;
        Ok(Self {
            producer,
            e_in,
            e_out,
        })
    }

    pub(crate) const fn e_in(self) -> ResidentPlane<EqIn> {
        self.e_in
    }

    pub(crate) const fn e_out(self) -> ResidentPlane<EqOut> {
        self.e_out
    }

    pub(crate) const fn producer(self) -> ProducerIdentity {
        self.producer
    }
}

/// Complete checked input lease for one InstructionReadRaf invocation.
///
/// The source facts remain borrowed by instruction RA virtualization and the
/// one-hot booleanity stages.  This lease neither consumes nor republishes
/// them, which makes those adjacent-stage fusions safe without extending any
/// Fiat--Shamir boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResidentReadRafInputs {
    facts: ResidentInstructionFacts,
    reduction_eq: ReductionEqReceipt,
    atoms: Option<AddressAtomTopologyReceipt>,
}

impl ResidentReadRafInputs {
    pub(crate) fn new(
        facts: &ResidentInstructionFacts,
        reduction_eq: &ReductionEqReceipt,
        atoms: Option<&AddressAtomTopologyReceipt>,
    ) -> Result<Self, InstructionReadRafV3Error> {
        let producer = facts.producer();
        if reduction_eq.producer() != producer {
            return Err(InstructionReadRafV3Error::ProducerMismatch {
                plane: "reduction equality factors",
            });
        }
        if atoms.is_some_and(|topology| topology.producer() != producer) {
            return Err(InstructionReadRafV3Error::ProducerMismatch {
                plane: "address atom topology",
            });
        }
        let mut identities = vec![
            facts.rows().allocation_identity(),
            facts.claims().allocation_identity(),
            reduction_eq.e_in().allocation_identity(),
            reduction_eq.e_out().allocation_identity(),
        ];
        if let Some(topology) = atoms {
            identities.extend(topology.allocation_identities());
        }
        reject_aliases(&identities)?;
        Ok(Self {
            facts: *facts,
            reduction_eq: *reduction_eq,
            atoms: atoms.copied(),
        })
    }

    pub(crate) const fn facts(self) -> ResidentInstructionFacts {
        self.facts
    }

    pub(crate) const fn reduction_eq(self) -> ReductionEqReceipt {
        self.reduction_eq
    }

    pub(crate) const fn atoms(self) -> Option<AddressAtomTopologyReceipt> {
        self.atoms
    }

    pub(crate) const fn uses_atom_path(self) -> bool {
        self.atoms.is_some()
    }
}

/// Optional producer-owned CSR over exact `(table, RAF, raw u128 lookup)`
/// atoms.  Raw lookup bits are mandatory: reducing atom keys into fp128 would
/// merge non-canonical aliases before the gamma-cubed guard can reject them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AddressAtomTopologyReceipt {
    producer: ProducerIdentity,
    atoms: usize,
    lookups: ResidentPlane<AtomLookups>,
    claims: ResidentPlane<AtomClaims>,
    offsets: ResidentPlane<AtomOffsets>,
    cycles: ResidentPlane<AtomCycles>,
    segments: ResidentPlane<SegmentOffsets>,
    producer_coowned: bool,
}

impl AddressAtomTopologyReceipt {
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn new(
        producer: ProducerIdentity,
        atoms: usize,
        lookups: PlaneDescriptor,
        claims: PlaneDescriptor,
        offsets: PlaneDescriptor,
        cycles: PlaneDescriptor,
        segments: PlaneDescriptor,
        producer_coowned: bool,
    ) -> Result<Self, InstructionReadRafV3Error> {
        let rows = producer.geometry().cycles();
        if atoms == 0 || atoms > rows {
            return Err(InstructionReadRafV3Error::InvalidAtomCount { rows, atoms });
        }
        for (plane, descriptor) in [
            ("atom lookups", lookups),
            ("atom claims", claims),
            ("atom offsets", offsets),
            ("atom cycles", cycles),
            ("segment offsets", segments),
        ] {
            same_producer(producer, descriptor, plane)?;
        }
        let lookups = ResidentPlane::checked(
            lookups,
            atoms,
            checked_bytes(atoms, size_of::<u128>(), "atom lookup bytes")?,
            "atom lookups",
        )?;
        let claims = ResidentPlane::checked(claims, atoms, atoms, "atom claims")?;
        let offsets_len = atoms
            .checked_add(1)
            .ok_or(InstructionReadRafV3Error::SizeOverflow("atom offsets"))?;
        let offsets = ResidentPlane::checked(
            offsets,
            offsets_len,
            checked_bytes(offsets_len, size_of::<u32>(), "atom offset bytes")?,
            "atom offsets",
        )?;
        let cycles = ResidentPlane::checked(
            cycles,
            rows,
            checked_bytes(rows, size_of::<u32>(), "atom cycle bytes")?,
            "atom cycles",
        )?;
        let segments = ResidentPlane::checked(
            segments,
            ADDRESS_SEGMENT_OFFSETS,
            checked_bytes(
                ADDRESS_SEGMENT_OFFSETS,
                size_of::<u32>(),
                "segment offset bytes",
            )?,
            "segment offsets",
        )?;
        reject_aliases(&[
            lookups.allocation_identity(),
            claims.allocation_identity(),
            offsets.allocation_identity(),
            cycles.allocation_identity(),
            segments.allocation_identity(),
        ])?;
        Ok(Self {
            producer,
            atoms,
            lookups,
            claims,
            offsets,
            cycles,
            segments,
            producer_coowned,
        })
    }

    pub(crate) const fn producer(self) -> ProducerIdentity {
        self.producer
    }

    pub(crate) const fn atoms(self) -> usize {
        self.atoms
    }

    pub(crate) const fn producer_coowned(self) -> bool {
        self.producer_coowned
    }

    pub(crate) fn allocation_identities(self) -> [usize; 5] {
        [
            self.lookups.allocation_identity(),
            self.claims.allocation_identity(),
            self.offsets.allocation_identity(),
            self.cycles.allocation_identity(),
            self.segments.allocation_identity(),
        ]
    }
}

/// Device-resident address owner after a completed 8-variable phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AddressStateReceipt {
    producer: ProducerIdentity,
    domain_rows: usize,
    completed_phases: usize,
    masses: ResidentPlane<MutableMasses>,
    phase_tables: ResidentPlane<PhaseTables>,
    host_challenge_digest: u64,
}

impl AddressStateReceipt {
    pub(crate) fn new(
        producer: ProducerIdentity,
        domain_rows: usize,
        completed_phases: usize,
        masses: PlaneDescriptor,
        phase_tables: PlaneDescriptor,
        host_challenge_digest: u64,
    ) -> Result<Self, InstructionReadRafV3Error> {
        if domain_rows == 0 || domain_rows > producer.geometry().cycles() {
            return Err(InstructionReadRafV3Error::InvalidAtomCount {
                rows: producer.geometry().cycles(),
                atoms: domain_rows,
            });
        }
        if completed_phases == 0 || completed_phases > ADDRESS_PHASES {
            return Err(InstructionReadRafV3Error::PhaseMismatch {
                expected: ADDRESS_PHASES,
                got: completed_phases,
            });
        }
        if host_challenge_digest == 0 {
            return Err(InstructionReadRafV3Error::MissingIdentity {
                name: "host address-challenge digest",
            });
        }
        same_producer(producer, masses, "address masses")?;
        same_producer(producer, phase_tables, "address phase tables")?;
        let masses = ResidentPlane::checked(
            masses,
            domain_rows,
            checked_bytes(domain_rows, FP128_BYTES, "address mass bytes")?,
            "address masses",
        )?;
        let table_elements = ADDRESS_PHASES * ADDRESS_BINS;
        let phase_tables = ResidentPlane::checked(
            phase_tables,
            table_elements,
            checked_bytes(table_elements, FP128_BYTES, "phase table bytes")?,
            "address phase tables",
        )?;
        reject_aliases(&[
            masses.allocation_identity(),
            phase_tables.allocation_identity(),
        ])?;
        Ok(Self {
            producer,
            domain_rows,
            completed_phases,
            masses,
            phase_tables,
            host_challenge_digest,
        })
    }

    pub(crate) const fn completed_phases(self) -> usize {
        self.completed_phases
    }

    pub(crate) const fn domain_rows(self) -> usize {
        self.domain_rows
    }
}

/// Five-factor production cycle arena after at least one host challenge has
/// bound the full-width source directly into the half domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CycleFactorReceipt {
    producer: ProducerIdentity,
    width: usize,
    bound_cycle_rounds: usize,
    factors: ResidentPlane<CycleFactors>,
    host_challenge_digest: u64,
}

impl CycleFactorReceipt {
    pub(crate) fn new(
        producer: ProducerIdentity,
        width: usize,
        bound_cycle_rounds: usize,
        factors: PlaneDescriptor,
        host_challenge_digest: u64,
    ) -> Result<Self, InstructionReadRafV3Error> {
        let geometry = producer.geometry();
        if bound_cycle_rounds == 0 || bound_cycle_rounds >= geometry.log_t() {
            return Err(InstructionReadRafV3Error::PhaseMismatch {
                expected: geometry.log_t(),
                got: bound_cycle_rounds,
            });
        }
        let expected_width = geometry.cycles() >> bound_cycle_rounds;
        if width != expected_width {
            return Err(InstructionReadRafV3Error::WidthMismatch {
                expected: expected_width,
                got: width,
            });
        }
        if host_challenge_digest == 0 {
            return Err(InstructionReadRafV3Error::MissingIdentity {
                name: "host cycle-challenge digest",
            });
        }
        same_producer(producer, factors, "cycle factors")?;
        let elements = geometry.cycle_factors().checked_mul(width).ok_or(
            InstructionReadRafV3Error::SizeOverflow("cycle factor elements"),
        )?;
        let factors = ResidentPlane::checked(
            factors,
            elements,
            checked_bytes(elements, FP128_BYTES, "cycle factor bytes")?,
            "cycle factors",
        )?;
        Ok(Self {
            producer,
            width,
            bound_cycle_rounds,
            factors,
            host_challenge_digest,
        })
    }

    pub(crate) const fn width(self) -> usize {
        self.width
    }

    pub(crate) const fn bound_cycle_rounds(self) -> usize {
        self.bound_cycle_rounds
    }
}

/// Terminal 40 table flags, virtual-RA values, and one RAF flag.  The receipt
/// can only be constructed after the host supplies the final cycle challenge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StageOutputReceipt {
    producer: ProducerIdentity,
    claims: ResidentPlane<OutputClaims>,
    bound_cycle_rounds: usize,
    host_challenge_digest: u64,
}

impl StageOutputReceipt {
    pub(crate) fn new(
        producer: ProducerIdentity,
        claims: PlaneDescriptor,
        bound_cycle_rounds: usize,
        host_challenge_digest: u64,
    ) -> Result<Self, InstructionReadRafV3Error> {
        let geometry = producer.geometry();
        if bound_cycle_rounds != geometry.log_t() {
            return Err(InstructionReadRafV3Error::PhaseMismatch {
                expected: geometry.log_t(),
                got: bound_cycle_rounds,
            });
        }
        if host_challenge_digest == 0 {
            return Err(InstructionReadRafV3Error::MissingIdentity {
                name: "terminal host-challenge digest",
            });
        }
        same_producer(producer, claims, "stage output claims")?;
        let output_count = LOOKUP_TABLES + geometry.virtual_ra() + 1;
        let claims = ResidentPlane::checked(
            claims,
            output_count,
            checked_bytes(output_count, FP128_BYTES, "output claim bytes")?,
            "stage output claims",
        )?;
        Ok(Self {
            producer,
            claims,
            bound_cycle_rounds,
            host_challenge_digest,
        })
    }

    pub(crate) const fn output_count(self) -> usize {
        self.claims.descriptor().elements()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PriorBind {
    None,
    Address(usize),
    Cycle(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MemberMessage {
    Address(usize),
    Cycle(usize),
}

/// One host transcript boundary.  Device fusion may span arithmetic inside a
/// boundary, never the `message -> absorb -> challenge -> bind` edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HostRoundBoundary {
    member_call: usize,
    prior_bind: PriorBind,
    message: MemberMessage,
    starts_address_phase: Option<usize>,
    crosses_address_cycle_handoff: bool,
}

impl HostRoundBoundary {
    pub(crate) fn at(
        geometry: InstructionReadRafGeometry,
        member_call: usize,
    ) -> Result<Self, InstructionReadRafV3Error> {
        if member_call >= geometry.member_rounds() {
            return Err(InstructionReadRafV3Error::InvalidMemberCall(member_call));
        }
        if member_call < ADDRESS_BITS {
            return Ok(Self {
                member_call,
                prior_bind: member_call
                    .checked_sub(1)
                    .map_or(PriorBind::None, PriorBind::Address),
                message: MemberMessage::Address(member_call),
                starts_address_phase: member_call
                    .is_multiple_of(super::ADDRESS_PHASE_BITS)
                    .then_some(member_call / super::ADDRESS_PHASE_BITS),
                crosses_address_cycle_handoff: false,
            });
        }
        let cycle_round = member_call - ADDRESS_BITS;
        Ok(Self {
            member_call,
            prior_bind: if cycle_round == 0 {
                PriorBind::Address(ADDRESS_BITS - 1)
            } else {
                PriorBind::Cycle(cycle_round - 1)
            },
            message: MemberMessage::Cycle(cycle_round),
            starts_address_phase: None,
            crosses_address_cycle_handoff: cycle_round == 0,
        })
    }

    pub(crate) const fn prior_bind(self) -> PriorBind {
        self.prior_bind
    }

    pub(crate) const fn message(self) -> MemberMessage {
        self.message
    }

    pub(crate) const fn starts_address_phase(self) -> Option<usize> {
        self.starts_address_phase
    }

    pub(crate) const fn crosses_address_cycle_handoff(self) -> bool {
        self.crosses_address_cycle_handoff
    }
}

fn nonzero_u64(name: &'static str, value: u64) -> Result<NonZeroU64, InstructionReadRafV3Error> {
    NonZeroU64::new(value).ok_or(InstructionReadRafV3Error::MissingIdentity { name })
}

fn nonzero_usize(
    name: &'static str,
    value: usize,
) -> Result<NonZeroUsize, InstructionReadRafV3Error> {
    NonZeroUsize::new(value).ok_or(InstructionReadRafV3Error::MissingIdentity { name })
}

fn same_producer(
    expected: ProducerIdentity,
    descriptor: PlaneDescriptor,
    plane: &'static str,
) -> Result<(), InstructionReadRafV3Error> {
    if descriptor.producer != expected {
        return Err(InstructionReadRafV3Error::ProducerMismatch { plane });
    }
    Ok(())
}

fn checked_bytes(
    elements: usize,
    element_bytes: usize,
    name: &'static str,
) -> Result<usize, InstructionReadRafV3Error> {
    elements
        .checked_mul(element_bytes)
        .ok_or(InstructionReadRafV3Error::SizeOverflow(name))
}

fn reject_aliases(identities: &[usize]) -> Result<(), InstructionReadRafV3Error> {
    for (index, identity) in identities.iter().copied().enumerate() {
        if identities[..index].contains(&identity) {
            return Err(InstructionReadRafV3Error::AliasedAllocation { identity });
        }
    }
    Ok(())
}
