use crate::optimized::ram_trace::RamIncrementActivity;
use crate::ram_access::{RamAccessRecord, RamAccessTape};

use super::RamFamilyV2Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SparseRamSource {
    identity: usize,
    generation: u64,
}

impl SparseRamSource {
    pub(crate) fn new(identity: usize, generation: u64) -> Result<Self, RamFamilyV2Error> {
        if identity == 0 {
            return Err(RamFamilyV2Error::ZeroSourceIdentity);
        }
        if generation == 0 {
            return Err(RamFamilyV2Error::ZeroSourceGeneration);
        }
        Ok(Self {
            identity,
            generation,
        })
    }

    pub(crate) const fn identity(self) -> usize {
        self.identity
    }

    pub(crate) const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HammingSupportCertificate {
    Exact,
    Uncertified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SparseRamCertificates {
    hamming_support: HammingSupportCertificate,
}

impl SparseRamCertificates {
    pub(crate) const fn new(hamming_support: HammingSupportCertificate) -> Self {
        Self { hamming_support }
    }

    pub(crate) const fn hamming_support(self) -> HammingSupportCertificate {
        self.hamming_support
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SparseRamAccess {
    cycle: u32,
    address: u32,
    pre_value: u64,
    post_value: u64,
}

impl SparseRamAccess {
    pub(crate) const fn new(cycle: u32, address: u32, pre_value: u64, post_value: u64) -> Self {
        Self {
            cycle,
            address,
            pre_value,
            post_value,
        }
    }

    pub(crate) const fn cycle(self) -> u32 {
        self.cycle
    }

    pub(crate) const fn address(self) -> u32 {
        self.address
    }

    pub(crate) const fn pre_value(self) -> u64 {
        self.pre_value
    }

    pub(crate) const fn post_value(self) -> u64 {
        self.post_value
    }
}

impl From<RamAccessRecord> for SparseRamAccess {
    fn from(record: RamAccessRecord) -> Self {
        Self::new(
            record.cycle,
            record.address,
            record.pre_value,
            record.post_value,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SparseRamIncrement {
    cycle: u32,
    value: i128,
}

impl SparseRamIncrement {
    pub(crate) const fn new(cycle: u32, value: i128) -> Self {
        Self { cycle, value }
    }

    pub(crate) const fn cycle(self) -> u32 {
        self.cycle
    }

    pub(crate) const fn value(self) -> i128 {
        self.value
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SparseCycleLeaf {
    cycle: u32,
    access_index: Option<u32>,
    increment_index: Option<u32>,
}

impl SparseCycleLeaf {
    pub(crate) const fn cycle(self) -> u32 {
        self.cycle
    }

    pub(crate) const fn access_index(self) -> Option<u32> {
        self.access_index
    }

    pub(crate) const fn increment_index(self) -> Option<u32> {
        self.increment_index
    }
}

/// One active block after `level` low cycle bits have been bound.
///
/// Child indices are relative to the preceding level. At level zero both are
/// absent and [`SparseCycleTopology::leaves`] carries the source-row mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SparseCycleNode {
    block: u32,
    even_child: Option<u32>,
    odd_child: Option<u32>,
}

impl SparseCycleNode {
    pub(crate) const fn block(self) -> u32 {
        self.block
    }

    pub(crate) const fn even_child(self) -> Option<u32> {
        self.even_child
    }

    pub(crate) const fn odd_child(self) -> Option<u32> {
        self.odd_child
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SparseCycleTopology {
    log_t: usize,
    leaves: Vec<SparseCycleLeaf>,
    nodes: Vec<SparseCycleNode>,
    level_offsets: Vec<usize>,
}

impl SparseCycleTopology {
    fn build(
        log_t: usize,
        accesses: &[SparseRamAccess],
        increments: &[SparseRamIncrement],
    ) -> Result<Self, RamFamilyV2Error> {
        let leaves = merge_source_cycles(accesses, increments)?;
        let mut nodes = Vec::with_capacity(leaves.len());
        let mut level_offsets = Vec::with_capacity(log_t + 2);
        level_offsets.push(0);
        nodes.extend(leaves.iter().map(|leaf| SparseCycleNode {
            block: leaf.cycle,
            even_child: None,
            odd_child: None,
        }));
        level_offsets.push(nodes.len());

        let mut previous_blocks = leaves.iter().map(|leaf| leaf.cycle).collect::<Vec<_>>();
        for _ in 1..=log_t {
            let mut next_blocks = Vec::with_capacity(previous_blocks.len().div_ceil(2));
            let mut next_nodes = Vec::with_capacity(previous_blocks.len().div_ceil(2));
            let mut child = 0;
            while child < previous_blocks.len() {
                let parent = previous_blocks[child] >> 1;
                let mut even_child = None;
                let mut odd_child = None;
                while child < previous_blocks.len() && previous_blocks[child] >> 1 == parent {
                    let relative = u32::try_from(child)
                        .map_err(|_| RamFamilyV2Error::TopologyIndexOverflow)?;
                    if previous_blocks[child] & 1 == 0 {
                        even_child = Some(relative);
                    } else {
                        odd_child = Some(relative);
                    }
                    child += 1;
                }
                next_blocks.push(parent);
                next_nodes.push(SparseCycleNode {
                    block: parent,
                    even_child,
                    odd_child,
                });
            }
            nodes.extend(next_nodes);
            level_offsets.push(nodes.len());
            previous_blocks = next_blocks;
        }

        Ok(Self {
            log_t,
            leaves,
            nodes,
            level_offsets,
        })
    }

    pub(crate) const fn log_t(&self) -> usize {
        self.log_t
    }

    pub(crate) fn leaves(&self) -> &[SparseCycleLeaf] {
        &self.leaves
    }

    pub(crate) fn level(&self, level: usize) -> Option<&[SparseCycleNode]> {
        (level <= self.log_t).then(|| {
            let start = self.level_offsets[level];
            let end = self.level_offsets[level + 1];
            &self.nodes[start..end]
        })
    }

    pub(crate) fn level_offsets(&self) -> &[usize] {
        &self.level_offsets
    }

    pub(crate) const fn total_nodes(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SparseRamProvenance {
    source: SparseRamSource,
    log_t: usize,
    address_domain: usize,
    access_count: usize,
    increment_count: usize,
    hamming_support: HammingSupportCertificate,
}

impl SparseRamProvenance {
    pub(crate) const fn source(self) -> SparseRamSource {
        self.source
    }

    pub(crate) const fn log_t(self) -> usize {
        self.log_t
    }

    pub(crate) const fn address_domain(self) -> usize {
        self.address_domain
    }

    pub(crate) const fn access_count(self) -> usize {
        self.access_count
    }

    pub(crate) const fn increment_count(self) -> usize {
        self.increment_count
    }

    pub(crate) const fn hamming_support(self) -> HammingSupportCertificate {
        self.hamming_support
    }
}

/// Immutable sparse source shared by the RAM-family kernels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SparseRamOwner {
    provenance: SparseRamProvenance,
    accesses: Vec<SparseRamAccess>,
    increments: Vec<SparseRamIncrement>,
    topology: SparseCycleTopology,
}

impl SparseRamOwner {
    pub(crate) fn new(
        log_t: usize,
        address_domain: usize,
        source: SparseRamSource,
        certificates: SparseRamCertificates,
        accesses: Vec<SparseRamAccess>,
        increments: Vec<SparseRamIncrement>,
    ) -> Result<Self, RamFamilyV2Error> {
        validate_geometry(log_t, address_domain)?;
        validate_accesses(log_t, address_domain, &accesses)?;
        validate_increments(log_t, &increments)?;
        validate_access_deltas(&accesses, &increments)?;
        let topology = SparseCycleTopology::build(log_t, &accesses, &increments)?;
        let provenance = SparseRamProvenance {
            source,
            log_t,
            address_domain,
            access_count: accesses.len(),
            increment_count: increments.len(),
            hamming_support: certificates.hamming_support(),
        };
        Ok(Self {
            provenance,
            accesses,
            increments,
            topology,
        })
    }

    /// Adapts the retained production streams without trusting the tape's
    /// increment-compatibility flag, which excludes valid address-zero stores.
    /// Accessed-row deltas are checked against the separate activity stream by
    /// [`Self::new`].
    pub(crate) fn from_retained_trace(
        tape: &RamAccessTape,
        activity: &RamIncrementActivity,
        log_t: usize,
        address_domain: usize,
        source: SparseRamSource,
        certificates: SparseRamCertificates,
    ) -> Result<Self, RamFamilyV2Error> {
        validate_geometry(log_t, address_domain)?;
        tape.validate(log_t, address_domain)
            .map_err(|_| RamFamilyV2Error::TapeRejected)?;
        let records = tape
            .records()
            .ok_or(RamFamilyV2Error::AccessRecordsUnavailable {
                access_count: tape.access_count(),
            })?;
        if records.len() != tape.access_count() {
            return Err(RamFamilyV2Error::AccessCountMismatch {
                expected: tape.access_count(),
                got: records.len(),
            });
        }
        let accesses = records.iter().copied().map(Into::into).collect();
        let increments = activity
            .records()
            .map(|(cycle, value)| {
                u32::try_from(cycle)
                    .map(|cycle| SparseRamIncrement::new(cycle, value))
                    .map_err(|_| RamFamilyV2Error::IncrementCycleEncodingOverflow { cycle })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(
            log_t,
            address_domain,
            source,
            certificates,
            accesses,
            increments,
        )
    }

    pub(crate) const fn provenance(&self) -> SparseRamProvenance {
        self.provenance
    }

    pub(crate) const fn log_t(&self) -> usize {
        self.provenance.log_t
    }

    pub(crate) const fn address_domain(&self) -> usize {
        self.provenance.address_domain
    }

    pub(crate) fn address_bits(&self) -> usize {
        self.address_domain().ilog2() as usize
    }

    pub(crate) const fn cycle_domain(&self) -> u64 {
        1u64 << self.provenance.log_t
    }

    pub(crate) fn accesses(&self) -> &[SparseRamAccess] {
        &self.accesses
    }

    pub(crate) fn increments(&self) -> &[SparseRamIncrement] {
        &self.increments
    }

    pub(crate) const fn topology(&self) -> &SparseCycleTopology {
        &self.topology
    }

    pub(crate) fn require_hamming_support(&self) -> Result<(), RamFamilyV2Error> {
        match self.provenance.hamming_support {
            HammingSupportCertificate::Exact => Ok(()),
            HammingSupportCertificate::Uncertified => {
                Err(RamFamilyV2Error::HammingSupportUncertified)
            }
        }
    }

    pub(crate) fn certified_hamming_accesses(
        &self,
    ) -> Result<&[SparseRamAccess], RamFamilyV2Error> {
        self.require_hamming_support()?;
        Ok(&self.accesses)
    }
}

fn validate_geometry(log_t: usize, address_domain: usize) -> Result<(), RamFamilyV2Error> {
    if log_t > u32::BITS as usize {
        return Err(RamFamilyV2Error::InvalidLogT { log_t });
    }
    if !address_domain.is_power_of_two() || address_domain > u32::MAX as usize {
        return Err(RamFamilyV2Error::InvalidAddressDomain { address_domain });
    }
    Ok(())
}

fn validate_accesses(
    log_t: usize,
    address_domain: usize,
    accesses: &[SparseRamAccess],
) -> Result<(), RamFamilyV2Error> {
    let cycle_domain = 1u64 << log_t;
    let mut previous = None;
    for access in accesses {
        if u64::from(access.cycle) >= cycle_domain {
            return Err(RamFamilyV2Error::AccessCycleOutOfRange {
                cycle: access.cycle,
                log_t,
            });
        }
        if previous.is_some_and(|cycle| cycle >= access.cycle) {
            return Err(RamFamilyV2Error::AccessesOutOfOrder {
                cycle: access.cycle,
            });
        }
        if access.address == u32::MAX || access.address as usize >= address_domain {
            return Err(RamFamilyV2Error::AddressOutOfRange {
                address: access.address,
                address_domain,
            });
        }
        previous = Some(access.cycle);
    }
    Ok(())
}

fn validate_increments(
    log_t: usize,
    increments: &[SparseRamIncrement],
) -> Result<(), RamFamilyV2Error> {
    let cycle_domain = 1u64 << log_t;
    let mut previous = None;
    for increment in increments {
        if u64::from(increment.cycle) >= cycle_domain {
            return Err(RamFamilyV2Error::IncrementCycleOutOfRange {
                cycle: increment.cycle,
                log_t,
            });
        }
        if previous.is_some_and(|cycle| cycle >= increment.cycle) {
            return Err(RamFamilyV2Error::IncrementsOutOfOrder {
                cycle: increment.cycle,
            });
        }
        if increment.value == 0 {
            return Err(RamFamilyV2Error::ZeroIncrement {
                cycle: increment.cycle,
            });
        }
        previous = Some(increment.cycle);
    }
    Ok(())
}

fn validate_access_deltas(
    accesses: &[SparseRamAccess],
    increments: &[SparseRamIncrement],
) -> Result<(), RamFamilyV2Error> {
    for access in accesses {
        let expected = i128::from(access.post_value) - i128::from(access.pre_value);
        let actual = increments
            .binary_search_by_key(&access.cycle, |increment| increment.cycle)
            .ok()
            .map(|index| increments[index].value);
        if (expected == 0 && actual.is_some()) || (expected != 0 && actual != Some(expected)) {
            return Err(RamFamilyV2Error::IncrementDeltaMismatch {
                cycle: access.cycle,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn merge_source_cycles(
    accesses: &[SparseRamAccess],
    increments: &[SparseRamIncrement],
) -> Result<Vec<SparseCycleLeaf>, RamFamilyV2Error> {
    let mut leaves = Vec::with_capacity(accesses.len().saturating_add(increments.len()));
    let mut access = 0;
    let mut increment = 0;
    while access < accesses.len() || increment < increments.len() {
        let access_cycle = accesses.get(access).map(|row| row.cycle);
        let increment_cycle = increments.get(increment).map(|row| row.cycle);
        let cycle = match (access_cycle, increment_cycle) {
            (Some(lhs), Some(rhs)) => lhs.min(rhs),
            (Some(lhs), None) => lhs,
            (None, Some(rhs)) => rhs,
            (None, None) => break,
        };
        let access_index = (access_cycle == Some(cycle))
            .then(|| u32::try_from(access))
            .transpose()
            .map_err(|_| RamFamilyV2Error::TopologyIndexOverflow)?;
        let increment_index = (increment_cycle == Some(cycle))
            .then(|| u32::try_from(increment))
            .transpose()
            .map_err(|_| RamFamilyV2Error::TopologyIndexOverflow)?;
        leaves.push(SparseCycleLeaf {
            cycle,
            access_index,
            increment_index,
        });
        access += usize::from(access_index.is_some());
        increment += usize::from(increment_index.is_some());
    }
    Ok(leaves)
}
