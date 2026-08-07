//! Checked CSR topology for the compressed address path.

use core::mem::size_of;

use super::super::instruction_read_raf_producer::{
    AddressAtomTopology as ProducerAddressAtomTopology, ProducerShardPlan,
};
use super::oracle::InstructionReadRafRow;
use super::shader_abi::{
    segment_index, AddressJob, AddressLookup, AtomMassGroup, AtomMassJob, SplitAtom, SEGMENTS,
    SEGMENT_OFFSETS,
};
use super::{
    InstructionReadRafGeometry, InstructionReadRafV3Error, ADDRESS_PHASES, PRODUCTION_VIRTUAL_RA,
};

const DEFAULT_PHASE_ZERO_CYCLES_PER_GROUP: usize = 1 << 16;
const DEFAULT_MASS_JOB_CYCLES: usize = DEFAULT_PHASE_ZERO_CYCLES_PER_GROUP / 32;
const DEFAULT_ATOMS_PER_PHASE_JOB: usize = 1 << 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AddressAtomTopologyConfig {
    pub(crate) phase_zero_cycles_per_group: usize,
    pub(crate) mass_job_cycles: usize,
    pub(crate) atoms_per_phase_job: usize,
}

impl Default for AddressAtomTopologyConfig {
    fn default() -> Self {
        Self {
            phase_zero_cycles_per_group: DEFAULT_PHASE_ZERO_CYCLES_PER_GROUP,
            mass_job_cycles: DEFAULT_MASS_JOB_CYCLES,
            atoms_per_phase_job: DEFAULT_ATOMS_PER_PHASE_JOB,
        }
    }
}

impl AddressAtomTopologyConfig {
    fn validate(self) -> Result<Self, InstructionReadRafV3Error> {
        for (name, value) in [
            (
                "phase-zero cycles per group",
                self.phase_zero_cycles_per_group,
            ),
            ("mass-job cycles", self.mass_job_cycles),
            ("atoms per later-phase job", self.atoms_per_phase_job),
        ] {
            if value == 0 || value > 1 << 16 || !value.is_power_of_two() {
                return Err(InstructionReadRafV3Error::InvalidTopologyConfig { name, value });
            }
        }
        if self.mass_job_cycles > self.phase_zero_cycles_per_group
            || self.phase_zero_cycles_per_group / self.mass_job_cycles < 32
        {
            return Err(InstructionReadRafV3Error::InvalidTopologyConfig {
                name: "mass-job occupancy ratio",
                value: self.mass_job_cycles,
            });
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AddressAtomTopologyCensus {
    pub(super) rows: usize,
    pub(super) atoms: usize,
    pub(super) mass_jobs: usize,
    pub(super) mass_groups: usize,
    pub(super) split_atoms: usize,
    pub(super) mass_partials: usize,
    pub(super) later_phase_jobs: usize,
    pub(super) resident_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AddressAtomTopology {
    rows: usize,
    atom_lookups: Vec<AddressLookup>,
    atom_cycle_offsets: Vec<u32>,
    cycle_indices: Vec<u32>,
    segment_atom_offsets: [u32; SEGMENT_OFFSETS],
    mass_jobs: Vec<AtomMassJob>,
    mass_groups: Vec<AtomMassGroup>,
    phase_zero_group_offsets: [u32; SEGMENT_OFFSETS],
    split_atoms: Vec<SplitAtom>,
    mass_partials: usize,
    phase_jobs: Vec<AddressJob>,
    phase_job_offsets: [u32; SEGMENT_OFFSETS],
}

impl AddressAtomTopology {
    /// Publishes topology only after verifying an exact-key sorted permutation.
    pub(crate) fn from_sorted_cycles(
        rows: &[InstructionReadRafRow],
        sorted_cycles: &[u32],
        config: AddressAtomTopologyConfig,
    ) -> Result<Self, InstructionReadRafV3Error> {
        Self::from_sorted_cycles_by(rows.len(), sorted_cycles, config, |cycle| Ok(rows[cycle]))
    }

    pub(crate) fn from_sorted_cycles_by(
        rows: usize,
        sorted_cycles: &[u32],
        config: AddressAtomTopologyConfig,
        mut row_at: impl FnMut(usize) -> Result<InstructionReadRafRow, InstructionReadRafV3Error>,
    ) -> Result<Self, InstructionReadRafV3Error> {
        let _geometry = InstructionReadRafGeometry::new(rows, PRODUCTION_VIRTUAL_RA)?;
        let config = config.validate()?;
        if sorted_cycles.len() != rows {
            return Err(InstructionReadRafV3Error::TopologyCycleLength {
                expected: rows,
                got: sorted_cycles.len(),
            });
        }

        let mut seen = vec![0u64; rows.div_ceil(64)];
        for (position, &cycle_u32) in sorted_cycles.iter().enumerate() {
            let cycle = cycle_u32 as usize;
            if cycle >= rows {
                return Err(InstructionReadRafV3Error::TopologyCycleOutOfRange {
                    position,
                    cycle,
                    rows,
                });
            }
            let seen_word = &mut seen[cycle / 64];
            let seen_bit = 1u64 << (cycle % 64);
            if *seen_word & seen_bit != 0 {
                return Err(InstructionReadRafV3Error::DuplicateTopologyCycle { cycle });
            }
            *seen_word |= seen_bit;
        }

        let mut atom_lookups = Vec::new();
        let mut atom_cycle_offsets = vec![0u32];
        let mut segment_atom_counts = [0usize; SEGMENTS];
        let mut previous_key = None;

        for (position, &cycle_u32) in sorted_cycles.iter().enumerate() {
            let cycle = cycle_u32 as usize;
            let row = row_at(cycle)?;
            let segment = segment_index(row.table_index(), row.raf_flag())?;
            let key = (segment, row.lookup_index());
            if previous_key.is_some_and(|previous| key < previous) {
                return Err(InstructionReadRafV3Error::NonMonotoneTopologyKey { position });
            }
            if previous_key != Some(key) {
                if position != 0 {
                    atom_cycle_offsets.push(shader_u32(position, "atom cycle offset")?);
                }
                atom_lookups.push(AddressLookup::new(row.lookup_index()));
                segment_atom_counts[segment] = segment_atom_counts[segment].checked_add(1).ok_or(
                    InstructionReadRafV3Error::SizeOverflow("segment atom count"),
                )?;
                previous_key = Some(key);
            }
        }
        atom_cycle_offsets.push(shader_u32(rows, "terminal atom cycle offset")?);
        if atom_lookups.is_empty() || atom_lookups.len() > rows {
            return Err(InstructionReadRafV3Error::InvalidTopology(
                "atom count is outside 1..=rows",
            ));
        }

        let mut segment_atom_offsets = [0u32; SEGMENT_OFFSETS];
        let mut atom_cursor = 0usize;
        for (segment, count) in segment_atom_counts.into_iter().enumerate() {
            atom_cursor =
                atom_cursor
                    .checked_add(count)
                    .ok_or(InstructionReadRafV3Error::SizeOverflow(
                        "segment atom offsets",
                    ))?;
            segment_atom_offsets[segment + 1] = shader_u32(atom_cursor, "segment atom offset")?;
        }

        Self::from_checked_parts(
            rows,
            atom_lookups,
            atom_cycle_offsets,
            sorted_cycles.to_vec(),
            &segment_atom_offsets,
            config,
        )
    }

    /// Builds only v3 job plans around a producer-checked exact-key CSR.
    pub(crate) fn from_producer_topology(
        producer: &ProducerAddressAtomTopology,
        config: AddressAtomTopologyConfig,
    ) -> Result<Self, InstructionReadRafV3Error> {
        validate_one_shard(producer.shard())?;
        let parts = producer.parts();
        Self::from_checked_parts(
            producer.rows(),
            parts.atom_lookups.clone(),
            parts.atom_cycle_offsets.clone(),
            parts.cycle_indices.clone(),
            &parts.segment_atom_offsets,
            config,
        )
    }

    pub(crate) fn from_checked_parts(
        rows: usize,
        atom_lookups: Vec<AddressLookup>,
        atom_cycle_offsets: Vec<u32>,
        cycle_indices: Vec<u32>,
        segment_atom_offsets: &[u32; SEGMENT_OFFSETS],
        config: AddressAtomTopologyConfig,
    ) -> Result<Self, InstructionReadRafV3Error> {
        let _geometry = InstructionReadRafGeometry::new(rows, PRODUCTION_VIRTUAL_RA)?;
        let config = config.validate()?;
        validate_base_parts(
            rows,
            &atom_lookups,
            &atom_cycle_offsets,
            &cycle_indices,
            segment_atom_offsets,
        )?;
        let plans = build_job_plans(&atom_cycle_offsets, segment_atom_offsets, config)?;
        let topology = Self {
            rows,
            atom_lookups,
            atom_cycle_offsets,
            cycle_indices,
            segment_atom_offsets: *segment_atom_offsets,
            mass_jobs: plans.mass_jobs,
            mass_groups: plans.mass_groups,
            phase_zero_group_offsets: plans.phase_zero_group_offsets,
            split_atoms: plans.split_atoms,
            mass_partials: plans.mass_partials,
            phase_jobs: plans.phase_jobs,
            phase_job_offsets: plans.phase_job_offsets,
        };
        topology.validate(config)?;
        Ok(topology)
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub(super) fn from_rows_reference(
        rows: &[InstructionReadRafRow],
        config: AddressAtomTopologyConfig,
    ) -> Result<Self, InstructionReadRafV3Error> {
        let _geometry = InstructionReadRafGeometry::new(rows.len(), PRODUCTION_VIRTUAL_RA)?;
        let mut keyed = rows
            .iter()
            .copied()
            .enumerate()
            .map(|(cycle, row)| {
                Ok((
                    (
                        segment_index(row.table_index(), row.raf_flag())?,
                        row.lookup_index(),
                    ),
                    shader_u32(cycle, "reference topology cycle")?,
                ))
            })
            .collect::<Result<Vec<_>, InstructionReadRafV3Error>>()?;
        keyed.sort_unstable_by_key(|(key, _)| *key);
        let sorted_cycles = keyed
            .into_iter()
            .map(|(_, cycle)| cycle)
            .collect::<Vec<_>>();
        Self::from_sorted_cycles(rows, &sorted_cycles, config)
    }

    pub(super) const fn rows(&self) -> usize {
        self.rows
    }

    pub(super) fn atom_lookups(&self) -> &[AddressLookup] {
        &self.atom_lookups
    }

    pub(super) fn atom_cycle_offsets(&self) -> &[u32] {
        &self.atom_cycle_offsets
    }

    pub(super) fn cycle_indices(&self) -> &[u32] {
        &self.cycle_indices
    }

    pub(super) const fn segment_atom_offsets(&self) -> &[u32; SEGMENT_OFFSETS] {
        &self.segment_atom_offsets
    }

    pub(super) fn mass_jobs(&self) -> &[AtomMassJob] {
        &self.mass_jobs
    }

    pub(super) fn mass_groups(&self) -> &[AtomMassGroup] {
        &self.mass_groups
    }

    pub(super) const fn phase_zero_group_offsets(&self) -> &[u32; SEGMENT_OFFSETS] {
        &self.phase_zero_group_offsets
    }

    pub(super) fn split_atoms(&self) -> &[SplitAtom] {
        &self.split_atoms
    }

    pub(super) const fn mass_partials(&self) -> usize {
        self.mass_partials
    }

    pub(super) fn phase_jobs(&self) -> &[AddressJob] {
        &self.phase_jobs
    }

    pub(super) const fn phase_job_offsets(&self) -> &[u32; SEGMENT_OFFSETS] {
        &self.phase_job_offsets
    }

    pub(super) fn census(&self) -> Result<AddressAtomTopologyCensus, InstructionReadRafV3Error> {
        Ok(AddressAtomTopologyCensus {
            rows: self.rows,
            atoms: self.atom_lookups.len(),
            mass_jobs: self.mass_jobs.len(),
            mass_groups: self.mass_groups.len(),
            split_atoms: self.split_atoms.len(),
            mass_partials: self.mass_partials,
            later_phase_jobs: self.phase_jobs.len(),
            resident_bytes: self.resident_bytes()?,
        })
    }

    pub(super) fn phase_jobs_census(&self) -> [u64; ADDRESS_PHASES] {
        let mut jobs = [self.phase_jobs.len() as u64; ADDRESS_PHASES];
        jobs[0] = self.mass_groups.len() as u64;
        jobs
    }

    fn resident_bytes(&self) -> Result<usize, InstructionReadRafV3Error> {
        checked_sum(&[
            checked_bytes::<AddressLookup>(self.atom_lookups.len(), "atom lookups")?,
            checked_bytes::<u32>(self.atom_cycle_offsets.len(), "atom cycle offsets")?,
            checked_bytes::<u32>(self.cycle_indices.len(), "cycle permutation")?,
            size_of::<[u32; SEGMENT_OFFSETS]>(),
            checked_bytes::<AtomMassJob>(self.mass_jobs.len(), "atom mass jobs")?,
            checked_bytes::<AtomMassGroup>(self.mass_groups.len(), "atom mass groups")?,
            size_of::<[u32; SEGMENT_OFFSETS]>(),
            checked_bytes::<SplitAtom>(self.split_atoms.len(), "split atoms")?,
            checked_bytes::<AddressJob>(self.phase_jobs.len(), "address jobs")?,
            size_of::<[u32; SEGMENT_OFFSETS]>(),
        ])
    }

    fn validate(&self, config: AddressAtomTopologyConfig) -> Result<(), InstructionReadRafV3Error> {
        let atoms = self.atom_lookups.len();
        if self.atom_cycle_offsets.len() != atoms + 1
            || self.atom_cycle_offsets.first() != Some(&0)
            || self.atom_cycle_offsets.last() != Some(&(self.rows as u32))
            || self
                .atom_cycle_offsets
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(InstructionReadRafV3Error::InvalidTopology(
                "atom cycle offsets do not strictly partition the rows",
            ));
        }
        validate_offsets(
            &self.segment_atom_offsets,
            atoms,
            "segment atom offsets do not cover the atoms",
        )?;
        validate_offsets(
            &self.phase_zero_group_offsets,
            self.mass_groups.len(),
            "phase-zero group offsets do not cover the groups",
        )?;
        validate_offsets(
            &self.phase_job_offsets,
            self.phase_jobs.len(),
            "phase job offsets do not cover the jobs",
        )?;
        self.validate_mass_plan(config)?;
        self.validate_phase_plan()?;
        Ok(())
    }

    fn validate_mass_plan(
        &self,
        config: AddressAtomTopologyConfig,
    ) -> Result<(), InstructionReadRafV3Error> {
        let atoms = self.atom_lookups.len();
        let mut next_cycle = self.atom_cycle_offsets[..atoms].to_vec();
        let mut jobs_per_atom = vec![0usize; atoms];
        let mut partials_per_atom = vec![None; atoms];
        let mut next_partial = 0usize;

        for job in &self.mass_jobs {
            let atom = job.atom as usize;
            if atom >= atoms
                || job.cycle_start >= job.cycle_end
                || job.cycle_start != next_cycle[atom]
                || job.cycle_end > self.atom_cycle_offsets[atom + 1]
                || job.cycle_end - job.cycle_start > config.mass_job_cycles as u32
            {
                return Err(InstructionReadRafV3Error::InvalidTopology(
                    "mass jobs do not exactly partition atom cycles",
                ));
            }
            next_cycle[atom] = job.cycle_end;
            jobs_per_atom[atom] += 1;
            if job.mass_partial_plus_one != 0 {
                if job.mass_partial_plus_one as usize != next_partial + 1 {
                    return Err(InstructionReadRafV3Error::InvalidTopology(
                        "mass partial indices are not a dense permutation",
                    ));
                }
                let range = partials_per_atom[atom].get_or_insert((next_partial, next_partial));
                range.1 = next_partial + 1;
                next_partial += 1;
            }
        }
        if next_partial != self.mass_partials
            || next_cycle
                .iter()
                .zip(&self.atom_cycle_offsets[1..])
                .any(|(got, expected)| got != expected)
            || jobs_per_atom.contains(&0)
        {
            return Err(InstructionReadRafV3Error::InvalidTopology(
                "mass jobs leave an atom or partial uncovered",
            ));
        }
        let expected_jobs = atoms
            .checked_sub(self.split_atoms.len())
            .and_then(|value| value.checked_add(self.mass_partials))
            .ok_or(InstructionReadRafV3Error::SizeOverflow("mass-job identity"))?;
        if self.mass_jobs.len() != expected_jobs || self.split_atoms.len() > atoms {
            return Err(InstructionReadRafV3Error::InvalidTopology(
                "mass-job census identity failed",
            ));
        }

        let expected_splits = jobs_per_atom
            .iter()
            .enumerate()
            .filter_map(|(atom, &jobs)| (jobs > 1).then_some(atom))
            .collect::<Vec<_>>();
        if expected_splits.len() != self.split_atoms.len() {
            return Err(InstructionReadRafV3Error::InvalidTopology(
                "split atom list differs from the mass plan",
            ));
        }
        for (&atom, split) in expected_splits.iter().zip(&self.split_atoms) {
            let partial_range = partials_per_atom[atom].ok_or(
                InstructionReadRafV3Error::InvalidTopology("split atom is missing partials"),
            )?;
            if split.atom as usize != atom
                || split.partial_start as usize != partial_range.0
                || split.partial_end as usize != partial_range.1
                || split.reserved != 0
            {
                return Err(InstructionReadRafV3Error::InvalidTopology(
                    "split atom range differs from the mass plan",
                ));
            }
        }

        let mut job_cursor = 0usize;
        for segment in 0..SEGMENTS {
            let group_start = self.phase_zero_group_offsets[segment] as usize;
            let group_end = self.phase_zero_group_offsets[segment + 1] as usize;
            let atom_start = self.segment_atom_offsets[segment] as usize;
            let atom_end = self.segment_atom_offsets[segment + 1] as usize;
            for group in &self.mass_groups[group_start..group_end] {
                if group.reserved != 0
                    || group.segment as usize != segment
                    || group.job_start as usize != job_cursor
                    || group.job_start >= group.job_end
                    || group.job_end as usize > self.mass_jobs.len()
                {
                    return Err(InstructionReadRafV3Error::InvalidTopology(
                        "mass groups do not exactly cover mass jobs",
                    ));
                }
                let mut cycles = 0usize;
                for job in &self.mass_jobs[group.job_start as usize..group.job_end as usize] {
                    if !(atom_start..atom_end).contains(&(job.atom as usize)) {
                        return Err(InstructionReadRafV3Error::InvalidTopology(
                            "mass group crosses a claim segment",
                        ));
                    }
                    cycles = cycles
                        .checked_add((job.cycle_end - job.cycle_start) as usize)
                        .ok_or(InstructionReadRafV3Error::SizeOverflow(
                            "mass group cycle count",
                        ))?;
                }
                if cycles > config.phase_zero_cycles_per_group {
                    return Err(InstructionReadRafV3Error::InvalidTopology(
                        "mass group exceeds its cycle budget",
                    ));
                }
                job_cursor = group.job_end as usize;
            }
        }
        if job_cursor != self.mass_jobs.len() {
            return Err(InstructionReadRafV3Error::InvalidTopology(
                "mass groups leave jobs uncovered",
            ));
        }
        Ok(())
    }

    fn validate_phase_plan(&self) -> Result<(), InstructionReadRafV3Error> {
        for segment in 0..SEGMENTS {
            let mut atom_cursor = self.segment_atom_offsets[segment] as usize;
            let atom_end = self.segment_atom_offsets[segment + 1] as usize;
            let job_start = self.phase_job_offsets[segment] as usize;
            let job_end = self.phase_job_offsets[segment + 1] as usize;
            for job in &self.phase_jobs[job_start..job_end] {
                if job.reserved != 0
                    || job.segment as usize != segment
                    || job.start as usize != atom_cursor
                    || job.start >= job.end
                    || job.end as usize > atom_end
                {
                    return Err(InstructionReadRafV3Error::InvalidTopology(
                        "later-phase jobs do not exactly partition segment atoms",
                    ));
                }
                atom_cursor = job.end as usize;
            }
            if atom_cursor != atom_end {
                return Err(InstructionReadRafV3Error::InvalidTopology(
                    "later-phase jobs leave atoms uncovered",
                ));
            }
        }
        Ok(())
    }
}

struct JobPlans {
    mass_jobs: Vec<AtomMassJob>,
    mass_groups: Vec<AtomMassGroup>,
    phase_zero_group_offsets: [u32; SEGMENT_OFFSETS],
    split_atoms: Vec<SplitAtom>,
    mass_partials: usize,
    phase_jobs: Vec<AddressJob>,
    phase_job_offsets: [u32; SEGMENT_OFFSETS],
}

pub(super) fn validate_one_shard(
    shard: ProducerShardPlan,
) -> Result<(), InstructionReadRafV3Error> {
    if shard.shard_index() != 0
        || shard.absolute_row_start() != 0
        || shard.rows() != shard.total_rows()
    {
        return Err(InstructionReadRafV3Error::UnsupportedProducerShard {
            total_rows: shard.total_rows(),
            shard_index: shard.shard_index(),
            shard_rows: shard.rows(),
        });
    }
    Ok(())
}

fn validate_base_parts(
    rows: usize,
    atom_lookups: &[AddressLookup],
    atom_cycle_offsets: &[u32],
    cycle_indices: &[u32],
    segment_atom_offsets: &[u32; SEGMENT_OFFSETS],
) -> Result<(), InstructionReadRafV3Error> {
    let atoms = atom_lookups.len();
    if atoms == 0 || atoms > rows {
        return Err(InstructionReadRafV3Error::InvalidAtomCount { rows, atoms });
    }
    if atom_cycle_offsets.len() != atoms + 1 {
        return Err(InstructionReadRafV3Error::AtomTopologyLength {
            name: "atom cycle offsets",
            expected: atoms + 1,
            got: atom_cycle_offsets.len(),
        });
    }
    if cycle_indices.len() != rows {
        return Err(InstructionReadRafV3Error::AtomTopologyLength {
            name: "cycle indices",
            expected: rows,
            got: cycle_indices.len(),
        });
    }
    if atom_cycle_offsets.first() != Some(&0)
        || atom_cycle_offsets.last() != Some(&(rows as u32))
        || atom_cycle_offsets.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(InstructionReadRafV3Error::InvalidTopology(
            "atom cycle offsets do not strictly partition the rows",
        ));
    }
    validate_offsets(
        segment_atom_offsets,
        atoms,
        "segment atom offsets do not cover the atoms",
    )?;
    for segment in 0..SEGMENTS {
        let start = segment_atom_offsets[segment] as usize;
        let end = segment_atom_offsets[segment + 1] as usize;
        if atom_lookups[start..end]
            .windows(2)
            .any(|pair| pair[0].value() >= pair[1].value())
        {
            return Err(InstructionReadRafV3Error::InvalidTopology(
                "atom lookups are not strictly increasing inside a selector range",
            ));
        }
    }
    let mut seen = vec![false; rows];
    for (position, &cycle_u32) in cycle_indices.iter().enumerate() {
        let cycle = cycle_u32 as usize;
        if cycle >= rows {
            return Err(InstructionReadRafV3Error::TopologyCycleOutOfRange {
                position,
                cycle,
                rows,
            });
        }
        if seen[cycle] {
            return Err(InstructionReadRafV3Error::DuplicateTopologyCycle { cycle });
        }
        seen[cycle] = true;
    }
    Ok(())
}

fn build_job_plans(
    atom_cycle_offsets: &[u32],
    segment_atom_offsets: &[u32; SEGMENT_OFFSETS],
    config: AddressAtomTopologyConfig,
) -> Result<JobPlans, InstructionReadRafV3Error> {
    let mut mass_jobs = Vec::new();
    let mut mass_groups = Vec::new();
    let mut phase_zero_group_offsets = [0u32; SEGMENT_OFFSETS];
    let mut split_atoms = Vec::new();
    let mut mass_partials = 0usize;
    let mut phase_jobs = Vec::new();
    let mut phase_job_offsets = [0u32; SEGMENT_OFFSETS];

    for segment in 0..SEGMENTS {
        let atom_start = segment_atom_offsets[segment] as usize;
        let atom_end = segment_atom_offsets[segment + 1] as usize;
        let mut group_job_start = mass_jobs.len();
        let mut group_cycles = 0usize;

        for atom in atom_start..atom_end {
            let cycle_start = atom_cycle_offsets[atom] as usize;
            let cycle_end = atom_cycle_offsets[atom + 1] as usize;
            let split = cycle_end - cycle_start > config.mass_job_cycles;
            let partial_start = mass_partials;

            for start in (cycle_start..cycle_end).step_by(config.mass_job_cycles) {
                let end = (start + config.mass_job_cycles).min(cycle_end);
                let job_cycles = end - start;
                if group_cycles != 0
                    && group_cycles + job_cycles > config.phase_zero_cycles_per_group
                {
                    push_mass_group(&mut mass_groups, group_job_start, mass_jobs.len(), segment)?;
                    group_job_start = mass_jobs.len();
                    group_cycles = 0;
                }
                let mass_partial_plus_one = if split {
                    mass_partials = mass_partials.checked_add(1).ok_or(
                        InstructionReadRafV3Error::SizeOverflow("mass partial count"),
                    )?;
                    shader_u32(mass_partials, "mass partial index")?
                } else {
                    0
                };
                mass_jobs.push(AtomMassJob {
                    cycle_start: shader_u32(start, "mass-job cycle start")?,
                    cycle_end: shader_u32(end, "mass-job cycle end")?,
                    atom: shader_u32(atom, "mass-job atom")?,
                    mass_partial_plus_one,
                });
                group_cycles += job_cycles;
            }

            if split {
                split_atoms.push(SplitAtom {
                    atom: shader_u32(atom, "split atom")?,
                    partial_start: shader_u32(partial_start, "split partial start")?,
                    partial_end: shader_u32(mass_partials, "split partial end")?,
                    reserved: 0,
                });
            }
        }
        push_mass_group(&mut mass_groups, group_job_start, mass_jobs.len(), segment)?;
        phase_zero_group_offsets[segment + 1] =
            shader_u32(mass_groups.len(), "phase-zero group offset")?;

        for start in (atom_start..atom_end).step_by(config.atoms_per_phase_job) {
            let end = (start + config.atoms_per_phase_job).min(atom_end);
            phase_jobs.push(AddressJob {
                start: shader_u32(start, "address-job atom start")?,
                end: shader_u32(end, "address-job atom end")?,
                segment: shader_u32(segment, "address-job segment")?,
                reserved: 0,
            });
        }
        phase_job_offsets[segment + 1] = shader_u32(phase_jobs.len(), "address-job offset")?;
    }

    Ok(JobPlans {
        mass_jobs,
        mass_groups,
        phase_zero_group_offsets,
        split_atoms,
        mass_partials,
        phase_jobs,
        phase_job_offsets,
    })
}

fn push_mass_group(
    groups: &mut Vec<AtomMassGroup>,
    job_start: usize,
    job_end: usize,
    segment: usize,
) -> Result<(), InstructionReadRafV3Error> {
    if job_start == job_end {
        return Ok(());
    }
    groups.push(AtomMassGroup {
        job_start: shader_u32(job_start, "mass-group job start")?,
        job_end: shader_u32(job_end, "mass-group job end")?,
        segment: shader_u32(segment, "mass-group segment")?,
        reserved: 0,
    });
    Ok(())
}

fn validate_offsets<const N: usize>(
    offsets: &[u32; N],
    expected_end: usize,
    error: &'static str,
) -> Result<(), InstructionReadRafV3Error> {
    if offsets.first() != Some(&0)
        || offsets.last() != Some(&(expected_end as u32))
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(InstructionReadRafV3Error::InvalidTopology(error));
    }
    Ok(())
}

fn checked_bytes<T>(
    elements: usize,
    name: &'static str,
) -> Result<usize, InstructionReadRafV3Error> {
    elements
        .checked_mul(size_of::<T>())
        .ok_or(InstructionReadRafV3Error::SizeOverflow(name))
}

fn checked_sum(values: &[usize]) -> Result<usize, InstructionReadRafV3Error> {
    values.iter().try_fold(0usize, |sum, value| {
        sum.checked_add(*value)
            .ok_or(InstructionReadRafV3Error::SizeOverflow(
                "topology resident bytes",
            ))
    })
}

fn shader_u32(value: usize, name: &'static str) -> Result<u32, InstructionReadRafV3Error> {
    u32::try_from(value).map_err(|_| InstructionReadRafV3Error::SizeOverflow(name))
}
