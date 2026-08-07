use core::{mem::size_of, ops::Range};

use jolt_field::Field;

use super::super::{decode_claim, ProducerShardPlan, GROUPED_SEGMENTS, GROUPED_SEGMENT_OFFSETS};
use super::accounting::AddressAtomShape;
use super::error::{validate_atom_count, AddressAtomError, AddressAtomResult};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AddressAtomLookup {
    limbs: [u64; 2],
}

impl AddressAtomLookup {
    pub const fn new(value: u128) -> Self {
        Self {
            limbs: [value as u64, (value >> 64) as u64],
        }
    }

    pub const fn limbs(self) -> [u64; 2] {
        self.limbs
    }

    pub const fn value(self) -> u128 {
        self.limbs[0] as u128 | ((self.limbs[1] as u128) << 64)
    }
}

const _: () = assert!(size_of::<AddressAtomLookup>() == 16);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressAtomCycleRow {
    lookup: AddressAtomLookup,
    claim: u8,
}

impl AddressAtomCycleRow {
    pub const fn lookup(self) -> AddressAtomLookup {
        self.lookup
    }

    pub const fn claim(self) -> u8 {
        self.claim
    }

    fn key(self) -> AddressAtomResult<(usize, u128)> {
        Ok((decode_claim(self.claim)?.segment(), self.lookup.value()))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AddressAtomCycleSource<'a> {
    shard: ProducerShardPlan,
    cycle_lookup_lo: &'a [u64],
    cycle_lookup_hi: &'a [u64],
    cycle_claims: &'a [u8],
}

impl<'a> AddressAtomCycleSource<'a> {
    pub fn new(
        shard: ProducerShardPlan,
        cycle_lookup_lo: &'a [u64],
        cycle_lookup_hi: &'a [u64],
        cycle_claims: &'a [u8],
    ) -> AddressAtomResult<Self> {
        for (name, got) in [
            ("cycle lookup low limbs", cycle_lookup_lo.len()),
            ("cycle lookup high limbs", cycle_lookup_hi.len()),
            ("cycle claims", cycle_claims.len()),
        ] {
            if got != shard.rows() {
                return Err(AddressAtomError::SourcePlaneElements {
                    name,
                    expected: shard.rows(),
                    got,
                });
            }
        }
        for &claim in cycle_claims {
            let _selector = decode_claim(claim)?;
        }
        Ok(Self {
            shard,
            cycle_lookup_lo,
            cycle_lookup_hi,
            cycle_claims,
        })
    }

    pub const fn shard(self) -> ProducerShardPlan {
        self.shard
    }

    pub const fn rows(self) -> usize {
        self.shard.rows()
    }

    pub fn row(self, cycle: usize) -> AddressAtomResult<AddressAtomCycleRow> {
        if cycle >= self.rows() {
            return Err(AddressAtomError::CycleOutOfRange {
                position: cycle,
                cycle,
                rows: self.rows(),
            });
        }
        Ok(AddressAtomCycleRow {
            lookup: AddressAtomLookup {
                limbs: [self.cycle_lookup_lo[cycle], self.cycle_lookup_hi[cycle]],
            },
            claim: self.cycle_claims[cycle],
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressAtomTopologyParts {
    pub atom_lookups: Vec<AddressAtomLookup>,
    pub atom_claims: Vec<u8>,
    pub atom_cycle_offsets: Vec<u32>,
    pub cycle_indices: Vec<u32>,
    pub cycle_to_atom: Vec<u32>,
    pub segment_atom_offsets: [u32; GROUPED_SEGMENT_OFFSETS],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressAtomTopology {
    shard: ProducerShardPlan,
    parts: AddressAtomTopologyParts,
}

impl AddressAtomTopology {
    /// Checks an already sorted producer permutation before publishing the CSR.
    pub fn from_sorted_cycles(
        source: AddressAtomCycleSource<'_>,
        sorted_cycles: &[u32],
    ) -> AddressAtomResult<Self> {
        let rows = source.rows();
        if sorted_cycles.len() != rows {
            return Err(AddressAtomError::SortedCycleLength {
                expected: rows,
                got: sorted_cycles.len(),
            });
        }

        let mut seen = vec![0u64; rows.div_ceil(64)];
        let mut atom_lookups = Vec::new();
        let mut atom_claims = Vec::new();
        let mut atom_cycle_offsets = vec![0u32];
        let mut cycle_indices = Vec::with_capacity(rows);
        let mut cycle_to_atom = vec![u32::MAX; rows];
        let mut segment_atom_counts = [0usize; GROUPED_SEGMENTS];
        let mut previous_key = None;

        for (position, &cycle_u32) in sorted_cycles.iter().enumerate() {
            let cycle = cycle_u32 as usize;
            if cycle >= rows {
                return Err(AddressAtomError::CycleOutOfRange {
                    position,
                    cycle,
                    rows,
                });
            }
            let seen_word = &mut seen[cycle / 64];
            let seen_bit = 1u64 << (cycle % 64);
            if *seen_word & seen_bit != 0 {
                return Err(AddressAtomError::DuplicateCycle { cycle });
            }
            *seen_word |= seen_bit;

            let row = source.row(cycle)?;
            let key = row.key()?;
            if previous_key.is_some_and(|previous| key < previous) {
                return Err(AddressAtomError::NonMonotoneKey { position });
            }
            if previous_key != Some(key) {
                if position != 0 {
                    atom_cycle_offsets.push(shader_u32(position, "atom cycle offset")?);
                }
                atom_lookups.push(row.lookup());
                atom_claims.push(row.claim());
                segment_atom_counts[key.0] = segment_atom_counts[key.0]
                    .checked_add(1)
                    .ok_or(AddressAtomError::SizeOverflow("segment atom count"))?;
                previous_key = Some(key);
            }
            let atom =
                atom_lookups
                    .len()
                    .checked_sub(1)
                    .ok_or(AddressAtomError::InvalidTopology(
                        "the first cycle did not create an atom",
                    ))?;
            cycle_to_atom[cycle] = shader_u32(atom, "cycle-to-atom inverse")?;
            cycle_indices.push(cycle_u32);
        }

        atom_cycle_offsets.push(shader_u32(rows, "terminal atom cycle offset")?);
        let atoms = atom_lookups.len();
        validate_atom_count(rows, atoms)?;
        let mut segment_atom_offsets = [0u32; GROUPED_SEGMENT_OFFSETS];
        let mut atom_cursor = 0usize;
        for (segment, count) in segment_atom_counts.into_iter().enumerate() {
            atom_cursor = atom_cursor
                .checked_add(count)
                .ok_or(AddressAtomError::SizeOverflow("segment atom offsets"))?;
            segment_atom_offsets[segment + 1] = shader_u32(atom_cursor, "segment atom offset")?;
        }

        Self::from_checked_parts(
            source,
            AddressAtomTopologyParts {
                atom_lookups,
                atom_claims,
                atom_cycle_offsets,
                cycle_indices,
                cycle_to_atom,
                segment_atom_offsets,
            },
        )
    }

    /// Reference oracle. Production may radix-sort or emit the same permutation directly.
    pub fn from_cycle_source_reference(
        source: AddressAtomCycleSource<'_>,
    ) -> AddressAtomResult<Self> {
        let mut keyed = Vec::with_capacity(source.rows());
        for cycle in 0..source.rows() {
            keyed.push((source.row(cycle)?.key()?, shader_u32(cycle, "cycle index")?));
        }
        keyed.sort_unstable();
        let sorted_cycles = keyed
            .into_iter()
            .map(|(_, cycle)| cycle)
            .collect::<Vec<_>>();
        Self::from_sorted_cycles(source, &sorted_cycles)
    }

    pub fn from_checked_parts(
        source: AddressAtomCycleSource<'_>,
        parts: AddressAtomTopologyParts,
    ) -> AddressAtomResult<Self> {
        let topology = Self {
            shard: source.shard(),
            parts,
        };
        topology.validate_against(source)?;
        Ok(topology)
    }

    pub const fn shard(&self) -> ProducerShardPlan {
        self.shard
    }

    pub const fn rows(&self) -> usize {
        self.shard.rows()
    }

    pub fn atoms(&self) -> usize {
        self.parts.atom_lookups.len()
    }

    pub fn atom_lookups(&self) -> &[AddressAtomLookup] {
        &self.parts.atom_lookups
    }

    pub fn atom_claims(&self) -> &[u8] {
        &self.parts.atom_claims
    }

    pub fn atom_cycle_offsets(&self) -> &[u32] {
        &self.parts.atom_cycle_offsets
    }

    pub fn cycle_indices(&self) -> &[u32] {
        &self.parts.cycle_indices
    }

    pub fn cycle_to_atom(&self) -> &[u32] {
        &self.parts.cycle_to_atom
    }

    pub const fn segment_atom_offsets(&self) -> &[u32; GROUPED_SEGMENT_OFFSETS] {
        &self.parts.segment_atom_offsets
    }

    pub fn atom_cycle_range(&self, atom: usize) -> Option<Range<usize>> {
        let start = *self.parts.atom_cycle_offsets.get(atom)? as usize;
        let end = *self.parts.atom_cycle_offsets.get(atom + 1)? as usize;
        Some(start..end)
    }

    pub fn parts(&self) -> &AddressAtomTopologyParts {
        &self.parts
    }

    pub fn shape(&self) -> AddressAtomResult<AddressAtomShape> {
        AddressAtomShape::new(self.shard, self.atoms())
    }

    /// Computes `mass[a] = sum_{j in a} weight[j]` in shard-local cycle order.
    pub fn masses_from_cycle_weights<F: Field>(
        &self,
        cycle_weights: &[F],
    ) -> AddressAtomResult<Vec<F>> {
        if cycle_weights.len() != self.rows() {
            return Err(AddressAtomError::CycleWeightLength {
                expected: self.rows(),
                got: cycle_weights.len(),
            });
        }
        let mut masses = vec![F::zero(); self.atoms()];
        for (atom, mass) in masses.iter_mut().enumerate() {
            for &cycle in &self.parts.cycle_indices[self.atom_cycle_range(atom).ok_or(
                AddressAtomError::InvalidTopology("an atom is missing its cycle range"),
            )?] {
                *mass += cycle_weights[cycle as usize];
            }
        }
        Ok(masses)
    }

    /// Uses absolute cycle indices, including the shard prefix, for split equality factors.
    pub fn masses_from_split_equality<F: Field>(
        &self,
        e_in: &[F],
        e_out: &[F],
    ) -> AddressAtomResult<Vec<F>> {
        let mut masses = vec![F::zero(); self.atoms()];
        for (atom, mass) in masses.iter_mut().enumerate() {
            for &local_cycle in &self.parts.cycle_indices[self.atom_cycle_range(atom).ok_or(
                AddressAtomError::InvalidTopology("an atom is missing its cycle range"),
            )?] {
                let absolute_cycle = self
                    .shard
                    .absolute_row_start()
                    .checked_add(local_cycle as usize)
                    .ok_or(AddressAtomError::SizeOverflow("absolute cycle index"))?;
                *mass +=
                    split_equality_weight(self.shard.total_rows(), absolute_cycle, e_in, e_out)?;
            }
        }
        Ok(masses)
    }

    pub fn validate_against(&self, source: AddressAtomCycleSource<'_>) -> AddressAtomResult<()> {
        if self.shard != source.shard() {
            return Err(AddressAtomError::ShardMismatch);
        }
        let rows = self.rows();
        let atoms = self.atoms();
        validate_atom_count(rows, atoms)?;
        for (name, expected, got) in [
            ("atom claims", atoms, self.parts.atom_claims.len()),
            (
                "atom cycle offsets",
                atoms + 1,
                self.parts.atom_cycle_offsets.len(),
            ),
            ("cycle indices", rows, self.parts.cycle_indices.len()),
            (
                "cycle-to-atom inverse",
                rows,
                self.parts.cycle_to_atom.len(),
            ),
        ] {
            if got != expected {
                return Err(AddressAtomError::TopologyLength {
                    name,
                    expected,
                    got,
                });
            }
        }
        if self.parts.atom_cycle_offsets.first() != Some(&0)
            || self.parts.atom_cycle_offsets.last() != Some(&(rows as u32))
            || self
                .parts
                .atom_cycle_offsets
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(AddressAtomError::InvalidTopology(
                "atom cycle offsets must strictly partition all cycles",
            ));
        }
        validate_offsets(
            &self.parts.segment_atom_offsets,
            atoms,
            "segment atom offsets must cover every atom",
        )?;

        let mut seen = vec![false; rows];
        let mut previous_key = None;
        for atom in 0..atoms {
            let claim = self.parts.atom_claims[atom];
            let segment = decode_claim(claim)?.segment();
            let key = (segment, self.parts.atom_lookups[atom].value());
            if previous_key.is_some_and(|previous| previous >= key) {
                return Err(AddressAtomError::InvalidTopology(
                    "atom keys must be strictly increasing",
                ));
            }
            let segment_range = self.parts.segment_atom_offsets[segment] as usize
                ..self.parts.segment_atom_offsets[segment + 1] as usize;
            if !segment_range.contains(&atom) {
                return Err(AddressAtomError::InvalidTopology(
                    "atom claim is outside its selector range",
                ));
            }
            for &cycle_u32 in &self.parts.cycle_indices[self.atom_cycle_range(atom).ok_or(
                AddressAtomError::InvalidTopology("an atom is missing its cycle range"),
            )?] {
                let cycle = cycle_u32 as usize;
                if cycle >= rows || seen[cycle] {
                    return Err(AddressAtomError::InvalidTopology(
                        "atom cycles must be an in-range permutation",
                    ));
                }
                seen[cycle] = true;
                if self.parts.cycle_to_atom[cycle] as usize != atom {
                    return Err(AddressAtomError::InvalidTopology(
                        "cycle-to-atom is not the CSR inverse",
                    ));
                }
                let row = source.row(cycle)?;
                if row.claim() != claim || row.lookup() != self.parts.atom_lookups[atom] {
                    return Err(AddressAtomError::InvalidTopology(
                        "an atom key differs from its source cycle",
                    ));
                }
            }
            previous_key = Some(key);
        }
        if seen.contains(&false) {
            return Err(AddressAtomError::InvalidTopology(
                "atom cycles do not cover the source",
            ));
        }
        Ok(())
    }
}

pub fn split_equality_weight<F: Field>(
    total_rows: usize,
    absolute_cycle: usize,
    e_in: &[F],
    e_out: &[F],
) -> AddressAtomResult<F> {
    let factor_rows = e_in
        .len()
        .checked_mul(e_out.len())
        .ok_or(AddressAtomError::SizeOverflow("split equality domain"))?;
    if e_in.is_empty()
        || !e_in.len().is_power_of_two()
        || e_out.is_empty()
        || !e_out.len().is_power_of_two()
        || factor_rows != total_rows
    {
        return Err(AddressAtomError::SplitEqualityShape {
            total_rows,
            e_in: e_in.len(),
            e_out: e_out.len(),
        });
    }
    if absolute_cycle >= total_rows {
        return Err(AddressAtomError::CycleOutOfRange {
            position: absolute_cycle,
            cycle: absolute_cycle,
            rows: total_rows,
        });
    }
    Ok(e_out[absolute_cycle / e_in.len()] * e_in[absolute_cycle & (e_in.len() - 1)])
}

fn validate_offsets(
    offsets: &[u32; GROUPED_SEGMENT_OFFSETS],
    terminal: usize,
    message: &'static str,
) -> AddressAtomResult<()> {
    if offsets[0] != 0
        || offsets[GROUPED_SEGMENTS] as usize != terminal
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(AddressAtomError::InvalidTopology(message));
    }
    Ok(())
}

fn shader_u32(value: usize, name: &'static str) -> AddressAtomResult<u32> {
    u32::try_from(value).map_err(|_| AddressAtomError::SizeOverflow(name))
}
