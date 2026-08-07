use core::mem::size_of;

use jolt_lookup_tables::tables::suffixes::Suffixes;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};

use super::{InstructionReadRafV3Error, ADDRESS_BINS, ADDRESS_PHASES, ADDRESS_PHASE_BITS};

pub(crate) const TABLES: usize = LookupTableKind::<RISCV_XLEN>::COUNT;
pub(super) const TABLE_VALUES: usize = TABLES + 1;
pub(super) const RAF_VALUES: usize = 2;
pub(crate) const SEGMENTS: usize = TABLE_VALUES * RAF_VALUES;
pub(crate) const SEGMENT_OFFSETS: usize = SEGMENTS + 1;
pub(super) const RAF_LANES: usize = 3;
pub(super) const EXPLICIT_SUFFIX_LANES: usize = 3;
pub(super) const JOB_LANES: usize = RAF_LANES + EXPLICIT_SUFFIX_LANES;
pub(crate) const MAX_SUFFIXES: usize = 4;
pub(crate) const TOTAL_SUFFIXES: usize = 88;
pub(crate) const JOB_FIELDS: usize = JOB_LANES * ADDRESS_BINS;
pub(super) const FLAG_COLUMNS: usize = TABLES + 1;
pub(super) const DEFERRED_WORDS: usize = 5;
pub(crate) const SIMD_WIDTH: usize = 32;
pub(crate) const PHASE_THREADGROUP_BYTES: usize =
    JOB_FIELDS * DEFERRED_WORDS * size_of::<u32>();
pub(super) const FLAG_THREADGROUP_BYTES: usize = FLAG_COLUMNS * DEFERRED_WORDS * size_of::<u32>();

const _: () = assert!(TABLES == 40);
const _: () = assert!(SEGMENTS == 82);
const _: () = assert!(JOB_FIELDS == 1536);
const _: () = assert!(PHASE_THREADGROUP_BYTES == 30_720);
const _: () = assert!(FLAG_THREADGROUP_BYTES == 820);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct AddressLookup {
    pub(super) limbs: [u64; 2],
}

impl AddressLookup {
    pub(super) const fn new(value: u128) -> Self {
        Self {
            limbs: [value as u64, (value >> 64) as u64],
        }
    }

    pub(super) const fn value(self) -> u128 {
        self.limbs[0] as u128 | ((self.limbs[1] as u128) << 64)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct AddressJob {
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) segment: u32,
    pub(crate) reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct AtomMassJob {
    pub(super) cycle_start: u32,
    pub(super) cycle_end: u32,
    pub(super) atom: u32,
    pub(super) mass_partial_plus_one: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct AtomMassGroup {
    pub(super) job_start: u32,
    pub(super) job_end: u32,
    pub(super) segment: u32,
    pub(super) reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct SplitAtom {
    pub(super) atom: u32,
    pub(super) partial_start: u32,
    pub(super) partial_end: u32,
    pub(super) reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct TableDescriptor {
    pub(super) output_start: u32,
    pub(super) suffix_count: u32,
    pub(super) segment_raf_zero: u32,
    pub(super) segment_raf_one: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AtomPhaseParams {
    pub(super) suffix_len: u32,
    pub(super) job_count: u32,
    pub(super) reserved: [u32; 2],
}

impl AtomPhaseParams {
    pub(super) fn new(
        suffix_len: usize,
        job_count: usize,
    ) -> Result<Self, InstructionReadRafV3Error> {
        validate_phase(suffix_len, true)?;
        if job_count == 0 {
            return Err(InstructionReadRafV3Error::InvalidShaderAbi(
                "address phase has no jobs",
            ));
        }
        Ok(Self {
            suffix_len: shader_u32(suffix_len, "address suffix length")?,
            job_count: shader_u32(job_count, "address job count")?,
            reserved: [0; 2],
        })
    }

    pub(crate) fn grouped(
        suffix_len: usize,
        job_count: usize,
    ) -> Result<Self, InstructionReadRafV3Error> {
        validate_phase(suffix_len, suffix_len != 120)?;
        if job_count == 0 {
            return Err(InstructionReadRafV3Error::InvalidShaderAbi(
                "grouped address phase has no jobs",
            ));
        }
        Ok(Self {
            suffix_len: shader_u32(suffix_len, "grouped address suffix length")?,
            job_count: shader_u32(job_count, "grouped address job count")?,
            reserved: [0; 2],
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct AtomMassPhaseParams {
    pub(super) rows: u32,
    pub(super) atoms: u32,
    pub(super) mass_jobs: u32,
    pub(super) mass_groups: u32,
    pub(super) e_in_length: u32,
    pub(super) e_out_length: u32,
    pub(super) e_in_log2: u32,
    pub(super) suffix_len: u32,
}

impl AtomMassPhaseParams {
    pub(super) fn new(
        rows: usize,
        atoms: usize,
        mass_jobs: usize,
        mass_groups: usize,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<Self, InstructionReadRafV3Error> {
        validate_weight_shape(rows, e_in_length, e_out_length)?;
        if atoms == 0 || atoms > rows || mass_jobs < atoms || mass_groups == 0 {
            return Err(InstructionReadRafV3Error::InvalidShaderAbi(
                "atom mass geometry is inconsistent",
            ));
        }
        if mass_groups > mass_jobs {
            return Err(InstructionReadRafV3Error::InvalidShaderAbi(
                "atom mass groups exceed jobs",
            ));
        }
        Ok(Self {
            rows: shader_u32(rows, "atom mass rows")?,
            atoms: shader_u32(atoms, "atom count")?,
            mass_jobs: shader_u32(mass_jobs, "atom mass jobs")?,
            mass_groups: shader_u32(mass_groups, "atom mass groups")?,
            e_in_length: shader_u32(e_in_length, "atom E_in length")?,
            e_out_length: shader_u32(e_out_length, "atom E_out length")?,
            e_in_log2: e_in_length.ilog2(),
            suffix_len: 120,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct AtomMassFinalizeParams {
    pub(super) atoms: u32,
    pub(super) split_atoms: u32,
    pub(super) mass_partials: u32,
    pub(super) reserved: u32,
}

impl AtomMassFinalizeParams {
    pub(super) fn new(
        atoms: usize,
        split_atoms: usize,
        mass_partials: usize,
    ) -> Result<Self, InstructionReadRafV3Error> {
        if atoms == 0
            || split_atoms > atoms
            || (split_atoms == 0) != (mass_partials == 0)
            || (split_atoms != 0 && mass_partials < 2 * split_atoms)
        {
            return Err(InstructionReadRafV3Error::InvalidShaderAbi(
                "split atom counts are inconsistent",
            ));
        }
        Ok(Self {
            atoms: shader_u32(atoms, "atom mass finalize atoms")?,
            split_atoms: shader_u32(split_atoms, "split atom count")?,
            mass_partials: shader_u32(mass_partials, "mass partial count")?,
            reserved: 0,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct FlagOpeningParams {
    pub(super) rows: u32,
    pub(super) e_in_length: u32,
    pub(super) e_out_length: u32,
    pub(super) columns: u32,
}

impl FlagOpeningParams {
    pub(super) fn new(
        rows: usize,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<Self, InstructionReadRafV3Error> {
        validate_weight_shape(rows, e_in_length, e_out_length)?;
        Ok(Self {
            rows: shader_u32(rows, "flag rows")?,
            e_in_length: shader_u32(e_in_length, "flag E_in length")?,
            e_out_length: shader_u32(e_out_length, "flag E_out length")?,
            columns: FLAG_COLUMNS as u32,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ReductionParams {
    pub(super) input_count: u32,
    pub(super) output_count: u32,
    pub(super) columns: u32,
    pub(super) reserved: u32,
}

impl ReductionParams {
    pub(super) fn new(input_count: usize) -> Result<Self, InstructionReadRafV3Error> {
        if input_count == 0 {
            return Err(InstructionReadRafV3Error::InvalidShaderAbi(
                "reduction input is empty",
            ));
        }
        Ok(Self {
            input_count: shader_u32(input_count, "reduction input")?,
            output_count: shader_u32(input_count.div_ceil(SIMD_WIDTH), "reduction output")?,
            columns: FLAG_COLUMNS as u32,
            reserved: 0,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SuffixPlan {
    explicit_kinds: [u8; TABLES * EXPLICIT_SUFFIX_LANES],
    explicit_counts: [u8; TABLES],
    output_lanes: [u8; TABLES * MAX_SUFFIXES],
    descriptors: [TableDescriptor; TABLES],
}

impl SuffixPlan {
    pub(crate) fn production() -> Result<Self, InstructionReadRafV3Error> {
        let mut plan = Self {
            explicit_kinds: [0; TABLES * EXPLICIT_SUFFIX_LANES],
            explicit_counts: [0; TABLES],
            output_lanes: [0; TABLES * MAX_SUFFIXES],
            descriptors: [TableDescriptor::default(); TABLES],
        };
        let mut output_start = 0usize;
        for table in LookupTableKind::<RISCV_XLEN>::iter() {
            let index = table.index();
            let suffixes = table.suffixes();
            if index >= TABLES || suffixes.len() > MAX_SUFFIXES {
                return Err(InstructionReadRafV3Error::InvalidShaderAbi(
                    "lookup-table suffix count exceeds the shader ABI",
                ));
            }
            let mut explicit = 0usize;
            for (slot, suffix) in suffixes.iter().copied().enumerate() {
                let lane = if suffix == Suffixes::One {
                    0
                } else {
                    if explicit == EXPLICIT_SUFFIX_LANES {
                        return Err(InstructionReadRafV3Error::InvalidShaderAbi(
                            "lookup table has too many explicit suffixes",
                        ));
                    }
                    plan.explicit_kinds[index * EXPLICIT_SUFFIX_LANES + explicit] = suffix as u8;
                    let lane = RAF_LANES + explicit;
                    explicit += 1;
                    lane
                };
                plan.output_lanes[index * MAX_SUFFIXES + slot] = lane as u8;
            }
            plan.explicit_counts[index] = explicit as u8;
            plan.descriptors[index] = TableDescriptor {
                output_start: shader_u32(output_start, "suffix output start")?,
                suffix_count: shader_u32(suffixes.len(), "suffix count")?,
                segment_raf_zero: shader_u32(2 * (index + 1), "RAF-zero segment")?,
                segment_raf_one: shader_u32(2 * (index + 1) + 1, "RAF-one segment")?,
            };
            output_start = output_start.checked_add(suffixes.len()).ok_or(
                InstructionReadRafV3Error::SizeOverflow("suffix output count"),
            )?;
        }
        if output_start != TOTAL_SUFFIXES {
            return Err(InstructionReadRafV3Error::InvalidShaderAbi(
                "production suffix topology changed",
            ));
        }
        Ok(plan)
    }

    pub(crate) const fn explicit_kinds(&self) -> &[u8; TABLES * EXPLICIT_SUFFIX_LANES] {
        &self.explicit_kinds
    }

    pub(crate) const fn explicit_counts(&self) -> &[u8; TABLES] {
        &self.explicit_counts
    }

    pub(crate) const fn output_lanes(&self) -> &[u8; TABLES * MAX_SUFFIXES] {
        &self.output_lanes
    }

    pub(crate) const fn descriptors(&self) -> &[TableDescriptor; TABLES] {
        &self.descriptors
    }
}

pub(super) fn pack_claim(
    table: Option<usize>,
    raf_flag: bool,
) -> Result<u8, InstructionReadRafV3Error> {
    if table.is_some_and(|index| index >= TABLES) {
        return Err(InstructionReadRafV3Error::InvalidTable(
            table.unwrap_or(TABLES),
        ));
    }
    let table_plus_one = table.map_or(0, |index| index + 1);
    Ok(table_plus_one as u8 | if raf_flag { 0x80 } else { 0 })
}

pub(super) fn segment_index(
    table: Option<usize>,
    raf_flag: bool,
) -> Result<usize, InstructionReadRafV3Error> {
    Ok(2 * usize::from(pack_claim(table, raf_flag)? & 0x7f) + usize::from(raf_flag))
}

fn validate_phase(suffix_len: usize, condense: bool) -> Result<(), InstructionReadRafV3Error> {
    if suffix_len > 120 || !suffix_len.is_multiple_of(ADDRESS_PHASE_BITS) {
        return Err(InstructionReadRafV3Error::InvalidShaderAbi(
            "address suffix length is invalid",
        ));
    }
    if condense && suffix_len > 112 {
        return Err(InstructionReadRafV3Error::InvalidShaderAbi(
            "condensation phase has no preceding byte",
        ));
    }
    Ok(())
}

fn validate_weight_shape(
    rows: usize,
    e_in: usize,
    e_out: usize,
) -> Result<(), InstructionReadRafV3Error> {
    if rows < 2
        || !rows.is_power_of_two()
        || e_in == 0
        || !e_in.is_power_of_two()
        || e_out == 0
        || !e_out.is_power_of_two()
        || e_in.checked_mul(e_out) != Some(rows)
    {
        return Err(InstructionReadRafV3Error::InvalidShaderAbi(
            "split equality shape does not cover the rows",
        ));
    }
    Ok(())
}

fn shader_u32(value: usize, name: &'static str) -> Result<u32, InstructionReadRafV3Error> {
    u32::try_from(value).map_err(|_| InstructionReadRafV3Error::SizeOverflow(name))
}

const _: [(); 16] = [(); size_of::<AddressLookup>()];
const _: [(); 16] = [(); size_of::<AddressJob>()];
const _: [(); 16] = [(); size_of::<AtomMassJob>()];
const _: [(); 16] = [(); size_of::<AtomMassGroup>()];
const _: [(); 16] = [(); size_of::<SplitAtom>()];
const _: [(); 16] = [(); size_of::<TableDescriptor>()];
const _: [(); 16] = [(); size_of::<AtomPhaseParams>()];
const _: [(); 32] = [(); size_of::<AtomMassPhaseParams>()];
const _: [(); 16] = [(); size_of::<AtomMassFinalizeParams>()];
const _: [(); 16] = [(); size_of::<FlagOpeningParams>()];
const _: [(); 16] = [(); size_of::<ReductionParams>()];

const _: () = assert!(ADDRESS_PHASES == 16);
