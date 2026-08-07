#define INSTRUCTION_READ_RAF_V3_BINS 256u
#define INSTRUCTION_READ_RAF_V3_TABLES 40u
#define INSTRUCTION_READ_RAF_V3_TABLE_VALUES 41u
#define INSTRUCTION_READ_RAF_V3_RAF_VALUES 2u
#define INSTRUCTION_READ_RAF_V3_SEGMENTS 82u
#define INSTRUCTION_READ_RAF_V3_RAF_LANES 3u
#define INSTRUCTION_READ_RAF_V3_EXPLICIT_SUFFIX_LANES 3u
#define INSTRUCTION_READ_RAF_V3_MAX_SUFFIXES 4u
#define INSTRUCTION_READ_RAF_V3_JOB_LANES 6u
#define INSTRUCTION_READ_RAF_V3_JOB_FIELDS 1536u
#define INSTRUCTION_READ_RAF_V3_FLAG_COLUMNS 41u
#define INSTRUCTION_READ_RAF_V3_WORDS 5u

struct InstructionReadRafV3Lookup {
    ulong2 limbs;
};

struct InstructionReadRafV3Job {
    uint start;
    uint end;
    uint segment;
    uint reserved;
};

struct InstructionReadRafV3AtomMassJob {
    uint cycle_start;
    uint cycle_end;
    uint atom;
    uint mass_partial_plus_one;
};

struct InstructionReadRafV3AtomMassGroup {
    uint job_start;
    uint job_end;
    uint segment;
    uint reserved;
};

struct InstructionReadRafV3SplitAtom {
    uint atom;
    uint partial_start;
    uint partial_end;
    uint reserved;
};

struct InstructionReadRafV3Table {
    uint output_start;
    uint suffix_count;
    uint segment_raf_zero;
    uint segment_raf_one;
};

struct InstructionReadRafV3WeightParams {
    uint rows;
    uint e_in_length;
    uint e_out_length;
    uint e_in_log2;
};

struct InstructionReadRafV3PhaseParams {
    uint suffix_len;
    uint job_count;
    uint condense;
    uint reserved;
};

struct InstructionReadRafV3AtomPhaseParams {
    uint suffix_len;
    uint job_count;
    uint reserved_0;
    uint reserved_1;
};

struct InstructionReadRafV3AtomMassPhaseParams {
    uint rows;
    uint atoms;
    uint mass_jobs;
    uint mass_groups;
    uint e_in_length;
    uint e_out_length;
    uint e_in_log2;
    uint suffix_len;
};

struct InstructionReadRafV3AtomMassFinalizeParams {
    uint atoms;
    uint split_atoms;
    uint mass_partials;
    uint reserved;
};

struct InstructionReadRafV3FlagParams {
    uint rows;
    uint e_in_length;
    uint e_out_length;
    uint columns;
};

struct InstructionReadRafV3ReductionParams {
    uint input_count;
    uint output_count;
    uint columns;
    uint reserved;
};

struct InstructionReadRafV3Bits {
    ulong lo;
    ulong hi;
    ulong x;
    ulong y;
    uint len;
    uint operand_len;
};

inline ulong instruction_read_raf_v3_mask(uint bits) {
    return bits == 0u ? 0ul : (bits >= 64u ? ~0ul : ((1ul << bits) - 1ul));
}

inline ulong instruction_read_raf_v3_compact_even_bits(ulong value) {
    value &= 0x5555555555555555ul;
    value = (value | (value >> 1)) & 0x3333333333333333ul;
    value = (value | (value >> 2)) & 0x0f0f0f0f0f0f0f0ful;
    value = (value | (value >> 4)) & 0x00ff00ff00ff00fful;
    value = (value | (value >> 8)) & 0x0000ffff0000fffful;
    value = (value | (value >> 16)) & 0x00000000fffffffful;
    return value;
}

inline uint instruction_read_raf_v3_lookup_byte(
    InstructionReadRafV3Lookup lookup,
    uint shift)
{
    return shift < 64u
        ? (uint)(lookup.limbs[0] >> shift) & 0xffu
        : (uint)(lookup.limbs[1] >> (shift - 64u)) & 0xffu;
}

inline InstructionReadRafV3Bits instruction_read_raf_v3_bits(
    InstructionReadRafV3Lookup lookup,
    uint suffix_len)
{
    InstructionReadRafV3Bits bits;
    bits.len = suffix_len;
    bits.operand_len = suffix_len / 2u;
    bits.lo = 0ul;
    bits.hi = 0ul;
    if (suffix_len == 64u) {
        bits.lo = lookup.limbs[0];
    } else if (suffix_len > 64u) {
        bits.lo = lookup.limbs[0];
        bits.hi = lookup.limbs[1]
            & instruction_read_raf_v3_mask(suffix_len - 64u);
    } else if (suffix_len != 0u) {
        bits.lo = lookup.limbs[0]
            & instruction_read_raf_v3_mask(suffix_len);
    }
    bits.x = instruction_read_raf_v3_compact_even_bits(bits.lo >> 1)
        | (instruction_read_raf_v3_compact_even_bits(bits.hi >> 1) << 32);
    bits.y = instruction_read_raf_v3_compact_even_bits(bits.lo)
        | (instruction_read_raf_v3_compact_even_bits(bits.hi) << 32);
    return bits;
}

inline SolinasFp128 instruction_read_raf_v3_field_u64(ulong scalar) {
    SolinasFp128 value = solinas_zero();
    value.limb[0] = (uint)scalar;
    value.limb[1] = (uint)(scalar >> 32);
    return value;
}

inline SolinasFp128 instruction_read_raf_v3_field_u128(ulong lo, ulong hi) {
    SolinasFp128 value;
    value.limb = uint4((uint)lo, (uint)(lo >> 32), (uint)hi, (uint)(hi >> 32));
    return value;
}

inline uint instruction_read_raf_v3_trailing_zeros(ulong value, uint len) {
    return value == 0ul ? len : min((uint)ctz(value), len);
}

inline uint instruction_read_raf_v3_leading_ones(ulong value, uint len) {
    if (len == 0u) {
        return 0u;
    }
    ulong inverse = (~value) & instruction_read_raf_v3_mask(len);
    return inverse == 0ul ? len : (uint)clz(inverse << (64u - len));
}

inline ulong instruction_read_raf_v3_unbounded_shl(ulong value, uint shift) {
    return shift >= 64u ? 0ul : value << shift;
}

inline ulong instruction_read_raf_v3_unbounded_shr(ulong value, uint shift) {
    return shift >= 64u ? 0ul : value >> shift;
}

inline ulong instruction_read_raf_v3_rotate_right(ulong value, uint shift) {
    return (value >> shift) | (value << (64u - shift));
}

inline uint instruction_read_raf_v3_rotate_right_32(uint value, uint shift) {
    return (value >> shift) | (value << (32u - shift));
}

inline uint instruction_read_raf_v3_swap_bytes_32(uint value) {
    return ((value & 0x000000ffu) << 24)
        | ((value & 0x0000ff00u) << 8)
        | ((value & 0x00ff0000u) >> 8)
        | ((value & 0xff000000u) >> 24);
}

inline ulong instruction_read_raf_v3_suffix(
    uchar kind,
    InstructionReadRafV3Bits bits)
{
    ulong operand_mask = instruction_read_raf_v3_mask(bits.operand_len);
    switch (kind) {
        case 0: return 1ul;
        case 1: return bits.x & bits.y;
        case 2: return bits.x & ~bits.y;
        case 3: return bits.x ^ bits.y;
        case 4: return bits.x | bits.y;
        case 5: return bits.y;
        case 6: return (ulong)(uint)bits.y;
        case 7: return (ulong)(bits.x == 0ul && bits.y == operand_mask);
        case 8: {
            uint len = min(bits.operand_len, 32u);
            return (ulong)((uint)bits.x == 0u
                && (uint)bits.y == (uint)instruction_read_raf_v3_mask(len));
        }
        case 9: return bits.hi;
        case 10: return bits.lo;
        case 11: return (ulong)(uint)bits.lo;
        case 12: return (ulong)(bits.x < bits.y);
        case 13: return (ulong)(bits.x > bits.y);
        case 14: return (ulong)(bits.x == bits.y);
        case 15: return (ulong)(bits.x == 0ul);
        case 16: return (ulong)(bits.y == 0ul);
        case 17: return bits.len == 0u ? 1ul : bits.lo & 1ul;
        case 18: return (ulong)(bits.x == 0ul && bits.y == operand_mask);
        case 19: return bits.len == 0u ? 1ul : 1ul << (bits.lo & 63ul);
        case 20: return bits.len == 0u ? 1ul : 1ul << (bits.lo & 31ul);
        case 21: {
            ulong lo = (ulong)instruction_read_raf_v3_swap_bytes_32((uint)bits.lo);
            ulong hi = (ulong)instruction_read_raf_v3_swap_bytes_32((uint)(bits.lo >> 32));
            return lo | (hi << 32);
        }
        case 22: return bits.len == 0u ? 1ul : 1ul << (63u - (uint)(bits.lo & 63ul));
        case 23: return instruction_read_raf_v3_unbounded_shr(
            bits.x,
            instruction_read_raf_v3_trailing_zeros(bits.y, bits.operand_len));
        case 24: return 1ul << instruction_read_raf_v3_leading_ones(
            bits.y,
            bits.operand_len);
        case 25: {
            uint padding = instruction_read_raf_v3_trailing_zeros(
                bits.y,
                bits.operand_len);
            return padding == 0u ? 0ul : (~0ul << (64u - padding));
        }
        case 26: return instruction_read_raf_v3_unbounded_shl(
            bits.x & ~bits.y,
            instruction_read_raf_v3_leading_ones(bits.y, bits.operand_len));
        case 27: return (ulong)(bits.len == 0u || (bits.lo & 3ul) == 0ul);
        case 28: {
            if (bits.len < 32u) return 1ul;
            return ((bits.lo >> 31) & 1ul) != 0ul ? 0xffffffff00000000ul : 0ul;
        }
        case 29: {
            if (bits.len < 64u) return 1ul;
            return ((bits.lo >> 62) & 1ul) != 0ul ? 0xffffffff00000000ul : 0ul;
        }
        case 30: {
            uint shift = min(
                instruction_read_raf_v3_trailing_zeros(bits.y, bits.operand_len),
                32u);
            return shift == 32u ? 0ul : (ulong)((uint)bits.x >> shift);
        }
        case 31: {
            uint len = min(bits.operand_len, 32u);
            return 1ul << instruction_read_raf_v3_leading_ones((uint)bits.y, len);
        }
        case 32: {
            uint leading = instruction_read_raf_v3_leading_ones(
                (uint)bits.y,
                bits.operand_len);
            return leading >= 32u ? 0ul : (ulong)(1u << leading);
        }
        case 33: {
            uint len = min(bits.operand_len, 32u);
            uint leading = instruction_read_raf_v3_leading_ones((uint)bits.y, len);
            uint value = (uint)bits.x & ~(uint)bits.y;
            return leading >= 32u ? 0ul : (ulong)(value << leading);
        }
        case 34: return (ulong)(bits.hi == 0ul);
        case 35: return instruction_read_raf_v3_rotate_right(bits.x ^ bits.y, 16u);
        case 36: return instruction_read_raf_v3_rotate_right(bits.x ^ bits.y, 24u);
        case 37: return instruction_read_raf_v3_rotate_right(bits.x ^ bits.y, 32u);
        case 38: return instruction_read_raf_v3_rotate_right(bits.x ^ bits.y, 63u);
        case 39: return (ulong)instruction_read_raf_v3_rotate_right_32(
            (uint)bits.x ^ (uint)bits.y,
            16u);
        case 40: return (ulong)instruction_read_raf_v3_rotate_right_32(
            (uint)bits.x ^ (uint)bits.y,
            12u);
        case 41: return (ulong)instruction_read_raf_v3_rotate_right_32(
            (uint)bits.x ^ (uint)bits.y,
            8u);
        case 42: return (ulong)instruction_read_raf_v3_rotate_right_32(
            (uint)bits.x ^ (uint)bits.y,
            7u);
        default: return 0ul;
    }
}

inline void instruction_read_raf_v3_accumulate(
    threadgroup atomic_uint* sums,
    InstructionReadRafV3Lookup lookup,
    uint segment,
    uint suffix_len,
    device const uchar* suffix_kinds,
    device const uchar* suffix_counts,
    SolinasFp128 weight)
{
    uint table_plus_one = segment / INSTRUCTION_READ_RAF_V3_RAF_VALUES;
    uint raf_flag = segment & 1u;
    uint chunk = instruction_read_raf_v3_lookup_byte(lookup, suffix_len);
    InstructionReadRafV3Bits bits = instruction_read_raf_v3_bits(
        lookup,
        suffix_len);

    solinas_deferred_atomic_add_5(sums, chunk, weight);
    if (raf_flag == 0u) {
        if (bits.x != 0ul) {
            solinas_deferred_atomic_add_5(
                sums,
                INSTRUCTION_READ_RAF_V3_BINS + chunk,
                solinas_mul_wide(
                    weight,
                    instruction_read_raf_v3_field_u64(bits.x)));
        }
        if (bits.y != 0ul) {
            solinas_deferred_atomic_add_5(
                sums,
                2u * INSTRUCTION_READ_RAF_V3_BINS + chunk,
                solinas_mul_wide(
                    weight,
                    instruction_read_raf_v3_field_u64(bits.y)));
        }
    } else {
        if (bits.lo != 0ul || bits.hi != 0ul) {
            solinas_deferred_atomic_add_5(
                sums,
                INSTRUCTION_READ_RAF_V3_BINS + chunk,
                solinas_mul_wide(
                    weight,
                    instruction_read_raf_v3_field_u128(bits.lo, bits.hi)));
        }
        uint upper_bits = suffix_len > 64u ? suffix_len - 64u : 0u;
        bool upper_all_ones = upper_bits == 0u
            || bits.hi == instruction_read_raf_v3_mask(upper_bits);
        if (upper_all_ones) {
            solinas_deferred_atomic_add_5(
                sums,
                2u * INSTRUCTION_READ_RAF_V3_BINS + chunk,
                weight);
        }
    }

    uint explicit_suffix_count = table_plus_one == 0u
        ? 0u
        : suffix_counts[table_plus_one - 1u];
    for (uint suffix = 0u; suffix < explicit_suffix_count; suffix++) {
        uchar kind = suffix_kinds[
            (table_plus_one - 1u)
                * INSTRUCTION_READ_RAF_V3_EXPLICIT_SUFFIX_LANES
                + suffix];
        ulong scalar = instruction_read_raf_v3_suffix(kind, bits);
        if (scalar != 0ul) {
            SolinasFp128 contribution = scalar == 1ul
                ? weight
                : solinas_mul_wide(
                    weight,
                    instruction_read_raf_v3_field_u64(scalar));
            uint lane = INSTRUCTION_READ_RAF_V3_RAF_LANES + suffix;
            solinas_deferred_atomic_add_5(
                sums,
                lane * INSTRUCTION_READ_RAF_V3_BINS + chunk,
                contribution);
        }
    }
}

kernel void solinas_instruction_read_raf_v3_atom_mass_phase(
    device const InstructionReadRafV3Lookup* atom_lookups [[buffer(0)]],
    device const InstructionReadRafV3AtomMassJob* mass_jobs [[buffer(1)]],
    device const InstructionReadRafV3AtomMassGroup* mass_groups [[buffer(2)]],
    device const uint* cycle_indices [[buffer(3)]],
    device const SolinasFp128* e_in [[buffer(4)]],
    device const SolinasFp128* e_out [[buffer(5)]],
    device SolinasFp128* atom_masses [[buffer(6)]],
    device SolinasFp128* mass_partials [[buffer(7)]],
    device const uchar* suffix_kinds [[buffer(8)]],
    device const uchar* suffix_counts [[buffer(9)]],
    device SolinasFp128* address_partials [[buffer(10)]],
    constant InstructionReadRafV3AtomMassPhaseParams& params [[buffer(11)]],
    threadgroup atomic_uint* sums [[threadgroup(0)]],
    uint group_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (group_index >= params.mass_groups) {
        return;
    }
    for (uint counter = tid;
         counter < INSTRUCTION_READ_RAF_V3_JOB_FIELDS
            * INSTRUCTION_READ_RAF_V3_WORDS;
         counter += threads) {
        atomic_store_explicit(&sums[counter], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    InstructionReadRafV3AtomMassGroup group = mass_groups[group_index];
    uint simdgroup = tid / 32u;
    uint simdgroups = threads / 32u;
    for (uint job_index = group.job_start + simdgroup;
         job_index < group.job_end;
         job_index += simdgroups) {
        InstructionReadRafV3AtomMassJob job = mass_jobs[job_index];
        SolinasFp128 mass = solinas_zero();
        for (uint cursor = job.cycle_start + lane;
             cursor < job.cycle_end;
             cursor += 32u) {
            uint cycle = cycle_indices[cursor];
            uint x_in = cycle & (params.e_in_length - 1u);
            uint x_out = cycle >> params.e_in_log2;
            mass = solinas_add(mass, solinas_mul_wide(e_out[x_out], e_in[x_in]));
        }
        mass = solinas_simd_sum_32(mass);
        if (lane == 0u) {
            if (job.mass_partial_plus_one == 0u) {
                atom_masses[job.atom] = mass;
            } else {
                mass_partials[job.mass_partial_plus_one - 1u] = mass;
            }
            instruction_read_raf_v3_accumulate(
                sums,
                atom_lookups[job.atom],
                group.segment,
                params.suffix_len,
                suffix_kinds,
                suffix_counts,
                mass);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint output_base = group_index * INSTRUCTION_READ_RAF_V3_JOB_FIELDS;
    for (uint field = tid; field < INSTRUCTION_READ_RAF_V3_JOB_FIELDS;
         field += threads) {
        address_partials[output_base + field] =
            solinas_deferred_atomic_reduce_5(sums, field);
    }
}

kernel void solinas_instruction_read_raf_v3_atom_mass_finalize(
    device const InstructionReadRafV3SplitAtom* split_atoms [[buffer(0)]],
    device const SolinasFp128* mass_partials [[buffer(1)]],
    device SolinasFp128* atom_masses [[buffer(2)]],
    constant InstructionReadRafV3AtomMassFinalizeParams& params [[buffer(3)]],
    uint split_index [[thread_position_in_grid]])
{
    if (split_index >= params.split_atoms) {
        return;
    }
    InstructionReadRafV3SplitAtom split = split_atoms[split_index];
    SolinasFp128 mass = solinas_zero();
    for (uint partial = split.partial_start; partial < split.partial_end; partial++) {
        mass = solinas_add(mass, mass_partials[partial]);
    }
    atom_masses[split.atom] = mass;
}


kernel void solinas_instruction_read_raf_v3_atom_phase(
    device const InstructionReadRafV3Lookup* atom_lookups [[buffer(0)]],
    device SolinasFp128* atom_masses [[buffer(1)]],
    device const SolinasFp128* previous_phase_table [[buffer(2)]],
    device const InstructionReadRafV3Job* jobs [[buffer(3)]],
    device const uchar* suffix_kinds [[buffer(4)]],
    device const uchar* suffix_counts [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant InstructionReadRafV3AtomPhaseParams& params [[buffer(7)]],
    threadgroup atomic_uint* sums [[threadgroup(0)]],
    uint job_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (job_index >= params.job_count) {
        return;
    }
    for (uint counter = tid;
         counter < INSTRUCTION_READ_RAF_V3_JOB_FIELDS
            * INSTRUCTION_READ_RAF_V3_WORDS;
         counter += threads) {
        atomic_store_explicit(&sums[counter], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    InstructionReadRafV3Job job = jobs[job_index];
    for (uint atom = job.start + tid; atom < job.end; atom += threads) {
        InstructionReadRafV3Lookup lookup = atom_lookups[atom];
        SolinasFp128 mass = atom_masses[atom];
        uint previous_chunk = instruction_read_raf_v3_lookup_byte(
            lookup,
            params.suffix_len + 8u);
        mass = solinas_mul_wide(mass, previous_phase_table[previous_chunk]);
        atom_masses[atom] = mass;
        instruction_read_raf_v3_accumulate(
            sums,
            lookup,
            job.segment,
            params.suffix_len,
            suffix_kinds,
            suffix_counts,
            mass);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint output_base = job_index * INSTRUCTION_READ_RAF_V3_JOB_FIELDS;
    for (uint field = tid; field < INSTRUCTION_READ_RAF_V3_JOB_FIELDS;
         field += threads) {
        partials[output_base + field] = solinas_deferred_atomic_reduce_5(sums, field);
    }
}

kernel void solinas_instruction_read_raf_v3_finalize_raf(
    device const SolinasFp128* partials [[buffer(0)]],
    device const uint* segment_job_offsets [[buffer(1)]],
    device SolinasFp128* output [[buffer(2)]],
    uint output_lane [[threadgroup_position_in_grid]],
    uint chunk [[thread_index_in_threadgroup]])
{
    if (output_lane >= 6u || chunk >= INSTRUCTION_READ_RAF_V3_BINS) {
        return;
    }
    uint raf_flag = output_lane / INSTRUCTION_READ_RAF_V3_RAF_LANES;
    uint local_lane = output_lane % INSTRUCTION_READ_RAF_V3_RAF_LANES;
    SolinasFp128 sum = solinas_zero();
    for (uint table_plus_one = 0u;
         table_plus_one < INSTRUCTION_READ_RAF_V3_TABLE_VALUES;
         table_plus_one++) {
        uint segment = table_plus_one * INSTRUCTION_READ_RAF_V3_RAF_VALUES
            + raf_flag;
        for (uint job = segment_job_offsets[segment];
             job < segment_job_offsets[segment + 1u];
             job++) {
            uint field = job * INSTRUCTION_READ_RAF_V3_JOB_FIELDS
                + local_lane * INSTRUCTION_READ_RAF_V3_BINS
                + chunk;
            sum = solinas_add(sum, partials[field]);
        }
    }
    output[output_lane * INSTRUCTION_READ_RAF_V3_BINS + chunk] = sum;
}

kernel void solinas_instruction_read_raf_v3_finalize_suffix(
    device const SolinasFp128* partials [[buffer(0)]],
    device const uint* segment_job_offsets [[buffer(1)]],
    device const InstructionReadRafV3Table* tables [[buffer(2)]],
    device const uchar* output_lanes [[buffer(3)]],
    device SolinasFp128* output [[buffer(4)]],
    uint table [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (table >= INSTRUCTION_READ_RAF_V3_TABLES) {
        return;
    }
    InstructionReadRafV3Table descriptor = tables[table];
    uint fields = descriptor.suffix_count * INSTRUCTION_READ_RAF_V3_BINS;
    if (tid >= fields) {
        return;
    }
    uint suffix = tid / INSTRUCTION_READ_RAF_V3_BINS;
    uint chunk = tid & (INSTRUCTION_READ_RAF_V3_BINS - 1u);
    uint local_lane = output_lanes[
        table * INSTRUCTION_READ_RAF_V3_MAX_SUFFIXES + suffix];
    SolinasFp128 sum = solinas_zero();
    for (uint raf_flag = 0u; raf_flag < 2u; raf_flag++) {
        uint segment = raf_flag == 0u
            ? descriptor.segment_raf_zero
            : descriptor.segment_raf_one;
        for (uint job = segment_job_offsets[segment];
             job < segment_job_offsets[segment + 1u];
             job++) {
            uint field = job * INSTRUCTION_READ_RAF_V3_JOB_FIELDS
                + local_lane * INSTRUCTION_READ_RAF_V3_BINS
                + chunk;
            sum = solinas_add(sum, partials[field]);
        }
    }
    output[(descriptor.output_start + suffix) * INSTRUCTION_READ_RAF_V3_BINS + chunk]
        = sum;
}

kernel void solinas_instruction_read_raf_v3_open_flags(
    device const uchar* claim_columns [[buffer(0)]],
    device const SolinasFp128* e_in [[buffer(1)]],
    device const SolinasFp128* e_out [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    constant InstructionReadRafV3FlagParams& params [[buffer(4)]],
    threadgroup atomic_uint* sums [[threadgroup(0)]],
    uint x_out [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (x_out >= params.e_out_length) {
        return;
    }
    for (uint counter = tid;
         counter < INSTRUCTION_READ_RAF_V3_FLAG_COLUMNS
            * INSTRUCTION_READ_RAF_V3_WORDS;
         counter += threads) {
        atomic_store_explicit(&sums[counter], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint block_start = x_out * params.e_in_length;
    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint row = block_start + x_in;
        if (row >= params.rows) {
            continue;
        }
        uchar packed = claim_columns[row];
        uint table_plus_one = (uint)packed & 0x7fu;
        if (table_plus_one != 0u
            && table_plus_one <= INSTRUCTION_READ_RAF_V3_TABLES) {
            solinas_deferred_atomic_add_5(
                sums,
                table_plus_one - 1u,
                e_in[x_in]);
        }
        if ((packed & 0x80u) != 0u) {
            solinas_deferred_atomic_add_5(
                sums,
                INSTRUCTION_READ_RAF_V3_TABLES,
                e_in[x_in]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint column = tid; column < params.columns; column += threads) {
        SolinasFp128 value = solinas_deferred_atomic_reduce_5(sums, column);
        partials[column * params.e_out_length + x_out] = solinas_mul_wide(
            e_out[x_out],
            value);
    }
}

kernel void solinas_instruction_read_raf_v3_reduce(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant InstructionReadRafV3ReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint column = 0u; column < params.columns; column++) {
        SolinasFp128 value = gid < params.input_count
            ? input[column * params.input_count + gid]
            : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane == 0u && gid / 32u < params.output_count) {
            output[column * params.output_count + gid / 32u] = value;
        }
    }
}
