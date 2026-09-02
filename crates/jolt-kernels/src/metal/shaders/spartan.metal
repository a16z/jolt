// Spartan outer device package. The record lanes are the optimized backend's
// canonical per-cycle values; only the exact integer row arithmetic and Fr
// reductions move to the GPU.

#define JK_OUTER_DOMAIN 10u
#define JK_OUTER_SECOND 9u

struct JkI192 {
    ulong w0;
    ulong w1;
    ulong w2;
};

inline JkI192 jk_i192_add(JkI192 a, JkI192 b) {
    JkI192 r;
    r.w0 = a.w0 + b.w0;
    ulong carry0 = r.w0 < a.w0 ? 1ul : 0ul;
    ulong t1 = a.w1 + b.w1;
    ulong carry1a = t1 < a.w1 ? 1ul : 0ul;
    r.w1 = t1 + carry0;
    ulong carry1b = r.w1 < t1 ? 1ul : 0ul;
    r.w2 = a.w2 + b.w2 + carry1a + carry1b;
    return r;
}
inline JkI192 jk_i192_neg(JkI192 a) {
    return jk_i192_add(JkI192{~a.w0, ~a.w1, ~a.w2}, JkI192{1ul, 0ul, 0ul});
}

inline JkI192 jk_i192_sub(JkI192 a, JkI192 b) {
    return jk_i192_add(a, jk_i192_neg(b));
}

inline JkI192 jk_i192_u64(ulong value) {
    return JkI192{value, 0ul, 0ul};
}

inline JkI192 jk_i192_i64(long value) {
    ulong ext = value < 0 ? ~0ul : 0ul;
    return JkI192{(ulong)value, ext, ext};
}

inline JkI192 jk_i192_u128(device const uint* words) {
    return JkI192{
        (ulong)words[0] | ((ulong)words[1] << 32),
        (ulong)words[2] | ((ulong)words[3] << 32),
        0ul,
    };
}

inline JkI192 jk_i192_i128(device const uint* words) {
    ulong hi = (ulong)words[2] | ((ulong)words[3] << 32);
    return JkI192{
        (ulong)words[0] | ((ulong)words[1] << 32),
        hi,
        (hi >> 63) != 0ul ? ~0ul : 0ul,
    };
}

inline bool jk_i192_negative(JkI192 value) {
    return (value.w2 >> 63) != 0ul;
}

inline Fr256 jk_fr_from_i192(JkI192 value) {
    bool negative = jk_i192_negative(value);
    if (negative) {
        value = jk_i192_neg(value);
    }
    Fr256 raw = fr_zero();
    raw.v[0] = (uint)value.w0;
    raw.v[1] = (uint)(value.w0 >> 32);
    raw.v[2] = (uint)value.w1;
    raw.v[3] = (uint)(value.w1 >> 32);
    raw.v[4] = (uint)value.w2;
    raw.v[5] = (uint)(value.w2 >> 32);
    Fr256 out = fr_mont_mul(raw, fr_load_const(FR_R2, 0));
    return negative ? fr_sub(fr_zero(), out) : out;
}

inline void jk_mul64_wide(ulong a, ulong b, thread ulong& lo, thread ulong& hi) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;
    ulong middle = (p00 >> 32) + (ulong)(uint)p01 + (ulong)(uint)p10;
    lo = (p00 & 0xfffffffful) | (middle << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (middle >> 32);
}

inline JkI192 jk_i192_mul_i64(long scalar, JkI192 value) {
    bool negative = scalar < 0;
    ulong magnitude = negative ? (ulong)(-scalar) : (ulong)scalar;
    if (jk_i192_negative(value)) {
        negative = !negative;
        value = jk_i192_neg(value);
    }
    ulong p0lo, p0hi, p1lo, p1hi, p2lo, p2hi;
    jk_mul64_wide(value.w0, magnitude, p0lo, p0hi);
    jk_mul64_wide(value.w1, magnitude, p1lo, p1hi);
    jk_mul64_wide(value.w2, magnitude, p2lo, p2hi);
    ulong w1 = p1lo + p0hi;
    ulong carry1 = w1 < p1lo ? 1ul : 0ul;
    ulong t2 = p2lo + p1hi;
    ulong carry2a = t2 < p2lo ? 1ul : 0ul;
    ulong w2 = t2 + carry1;
    ulong carry2b = w2 < t2 ? 1ul : 0ul;
    JkI192 out = JkI192{p0lo, w1, w2};
    // The truncated host twin ignores the fourth limb; retaining the carry
    // computation above keeps this valid if coefficient bounds widen.
    (void)p2hi;
    (void)carry2a;
    (void)carry2b;
    return negative ? jk_i192_neg(out) : out;
}

inline bool jk_outer_flag(uint flags, uint bit) {
    return ((flags >> bit) & 1u) != 0u;
}

struct JkOuterRow {
    ulong left_input;
    JkI192 right_input;
    JkI192 product;
    ulong pc;
    ulong upc;
    JkI192 imm;
    ulong ram_address;
    ulong rs1;
    ulong rs2;
    ulong rd_write;
    ulong ram_read;
    ulong ram_write;
    ulong left_lookup;
    JkI192 right_lookup;
    ulong next_upc;
    ulong next_pc;
    bool next_is_virtual;
    bool next_is_first;
    ulong lookup_output;
    uint flags;
};

inline JkOuterRow jk_outer_load(
    uint j,
    uint len,
    device const ulong* pc,
    device const ulong* upc,
    device const uint* imm,
    device const ulong* rs1,
    device const ulong* rs2,
    device const ulong* rd_write,
    device const ulong* ram_address,
    device const ulong* ram_read,
    device const ulong* ram_write,
    device const ulong* left_lookup,
    device const uint* right_lookup,
    device const ulong* left_input,
    device const uint* right_input,
    device const ulong* product_lo,
    device const ulong* product_hi,
    device const ulong* lookup_output,
    device const uint* flags)
{
    JkOuterRow row;
    uint f = flags[j];
    row.left_input = left_input[j];
    row.right_input = jk_i192_i128(right_input + 4u * j);
    row.product = JkI192{product_lo[j], product_hi[j], 0ul};
    if (!jk_outer_flag(f, 27u)) {
        row.product = jk_i192_neg(row.product);
    }
    row.pc = pc[j];
    row.upc = upc[j];
    row.imm = jk_i192_i128(imm + 4u * j);
    row.ram_address = ram_address[j];
    row.rs1 = rs1[j];
    row.rs2 = rs2[j];
    row.rd_write = rd_write[j];
    row.ram_read = ram_read[j];
    row.ram_write = ram_write[j];
    row.left_lookup = left_lookup[j];
    row.right_lookup = jk_i192_u128(right_lookup + 4u * j);
    row.lookup_output = lookup_output[j];
    row.flags = f;
    if (j + 1u < len) {
        row.next_upc = upc[j + 1u];
        row.next_pc = pc[j + 1u];
        uint next_flags = flags[j + 1u];
        row.next_is_virtual = jk_outer_flag(next_flags, 7u);
        row.next_is_first = jk_outer_flag(next_flags, 12u);
    } else {
        row.next_upc = 0ul;
        row.next_pc = 0ul;
        row.next_is_virtual = false;
        row.next_is_first = false;
    }
    return row;
}

struct JkOuterGroups {
    long a_first[JK_OUTER_DOMAIN];
    long a_second[JK_OUTER_SECOND];
    JkI192 b_first[JK_OUTER_DOMAIN];
    JkI192 b_second[JK_OUTER_SECOND];
};

inline JkOuterGroups jk_outer_groups(JkOuterRow row) {
    JkOuterGroups g;
    uint f = row.flags;
    long load = jk_outer_flag(f, 3u) ? 1 : 0;
    long store = jk_outer_flag(f, 4u) ? 1 : 0;
    long add = jk_outer_flag(f, 0u) ? 1 : 0;
    long sub = jk_outer_flag(f, 1u) ? 1 : 0;
    long mul = jk_outer_flag(f, 2u) ? 1 : 0;
    long jump = jk_outer_flag(f, 5u) ? 1 : 0;
    long should_branch = jk_outer_flag(f, 24u) ? 1 : 0;

    g.a_first[0] = 1 - load - store;
    g.a_first[1] = load;
    g.a_first[2] = load;
    g.a_first[3] = store;
    g.a_first[4] = add + sub + mul;
    g.a_first[5] = 1 - add - sub - mul;
    g.a_first[6] = jk_outer_flag(f, 8u) ? 1 : 0;
    g.a_first[7] = jk_outer_flag(f, 25u) ? 1 : 0;
    g.a_first[8] = (jk_outer_flag(f, 7u) ? 1 : 0) - (jk_outer_flag(f, 13u) ? 1 : 0);
    g.a_first[9] = (row.next_is_virtual ? 1 : 0) - (row.next_is_first ? 1 : 0);

    g.a_second[0] = load + store;
    g.a_second[1] = add;
    g.a_second[2] = sub;
    g.a_second[3] = mul;
    g.a_second[4] = 1 - add - sub - mul - (jk_outer_flag(f, 10u) ? 1 : 0);
    g.a_second[5] = jk_outer_flag(f, 6u) ? 1 : 0;
    g.a_second[6] = jump;
    g.a_second[7] = should_branch;
    g.a_second[8] = 1 - should_branch - jump;

    g.b_first[0] = jk_i192_u64(row.ram_address);
    g.b_first[1] = jk_i192_sub(jk_i192_u64(row.ram_read), jk_i192_u64(row.ram_write));
    g.b_first[2] = jk_i192_sub(jk_i192_u64(row.ram_read), jk_i192_u64(row.rd_write));
    g.b_first[3] = jk_i192_sub(jk_i192_u64(row.rs2), jk_i192_u64(row.ram_write));
    g.b_first[4] = jk_i192_u64(row.left_lookup);
    g.b_first[5] = jk_i192_sub(jk_i192_u64(row.left_lookup), jk_i192_u64(row.left_input));
    g.b_first[6] = jk_i192_sub(jk_i192_u64(row.lookup_output), jk_i192_u64(1ul));
    g.b_first[7] = jk_i192_sub(jk_i192_u64(row.next_upc), jk_i192_u64(row.lookup_output));
    g.b_first[8] = jk_i192_sub(
        jk_i192_sub(jk_i192_u64(row.next_pc), jk_i192_u64(row.pc)), jk_i192_u64(1ul));
    g.b_first[9] = jk_i192_i64(1 - (jk_outer_flag(f, 9u) ? 1 : 0));

    JkI192 two_pow_64 = JkI192{0ul, 1ul, 0ul};
    long compressed = jk_outer_flag(f, 11u) ? 1 : 0;
    long dnupc = jk_outer_flag(f, 9u) ? 1 : 0;
    g.b_second[0] = jk_i192_sub(
        jk_i192_sub(jk_i192_u64(row.ram_address), jk_i192_u64(row.rs1)), row.imm);
    g.b_second[1] = jk_i192_sub(
        jk_i192_sub(row.right_lookup, jk_i192_u64(row.left_input)), row.right_input);
    g.b_second[2] = jk_i192_sub(
        jk_i192_add(jk_i192_sub(row.right_lookup, jk_i192_u64(row.left_input)), row.right_input),
        two_pow_64);
    g.b_second[3] = jk_i192_sub(row.right_lookup, row.product);
    g.b_second[4] = jk_i192_sub(row.right_lookup, row.right_input);
    g.b_second[5] = jk_i192_sub(jk_i192_u64(row.rd_write), jk_i192_u64(row.lookup_output));
    g.b_second[6] = jk_i192_add(
        jk_i192_sub(jk_i192_sub(jk_i192_u64(row.rd_write), jk_i192_u64(row.upc)), jk_i192_u64(4ul)),
        jk_i192_i64(2 * compressed));
    g.b_second[7] = jk_i192_sub(
        jk_i192_sub(jk_i192_u64(row.next_upc), jk_i192_u64(row.upc)), row.imm);
    g.b_second[8] = jk_i192_add(
        jk_i192_sub(jk_i192_u64(row.next_upc), jk_i192_add(jk_i192_u64(row.upc), jk_i192_u64(4ul))),
        jk_i192_i64(4 * dnupc + 2 * compressed));
    return g;
}

struct JkOuterPrepareParams {
    uint len;
    uint num_tgs;
    uint log_in;
};

#define JK_OUTER_RECORD_ARGS \
    device const ulong* pc [[buffer(0)]], \
    device const ulong* upc [[buffer(1)]], \
    device const uint* imm [[buffer(2)]], \
    device const ulong* rs1 [[buffer(3)]], \
    device const ulong* rs2 [[buffer(4)]], \
    device const ulong* rd_write [[buffer(5)]], \
    device const ulong* ram_address [[buffer(6)]], \
    device const ulong* ram_read [[buffer(7)]], \
    device const ulong* ram_write [[buffer(8)]], \
    device const ulong* left_lookup [[buffer(9)]], \
    device const uint* right_lookup [[buffer(10)]], \
    device const ulong* left_input [[buffer(11)]], \
    device const uint* right_input [[buffer(12)]], \
    device const ulong* product_lo [[buffer(13)]], \
    device const ulong* product_hi [[buffer(14)]], \
    device const ulong* lookup_output [[buffer(15)]], \
    device const uint* flags [[buffer(16)]]

#define JK_OUTER_LOAD_ROW(j, len) jk_outer_load( \
    j, len, pc, upc, imm, rs1, rs2, rd_write, ram_address, ram_read, ram_write, \
    left_lookup, right_lookup, left_input, right_input, product_lo, product_hi, \
    lookup_output, flags)

kernel void jk_outer_t1(
    JK_OUTER_RECORD_ARGS,
    device const uint* e_in [[buffer(17)]],
    device const uint* e_out [[buffer(18)]],
    device const long* coefficients [[buffer(19)]],
    device uint* partials [[buffer(20)]],
    constant JkOuterPrepareParams& p [[buffer(21)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.len;
    JkOuterGroups groups;
    Fr256 w0 = fr_zero();
    Fr256 w1 = fr_zero();
    if (active) {
        groups = jk_outer_groups(JK_OUTER_LOAD_ROW(gid, p.len));
        uint j2 = gid << 1;
        uint mask = (1u << p.log_in) - 1u;
        Fr256 eo = fr_load(e_out, j2 >> p.log_in);
        w0 = fr_mont_mul(eo, fr_load(e_in, j2 & mask));
        w1 = fr_mont_mul(eo, fr_load(e_in, (j2 + 1u) & mask));
    }
    for (uint node = 0u; node < 9u; node++) {
        Fr256 value = fr_zero();
        if (active) {
            long az0 = 0;
            long az1 = 0;
            JkI192 bz0 = JkI192{0ul, 0ul, 0ul};
            JkI192 bz1 = JkI192{0ul, 0ul, 0ul};
            for (uint i = 0u; i < JK_OUTER_DOMAIN; i++) {
                long c = coefficients[node * JK_OUTER_DOMAIN + i];
                az0 += c * groups.a_first[i];
                bz0 = jk_i192_add(bz0, jk_i192_mul_i64(c, groups.b_first[i]));
                if (i < JK_OUTER_SECOND) {
                    az1 += c * groups.a_second[i];
                    bz1 = jk_i192_add(bz1, jk_i192_mul_i64(c, groups.b_second[i]));
                }
            }
            Fr256 first = fr_mont_mul(jk_fr_from_i192(jk_i192_i64(az0)), jk_fr_from_i192(bz0));
            Fr256 second = fr_mont_mul(jk_fr_from_i192(jk_i192_i64(az1)), jk_fr_from_i192(bz1));
            value = fr_add(fr_mont_mul(w0, first), fr_mont_mul(w1, second));
        }
        jk_tg_sum(scratch, lid, tg, value, partials, node, p.num_tgs);
    }
}

kernel void jk_outer_azbz(
    JK_OUTER_RECORD_ARGS,
    device const uint* lagrange [[buffer(17)]],
    device const uint* e_in [[buffer(18)]],
    device const uint* e_out [[buffer(19)]],
    device uint* tables [[buffer(20)]],
    device uint* partials [[buffer(21)]],
    constant JkOuterPrepareParams& p [[buffer(22)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.len;
    Fr256 q0 = fr_zero();
    Fr256 qinf = fr_zero();
    if (active) {
        JkOuterGroups groups = jk_outer_groups(JK_OUTER_LOAD_ROW(gid, p.len));
        Fr256 az0 = fr_zero();
        Fr256 az1 = fr_zero();
        Fr256 bz0 = fr_zero();
        Fr256 bz1 = fr_zero();
        for (uint i = 0u; i < JK_OUTER_DOMAIN; i++) {
            Fr256 weight = fr_load(lagrange, i);
            az0 = fr_add(az0, fr_mont_mul(weight, jk_fr_from_i192(jk_i192_i64(groups.a_first[i]))));
            bz0 = fr_add(bz0, fr_mont_mul(weight, jk_fr_from_i192(groups.b_first[i])));
            if (i < JK_OUTER_SECOND) {
                az1 = fr_add(az1, fr_mont_mul(weight, jk_fr_from_i192(jk_i192_i64(groups.a_second[i]))));
                bz1 = fr_add(bz1, fr_mont_mul(weight, jk_fr_from_i192(groups.b_second[i])));
            }
        }
        fr_store(tables, 2u * gid, az0);
        fr_store(tables, 2u * gid + 1u, az1);
        uint table_len = 2u * p.len;
        fr_store(tables, table_len + 2u * gid, bz0);
        fr_store(tables, table_len + 2u * gid + 1u, bz1);
        uint mask = (1u << p.log_in) - 1u;
        Fr256 eq = fr_mont_mul(
            fr_load(e_out, gid >> p.log_in), fr_load(e_in, gid & mask));
        q0 = fr_mont_mul(eq, fr_mont_mul(az0, bz0));
        qinf = fr_mont_mul(eq, fr_mont_mul(fr_sub(az1, az0), fr_sub(bz1, bz0)));
    }
    jk_tg_sum(scratch, lid, tg, q0, partials, 0u, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, qinf, partials, 1u, p.num_tgs);
}

struct JkOuterRoundParams {
    uint groups;
    uint num_tgs;
    uint log_in;
    uint len;
    uint r[FR_LIMBS];
};

kernel void jk_outer_round(
    device const uint* cur [[buffer(0)]],
    device uint* nxt [[buffer(1)]],
    device const uint* e_in [[buffer(2)]],
    device const uint* e_out [[buffer(3)]],
    device uint* partials [[buffer(4)]],
    constant JkOuterRoundParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    Fr256 r = fr_load_const(p.r, 0);
    Fr256 az0, az1, bz0, bz1;
    jk_round_pair(cur, nxt, true, r, gid, active, az0, az1);
    jk_round_pair(
        cur + p.len * FR_LIMBS,
        nxt + (p.len >> 1) * FR_LIMBS,
        true,
        r,
        gid,
        active,
        bz0,
        bz1);
    Fr256 eq = fr_zero();
    if (active) {
        uint mask = (1u << p.log_in) - 1u;
        eq = fr_mont_mul(fr_load(e_out, gid >> p.log_in), fr_load(e_in, gid & mask));
    }
    Fr256 q0 = fr_mont_mul(eq, fr_mont_mul(az0, bz0));
    Fr256 qinf = fr_mont_mul(eq, fr_mont_mul(fr_sub(az1, az0), fr_sub(bz1, bz0)));
    jk_tg_sum(scratch, lid, tg, q0, partials, 0u, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, qinf, partials, 1u, p.num_tgs);
}

#define JK_OUTER_VARIABLES 35u
#define JK_OUTER_CLAIM_TILE 4u

struct JkOuterClaimsParams {
    uint len;
    uint in_len;
    uint out_len;
};

inline Fr256 jk_outer_bool(bool value) {
    Fr256 out;
    uint mask = value ? 0xffffffffu : 0u;
    for (uint limb = 0u; limb < FR_LIMBS; limb++) {
        out.v[limb] = FR_ONE[limb] & mask;
    }
    return out;
}

inline Fr256 jk_outer_claim_value(
    uint column,
    uint j,
    uint len,
    device const ulong* pc,
    device const ulong* upc,
    device const uint* imm,
    device const ulong* rs1,
    device const ulong* rs2,
    device const ulong* rd_write,
    device const ulong* ram_address,
    device const ulong* ram_read,
    device const ulong* ram_write,
    device const ulong* left_lookup,
    device const uint* right_lookup,
    device const ulong* left_input,
    device const uint* right_input,
    device const ulong* product_lo,
    device const ulong* product_hi,
    device const ulong* lookup_output,
    device const uint* flags)
{
    uint f = flags[j];
    switch (column) {
        case 0u: return jk_fr_from_i192(jk_i192_u64(left_input[j]));
        case 1u: return jk_fr_from_i192(jk_i192_i128(right_input + 4u * j));
        case 2u: {
            JkI192 product = JkI192{product_lo[j], product_hi[j], 0ul};
            return jk_fr_from_i192(jk_outer_flag(f, 27u) ? product : jk_i192_neg(product));
        }
        case 3u: return jk_outer_bool(jk_outer_flag(f, 24u));
        case 4u: return jk_fr_from_i192(jk_i192_u64(pc[j]));
        case 5u: return jk_fr_from_i192(jk_i192_u64(upc[j]));
        case 6u: return jk_fr_from_i192(jk_i192_i128(imm + 4u * j));
        case 7u: return jk_fr_from_i192(jk_i192_u64(ram_address[j]));
        case 8u: return jk_fr_from_i192(jk_i192_u64(rs1[j]));
        case 9u: return jk_fr_from_i192(jk_i192_u64(rs2[j]));
        case 10u: return jk_fr_from_i192(jk_i192_u64(rd_write[j]));
        case 11u: return jk_fr_from_i192(jk_i192_u64(ram_read[j]));
        case 12u: return jk_fr_from_i192(jk_i192_u64(ram_write[j]));
        case 13u: return jk_fr_from_i192(jk_i192_u64(left_lookup[j]));
        case 14u: return jk_fr_from_i192(jk_i192_u128(right_lookup + 4u * j));
        case 15u: return jk_fr_from_i192(jk_i192_u64(j + 1u < len ? upc[j + 1u] : 0ul));
        case 16u: return jk_fr_from_i192(jk_i192_u64(j + 1u < len ? pc[j + 1u] : 0ul));
        case 17u: return jk_outer_bool(j + 1u < len && jk_outer_flag(flags[j + 1u], 7u));
        case 18u: return jk_outer_bool(j + 1u < len && jk_outer_flag(flags[j + 1u], 12u));
        case 19u: return jk_fr_from_i192(jk_i192_u64(lookup_output[j]));
        case 20u: return jk_outer_bool(jk_outer_flag(f, 25u));
        default: return jk_outer_bool(jk_outer_flag(f, column - 21u));
    }
}

kernel void jk_outer_claims(
    JK_OUTER_RECORD_ARGS,
    device const uint* e_in [[buffer(17)]],
    device uint* partials [[buffer(18)]],
    constant JkOuterClaimsParams& p [[buffer(19)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    uint tile = tg / p.out_len;
    uint x_out = tg - tile * p.out_len;
    uint first_column = tile * JK_OUTER_CLAIM_TILE;
    Fr256 accumulators[JK_OUTER_CLAIM_TILE] = {
        fr_zero(), fr_zero(), fr_zero(), fr_zero()
    };
    for (uint x_in = lid; x_in < p.in_len; x_in += JK_TG_SIZE) {
        uint j = x_out * p.in_len + x_in;
        Fr256 weight = fr_load(e_in, x_in);
        for (uint offset = 0u; offset < JK_OUTER_CLAIM_TILE; offset++) {
            uint column = first_column + offset;
            if (column < JK_OUTER_VARIABLES) {
                Fr256 value = jk_outer_claim_value(
                    column, j, p.len, pc, upc, imm, rs1, rs2, rd_write,
                    ram_address, ram_read, ram_write, left_lookup, right_lookup,
                    left_input, right_input, product_lo, product_hi, lookup_output, flags);
                accumulators[offset] = fr_add(
                    accumulators[offset], fr_mont_mul(weight, value));
            }
        }
    }
    for (uint offset = 0u; offset < JK_OUTER_CLAIM_TILE; offset++) {
        uint column = first_column + offset;
        if (column < JK_OUTER_VARIABLES) {
            jk_tg_sum(
                scratch, lid, x_out, accumulators[offset], partials, column, p.out_len);
        }
    }
}

struct JkI256 {
    ulong m0;
    ulong m1;
    ulong m2;
    ulong m3;
    bool negative;
};

inline JkI256 jk_mul_i128_i192(JkI192 left, JkI192 right) {
    bool negative = jk_i192_negative(left);
    if (negative) {
        left = jk_i192_neg(left);
    }
    if (jk_i192_negative(right)) {
        negative = !negative;
        right = jk_i192_neg(right);
    }
    ulong r0, h00, p01, h01, p02, h02, p10, h10, p11, h11, p12, ignored;
    jk_mul64_wide(left.w0, right.w0, r0, h00);
    jk_mul64_wide(left.w0, right.w1, p01, h01);
    jk_mul64_wide(left.w0, right.w2, p02, h02);
    jk_mul64_wide(left.w1, right.w0, p10, h10);
    jk_mul64_wide(left.w1, right.w1, p11, h11);
    jk_mul64_wide(left.w1, right.w2, p12, ignored);
    ulong m1 = h00 + p01;
    ulong carry2 = m1 < h00 ? 1ul : 0ul;
    m1 += p10;
    carry2 += m1 < p10 ? 1ul : 0ul;
    ulong m2 = h01 + p02;
    ulong carry3 = m2 < h01 ? 1ul : 0ul;
    ulong t = h10 + p11;
    carry3 += t < h10 ? 1ul : 0ul;
    m2 += t;
    carry3 += m2 < t ? 1ul : 0ul;
    m2 += carry2;
    carry3 += m2 < carry2 ? 1ul : 0ul;
    return JkI256{r0, m1, m2, h02 + h11 + p12 + carry3, negative};
}

inline Fr256 jk_fr_from_i256(JkI256 value) {
    Fr256 raw = fr_zero();
    raw.v[0] = (uint)value.m0;
    raw.v[1] = (uint)(value.m0 >> 32);
    raw.v[2] = (uint)value.m1;
    raw.v[3] = (uint)(value.m1 >> 32);
    raw.v[4] = (uint)value.m2;
    raw.v[5] = (uint)(value.m2 >> 32);
    raw.v[6] = (uint)value.m3;
    raw.v[7] = (uint)(value.m3 >> 32);
    Fr256 out = fr_mont_mul(raw, fr_load_const(FR_R2, 0));
    return value.negative ? fr_sub(fr_zero(), out) : out;
}

kernel void jk_product_t1(
    device const ulong* left_input [[buffer(0)]],
    device const uint* right_input [[buffer(1)]],
    device const ulong* lookup_output [[buffer(2)]],
    device const uint* flags [[buffer(3)]],
    device const uint* e_in [[buffer(4)]],
    device const uint* e_out [[buffer(5)]],
    device const long* coefficients [[buffer(6)]],
    device uint* partials [[buffer(7)]],
    constant JkOuterPrepareParams& p [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.len;
    JkI192 left_lanes[3];
    JkI192 right_wide;
    long right_flags[2];
    Fr256 weight = fr_zero();
    if (active) {
        uint f = flags[gid];
        left_lanes[0] = jk_i192_u64(left_input[gid]);
        left_lanes[1] = jk_i192_u64(lookup_output[gid]);
        left_lanes[2] = jk_i192_u64(jk_outer_flag(f, 5u) ? 1ul : 0ul);
        right_wide = jk_i192_i128(right_input + 4u * gid);
        right_flags[0] = jk_outer_flag(f, 20u) ? 1 : 0;
        right_flags[1] = jk_outer_flag(f, 26u) ? 0 : 1;
        uint mask = (1u << p.log_in) - 1u;
        weight = fr_mont_mul(
            fr_load(e_out, gid >> p.log_in), fr_load(e_in, gid & mask));
    }
    for (uint node = 0u; node < 5u; node++) {
        Fr256 value = fr_zero();
        if (active) {
            device const long* c = coefficients + 3u * node;
            JkI192 left = JkI192{0ul, 0ul, 0ul};
            for (uint i = 0u; i < 3u; i++) {
                left = jk_i192_add(left, jk_i192_mul_i64(c[i], left_lanes[i]));
            }
            JkI192 right = jk_i192_mul_i64(c[0], right_wide);
            right = jk_i192_add(
                right, jk_i192_i64(c[1] * right_flags[0] + c[2] * right_flags[1]));
            value = fr_mont_mul(weight, jk_fr_from_i256(jk_mul_i128_i192(left, right)));
        }
        jk_tg_sum(scratch, lid, tg, value, partials, node, p.num_tgs);
    }
}

kernel void jk_product_lr(
    device const ulong* left_input [[buffer(0)]],
    device const uint* right_input [[buffer(1)]],
    device const ulong* lookup_output [[buffer(2)]],
    device const uint* flags [[buffer(3)]],
    device const uint* lagrange [[buffer(4)]],
    device const uint* e_in [[buffer(5)]],
    device const uint* e_out [[buffer(6)]],
    device uint* tables [[buffer(7)]],
    device uint* partials [[buffer(8)]],
    constant JkOuterPrepareParams& p [[buffer(9)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < (p.len >> 1);
    Fr256 left[2];
    Fr256 right[2];
    Fr256 q0 = fr_zero();
    Fr256 qinf = fr_zero();
    if (active) {
        for (uint side = 0u; side < 2u; side++) {
            uint j = 2u * gid + side;
            uint f = flags[j];
            left[side] = fr_add(
                fr_mont_mul(fr_load(lagrange, 0), jk_fr_from_i192(jk_i192_u64(left_input[j]))),
                fr_mont_mul(fr_load(lagrange, 1), jk_fr_from_i192(jk_i192_u64(lookup_output[j]))));
            if (jk_outer_flag(f, 5u)) {
                left[side] = fr_add(left[side], fr_load(lagrange, 2));
            }
            right[side] = fr_mont_mul(
                fr_load(lagrange, 0), jk_fr_from_i192(jk_i192_i128(right_input + 4u * j)));
            if (jk_outer_flag(f, 20u)) {
                right[side] = fr_add(right[side], fr_load(lagrange, 1));
            }
            if (!jk_outer_flag(f, 26u)) {
                right[side] = fr_add(right[side], fr_load(lagrange, 2));
            }
            fr_store(tables, j, left[side]);
            fr_store(tables, p.len + j, right[side]);
        }
        uint mask = (1u << p.log_in) - 1u;
        Fr256 eq = fr_mont_mul(
            fr_load(e_out, gid >> p.log_in), fr_load(e_in, gid & mask));
        q0 = fr_mont_mul(eq, fr_mont_mul(left[0], right[0]));
        qinf = fr_mont_mul(
            eq, fr_mont_mul(fr_sub(left[1], left[0]), fr_sub(right[1], right[0])));
    }
    jk_tg_sum(scratch, lid, tg, q0, partials, 0u, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, qinf, partials, 1u, p.num_tgs);
}

kernel void jk_icr_init(
    device const ulong* lookup_output [[buffer(0)]],
    device const ulong* left_lookup [[buffer(1)]],
    device const uint* right_lookup [[buffer(2)]],
    device const ulong* left_input [[buffer(3)]],
    device const uint* right_input [[buffer(4)]],
    device const uint* gamma [[buffer(5)]],
    device const uint* e_in [[buffer(6)]],
    device const uint* e_out [[buffer(7)]],
    device uint* combined [[buffer(8)]],
    device uint* partials [[buffer(9)]],
    constant JkOuterPrepareParams& p [[buffer(10)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < (p.len >> 1);
    Fr256 cells[2];
    Fr256 q0 = fr_zero();
    Fr256 q1 = fr_zero();
    Fr256 q2 = fr_zero();
    if (active) {
        for (uint side = 0u; side < 2u; side++) {
            uint j = 2u * gid + side;
            Fr256 value = fr_mont_mul(
                fr_load(gamma, 0), jk_fr_from_i192(jk_i192_u64(lookup_output[j])));
            value = fr_add(value, fr_mont_mul(
                fr_load(gamma, 1), jk_fr_from_i192(jk_i192_u64(left_lookup[j]))));
            value = fr_add(value, fr_mont_mul(
                fr_load(gamma, 2), jk_fr_from_i192(jk_i192_u128(right_lookup + 4u * j))));
            value = fr_add(value, fr_mont_mul(
                fr_load(gamma, 3), jk_fr_from_i192(jk_i192_u64(left_input[j]))));
            value = fr_add(value, fr_mont_mul(
                fr_load(gamma, 4), jk_fr_from_i192(jk_i192_i128(right_input + 4u * j))));
            cells[side] = value;
            fr_store(combined, j, value);
        }
        uint mask = (1u << p.log_in) - 1u;
        Fr256 eq = fr_mont_mul(
            fr_load(e_out, gid >> p.log_in), fr_load(e_in, gid & mask));
        q0 = fr_mont_mul(eq, cells[0]);
        q1 = fr_mont_mul(eq, cells[1]);
        q2 = fr_mont_mul(eq, fr_sub(fr_add(cells[1], cells[1]), cells[0]));
    }
    jk_tg_sum(scratch, lid, tg, q0, partials, 0u, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, q1, partials, 1u, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, q2, partials, 2u, p.num_tgs);
}

struct JkIcrRoundParams {
    uint groups;
    uint num_tgs;
    uint log_in;
    uint r[FR_LIMBS];
};

kernel void jk_icr_round(
    device const uint* cur [[buffer(0)]],
    device uint* nxt [[buffer(1)]],
    device const uint* e_in [[buffer(2)]],
    device const uint* e_out [[buffer(3)]],
    device uint* partials [[buffer(4)]],
    constant JkIcrRoundParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    Fr256 lo = fr_zero();
    Fr256 hi = fr_zero();
    Fr256 q0 = fr_zero();
    Fr256 q1 = fr_zero();
    Fr256 q2 = fr_zero();
    if (active) {
        Fr256 r = fr_load_const(p.r, 0);
        uint index = 4u * gid;
        Fr256 v0 = fr_load(cur, index);
        Fr256 v1 = fr_load(cur, index + 1u);
        Fr256 v2 = fr_load(cur, index + 2u);
        Fr256 v3 = fr_load(cur, index + 3u);
        lo = fr_add(v0, fr_mont_mul(r, fr_sub(v1, v0)));
        hi = fr_add(v2, fr_mont_mul(r, fr_sub(v3, v2)));
        fr_store(nxt, 2u * gid, lo);
        fr_store(nxt, 2u * gid + 1u, hi);
        uint mask = (1u << p.log_in) - 1u;
        Fr256 eq = fr_mont_mul(
            fr_load(e_out, gid >> p.log_in), fr_load(e_in, gid & mask));
        q0 = fr_mont_mul(eq, lo);
        q1 = fr_mont_mul(eq, hi);
        q2 = fr_mont_mul(eq, fr_sub(fr_add(hi, hi), lo));
    }
    jk_tg_sum(scratch, lid, tg, q0, partials, 0u, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, q1, partials, 1u, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, q2, partials, 2u, p.num_tgs);
}

// ---------------------------------------------------------------------------
// Lazy-form outer kernels (`JOLT_METAL_OUTER_LAZY=0` restores the eager
// twins above). Same values, fewer Montgomery multiplies: integer work stays
// in the integer domain (mirroring the host `extended_products` pipeline,
// bounds |az| < 2^22, |bz| < 2^152, product < 2^174), and raw residues meet
// their weight in ONE CIOS multiply instead of a convert-then-multiply pair.
// `fr_mont_mul(a, b)` is exact for any a < 2^256 with b < p (T < 2p), so a
// raw magnitude is a legal operand; the product w·v then sits in standard
// form, and one multiply by R (device R2 fix, or host `mont_form_fix`)
// returns a whole SUM to the canonical Montgomery representation.

// scalar·value for scalars below 2^31 (the extension coefficients cap at
// |c| = 140140 — pinned by `extension_coefficients_fit_i32`): six 32x32
// partial products against jk_i192_mul_i64's twelve.
inline JkI192 jk_i192_mul_i32(long scalar, JkI192 value) {
    bool negative = scalar < 0;
    ulong m = negative ? (ulong)(-scalar) : (ulong)scalar;
    if (jk_i192_negative(value)) {
        negative = !negative;
        value = jk_i192_neg(value);
    }
    ulong words[3] = {value.w0, value.w1, value.w2};
    ulong out[3];
    ulong carry = 0ul;
    for (uint w = 0u; w < 3u; w++) {
        ulong lo = (ulong)(uint)words[w] * m + carry;
        ulong hi = (words[w] >> 32) * m + (lo >> 32);
        out[w] = (lo & 0xfffffffful) | (hi << 32);
        carry = hi >> 32;
    }
    JkI192 result = JkI192{out[0], out[1], out[2]};
    return negative ? jk_i192_neg(result) : result;
}

// weight (Montgomery) x value (raw signed integer): the standard-form
// product w·v in one multiply.
inline Fr256 jk_fr_mul_raw_i192(Fr256 weight, JkI192 value) {
    bool negative = jk_i192_negative(value);
    if (negative) {
        value = jk_i192_neg(value);
    }
    Fr256 raw = fr_zero();
    raw.v[0] = (uint)value.w0;
    raw.v[1] = (uint)(value.w0 >> 32);
    raw.v[2] = (uint)value.w1;
    raw.v[3] = (uint)(value.w1 >> 32);
    raw.v[4] = (uint)value.w2;
    raw.v[5] = (uint)(value.w2 >> 32);
    Fr256 out = fr_mont_mul(weight, raw);
    return negative ? fr_sub(fr_zero(), out) : out;
}

// acc += a·weight for the small guard integers of the outer constraint
// groups: ±1 fold as one masked add/sub (production flag combinations never
// leave {-1, 0, 1}); the generic arm keeps exactness for any i64.
inline Fr256 jk_fr_add_scaled_small(Fr256 acc, Fr256 weight, long a) {
    if (a == 1) {
        return fr_add(acc, weight);
    }
    if (a == -1) {
        return fr_sub(acc, weight);
    }
    if (a == 0) {
        return acc;
    }
    return fr_add(acc, fr_mont_mul(jk_fr_from_i192(jk_i192_i64(a)), weight));
}

kernel void jk_outer_t1_lazy(
    JK_OUTER_RECORD_ARGS,
    device const uint* e_in [[buffer(17)]],
    device const uint* e_out [[buffer(18)]],
    device const long* coefficients [[buffer(19)]],
    device uint* partials [[buffer(20)]],
    constant JkOuterPrepareParams& p [[buffer(21)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.len;
    JkOuterGroups groups;
    Fr256 w0 = fr_zero();
    Fr256 w1 = fr_zero();
    if (active) {
        groups = jk_outer_groups(JK_OUTER_LOAD_ROW(gid, p.len));
        uint j2 = gid << 1;
        uint mask = (1u << p.log_in) - 1u;
        Fr256 eo = fr_load(e_out, j2 >> p.log_in);
        w0 = fr_mont_mul(eo, fr_load(e_in, j2 & mask));
        w1 = fr_mont_mul(eo, fr_load(e_in, (j2 + 1u) & mask));
    }
    for (uint node = 0u; node < 9u; node++) {
        Fr256 value = fr_zero();
        if (active) {
            long az0 = 0;
            long az1 = 0;
            JkI192 bz0 = JkI192{0ul, 0ul, 0ul};
            JkI192 bz1 = JkI192{0ul, 0ul, 0ul};
            for (uint i = 0u; i < JK_OUTER_DOMAIN; i++) {
                long c = coefficients[node * JK_OUTER_DOMAIN + i];
                az0 += c * groups.a_first[i];
                bz0 = jk_i192_add(bz0, jk_i192_mul_i32(c, groups.b_first[i]));
                if (i < JK_OUTER_SECOND) {
                    az1 += c * groups.a_second[i];
                    bz1 = jk_i192_add(bz1, jk_i192_mul_i32(c, groups.b_second[i]));
                }
            }
            JkI256 first = jk_mul_i128_i192(jk_i192_i64(az0), bz0);
            JkI256 second = jk_mul_i128_i192(jk_i192_i64(az1), bz1);
            Fr256 raw = fr_zero();
            raw.v[0] = (uint)first.m0;
            raw.v[1] = (uint)(first.m0 >> 32);
            raw.v[2] = (uint)first.m1;
            raw.v[3] = (uint)(first.m1 >> 32);
            raw.v[4] = (uint)first.m2;
            raw.v[5] = (uint)(first.m2 >> 32);
            raw.v[6] = (uint)first.m3;
            raw.v[7] = (uint)(first.m3 >> 32);
            Fr256 v0 = fr_mont_mul(w0, raw);
            v0 = first.negative ? fr_sub(fr_zero(), v0) : v0;
            raw.v[0] = (uint)second.m0;
            raw.v[1] = (uint)(second.m0 >> 32);
            raw.v[2] = (uint)second.m1;
            raw.v[3] = (uint)(second.m1 >> 32);
            raw.v[4] = (uint)second.m2;
            raw.v[5] = (uint)(second.m2 >> 32);
            raw.v[6] = (uint)second.m3;
            raw.v[7] = (uint)(second.m3 >> 32);
            Fr256 v1 = fr_mont_mul(w1, raw);
            v1 = second.negative ? fr_sub(fr_zero(), v1) : v1;
            value = fr_add(v0, v1);
        }
        jk_tg_sum(scratch, lid, tg, value, partials, node, p.num_tgs);
    }
}

kernel void jk_outer_azbz_lazy(
    JK_OUTER_RECORD_ARGS,
    device const uint* lagrange [[buffer(17)]],
    device const uint* e_in [[buffer(18)]],
    device const uint* e_out [[buffer(19)]],
    device uint* tables [[buffer(20)]],
    device uint* partials [[buffer(21)]],
    constant JkOuterPrepareParams& p [[buffer(22)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.len;
    Fr256 q0 = fr_zero();
    Fr256 qinf = fr_zero();
    if (active) {
        JkOuterGroups groups = jk_outer_groups(JK_OUTER_LOAD_ROW(gid, p.len));
        // az accumulates in Montgomery form (guard-masked weight folds);
        // bz accumulates raw-weight products in standard form and pays one
        // R2 multiply per stream before the store.
        Fr256 az0 = fr_zero();
        Fr256 az1 = fr_zero();
        Fr256 bz0 = fr_zero();
        Fr256 bz1 = fr_zero();
        for (uint i = 0u; i < JK_OUTER_DOMAIN; i++) {
            Fr256 weight = fr_load(lagrange, i);
            az0 = jk_fr_add_scaled_small(az0, weight, groups.a_first[i]);
            bz0 = fr_add(bz0, jk_fr_mul_raw_i192(weight, groups.b_first[i]));
            if (i < JK_OUTER_SECOND) {
                az1 = jk_fr_add_scaled_small(az1, weight, groups.a_second[i]);
                bz1 = fr_add(bz1, jk_fr_mul_raw_i192(weight, groups.b_second[i]));
            }
        }
        Fr256 r2 = fr_load_const(FR_R2, 0);
        bz0 = fr_mont_mul(bz0, r2);
        bz1 = fr_mont_mul(bz1, r2);
        fr_store(tables, 2u * gid, az0);
        fr_store(tables, 2u * gid + 1u, az1);
        uint table_len = 2u * p.len;
        fr_store(tables, table_len + 2u * gid, bz0);
        fr_store(tables, table_len + 2u * gid + 1u, bz1);
        uint mask = (1u << p.log_in) - 1u;
        Fr256 eq = fr_mont_mul(
            fr_load(e_out, gid >> p.log_in), fr_load(e_in, gid & mask));
        q0 = fr_mont_mul(eq, fr_mont_mul(az0, bz0));
        qinf = fr_mont_mul(eq, fr_mont_mul(fr_sub(az1, az0), fr_sub(bz1, bz0)));
    }
    jk_tg_sum(scratch, lid, tg, q0, partials, 0u, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, qinf, partials, 1u, p.num_tgs);
}

// Boolean opening columns (their partial stays Montgomery: masked weight
// adds); every other column is a raw integer met by one weight multiply
// (standard-form partial — the host applies the R fix per column).
inline bool jk_outer_claim_is_bool(uint column) {
    return column == 3u || column == 17u || column == 18u || column >= 20u;
}

inline bool jk_outer_claim_flag(
    uint column, uint j, uint len, device const uint* flags)
{
    switch (column) {
        case 3u: return jk_outer_flag(flags[j], 24u);
        case 17u: return j + 1u < len && jk_outer_flag(flags[j + 1u], 7u);
        case 18u: return j + 1u < len && jk_outer_flag(flags[j + 1u], 12u);
        case 20u: return jk_outer_flag(flags[j], 25u);
        default: return jk_outer_flag(flags[j], column - 21u);
    }
}

inline JkI192 jk_outer_claim_int(
    uint column,
    uint j,
    uint len,
    device const ulong* pc,
    device const ulong* upc,
    device const uint* imm,
    device const ulong* rs1,
    device const ulong* rs2,
    device const ulong* rd_write,
    device const ulong* ram_address,
    device const ulong* ram_read,
    device const ulong* ram_write,
    device const ulong* left_lookup,
    device const uint* right_lookup,
    device const ulong* left_input,
    device const uint* right_input,
    device const ulong* product_lo,
    device const ulong* product_hi,
    device const ulong* lookup_output)
{
    switch (column) {
        case 0u: return jk_i192_u64(left_input[j]);
        case 1u: return jk_i192_i128(right_input + 4u * j);
        case 2u: return JkI192{product_lo[j], product_hi[j], 0ul};
        case 4u: return jk_i192_u64(pc[j]);
        case 5u: return jk_i192_u64(upc[j]);
        case 6u: return jk_i192_i128(imm + 4u * j);
        case 7u: return jk_i192_u64(ram_address[j]);
        case 8u: return jk_i192_u64(rs1[j]);
        case 9u: return jk_i192_u64(rs2[j]);
        case 10u: return jk_i192_u64(rd_write[j]);
        case 11u: return jk_i192_u64(ram_read[j]);
        case 12u: return jk_i192_u64(ram_write[j]);
        case 13u: return jk_i192_u64(left_lookup[j]);
        case 14u: return jk_i192_u128(right_lookup + 4u * j);
        case 15u: return jk_i192_u64(j + 1u < len ? upc[j + 1u] : 0ul);
        case 16u: return jk_i192_u64(j + 1u < len ? pc[j + 1u] : 0ul);
        default: return jk_i192_u64(lookup_output[j]);
    }
}

kernel void jk_outer_claims_lazy(
    JK_OUTER_RECORD_ARGS,
    device const uint* e_in [[buffer(17)]],
    device uint* partials [[buffer(18)]],
    constant JkOuterClaimsParams& p [[buffer(19)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    uint tile = tg / p.out_len;
    uint x_out = tg - tile * p.out_len;
    uint first_column = tile * JK_OUTER_CLAIM_TILE;
    Fr256 accumulators[JK_OUTER_CLAIM_TILE] = {
        fr_zero(), fr_zero(), fr_zero(), fr_zero()
    };
    for (uint x_in = lid; x_in < p.in_len; x_in += JK_TG_SIZE) {
        uint j = x_out * p.in_len + x_in;
        Fr256 weight = fr_load(e_in, x_in);
        for (uint offset = 0u; offset < JK_OUTER_CLAIM_TILE; offset++) {
            uint column = first_column + offset;
            if (column >= JK_OUTER_VARIABLES) {
                continue;
            }
            if (jk_outer_claim_is_bool(column)) {
                if (jk_outer_claim_flag(column, j, p.len, flags)) {
                    accumulators[offset] = fr_add(accumulators[offset], weight);
                }
                continue;
            }
            JkI192 value = jk_outer_claim_int(
                column, j, p.len, pc, upc, imm, rs1, rs2, rd_write,
                ram_address, ram_read, ram_write, left_lookup, right_lookup,
                left_input, right_input, product_lo, product_hi, lookup_output);
            if (column == 2u && !jk_outer_flag(flags[j], 27u)) {
                value = jk_i192_neg(value);
            }
            accumulators[offset] = fr_add(
                accumulators[offset], jk_fr_mul_raw_i192(weight, value));
        }
    }
    for (uint offset = 0u; offset < JK_OUTER_CLAIM_TILE; offset++) {
        uint column = first_column + offset;
        if (column < JK_OUTER_VARIABLES) {
            jk_tg_sum(
                scratch, lid, x_out, accumulators[offset], partials, column, p.out_len);
        }
    }
}
