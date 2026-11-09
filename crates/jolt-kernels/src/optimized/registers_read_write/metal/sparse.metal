struct RegisterSeed { ulong prev, next, meta; };

uint register_rank(uint4 mask, uint col) {
    uint word = col / 32, bit = col % 32, rank = 0;
    for (uint w = 0; w < word; w++) rank += metal::popcount(mask[w]);
    return rank + metal::popcount(mask[word] & ((1u << bit) - 1u));
}
bool register_present(uint4 mask, uint col) { return (mask[col / 32] & (1u << (col % 32))) != 0; }
uint register_next(thread uint4& mask) {
    uint word = 0;
    while (word < 4 && mask[word] == 0) word++;
    uint bit = 31u - metal::clz(mask[word] & (0u - mask[word]));
    mask[word] &= mask[word] - 1;
    return word * 32 + bit;
}
bool register_any(uint4 mask) { return metal::any(mask != uint4(0)); }

template <typename F> struct RegisterCell {
    bool present;
    F val, ra, wa;
    ulong prev, next;
    uint ra_code, wa_code;
};
template <typename F> RegisterCell<F> register_empty() {
    return {false, F::zero(), F::zero(), F::zero(), 0, 0, 0, 0};
}
template <typename F>
F register_raw_inc(device const ulong2* raw, uint count, uint row) {
    if (row >= count) return F::zero();
    ulong2 value = raw[row];
    // Signed high word times 2^64 plus the unsigned low word is the original i128.
    return F::from_u64(value.x) + F::from_i64(long(value.y)) * F{uint4(0, 0, 1, 0)};
}
template <typename F>
RegisterCell<F> register_seed_cell(device const RegisterSeed* entries, uint begin, uint end, uint row, uint col) {
    RegisterCell<F> cell = register_empty<F>();
    for (uint i = begin; i < end; i++) {
        RegisterSeed entry = entries[i];
        if (uint(entry.meta) == row && uint((entry.meta >> 48) & 255ul) == col) {
            cell.present = true;
            cell.prev = entry.prev; cell.next = entry.next;
            cell.val = F::from_u64(entry.prev);
            cell.ra_code = uint((entry.meta >> 32) & 255ul);
            cell.wa_code = uint((entry.meta >> 40) & 255ul);
            break;
        }
    }
    return cell;
}
template <typename F>
RegisterCell<F> register_bind_cell(RegisterCell<F> even, RegisterCell<F> odd, F r, uint ra_bits, uint wa_bits, bool direct) {
    if (!even.present && !odd.present) return register_empty<F>();
    RegisterCell<F> out;
    out.present = true;
    out.prev = even.present ? even.prev : odd.prev;
    out.next = odd.present ? odd.next : even.next;
    F low = even.present ? even.val : F::from_u64(odd.prev);
    F high = odd.present ? odd.val : F::from_u64(even.next);
    out.val = low + r * (high - low);
    out.ra_code = direct ? 0 : (odd.ra_code << ra_bits) | even.ra_code;
    out.wa_code = direct ? 0 : (odd.wa_code << wa_bits) | even.wa_code;
    out.ra = direct ? even.ra + r * (odd.ra - even.ra) : F::zero();
    out.wa = direct ? even.wa + r * (odd.wa - even.wa) : F::zero();
    return out;
}
template <typename F>
RegisterCell<F> register_seed_pair(device const RegisterSeed* entries, uint begin, uint end, uint row, uint col, F r) {
    return register_bind_cell(register_seed_cell<F>(entries, begin, end, row, col), register_seed_cell<F>(entries, begin, end, row + 1, col), r, 2, 1, false);
}
template <typename F>
RegisterCell<F> register_sparse_cell(
    device const F* values, device const ulong2* boundaries, device const uint* codes,
    device const F* ra, device const F* wa, device const uint4* masks, device const uint* offsets,
    device const F* ra_lut, device const F* wa_lut,
    uint rows, uint row, uint col, bool direct, bool resolve) {
    RegisterCell<F> cell = register_empty<F>();
    if (row >= rows || !register_present(masks[row], col)) return cell;
    uint index = offsets[row] + register_rank(masks[row], col);
    cell.present = true; cell.val = values[index];
    cell.prev = boundaries[index].x; cell.next = boundaries[index].y;
    if (direct) { cell.ra = ra[index]; cell.wa = wa[index]; }
    else {
        uint code = codes[index];
        cell.ra_code = code & 65535u; cell.wa_code = code >> 16;
        if (resolve) { cell.ra = ra_lut[cell.ra_code]; cell.wa = wa_lut[cell.wa_code]; }
    }
    return cell;
}
template <typename F>
void register_pair_evals(RegisterCell<F> even, RegisterCell<F> odd, F inc0, F inc1, bool need_one, thread jolt::Accumulator<F>* sums) {
    if (!even.present && !odd.present) return;
    F low = even.present ? even.val : F::from_u64(odd.prev);
    F high = odd.present ? odd.val : F::from_u64(even.next);
    F dv = high - low, dra = odd.ra - even.ra, dwa = odd.wa - even.wa;
    sums[0].fmadd(even.ra, low); sums[0].fmadd(even.wa, low + inc0);
    sums[1].fmadd(dra, dv); sums[1].fmadd(dwa, dv + inc1 - inc0);
    if (need_one) { sums[2].fmadd(odd.ra, high); sums[2].fmadd(odd.wa, high + inc1); }
}

template <typename F>
kernel void rw_seed_message(
    device const RegisterSeed* entries [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device const ulong2* raw_inc [[buffer(2)]],
    device const uint4* masks [[buffer(3)]],
    device const F* ra_lut [[buffer(4)]],
    device const F* wa_lut [[buffer(5)]],
    device const F* e_in [[buffer(6)]],
    device const F* e_out [[buffer(7)]],
    device F* partials [[buffer(8)]],
    constant uint* p [[buffer(9)]],
    constant F& r1 [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort simdgroups [[simdgroups_per_threadgroup]],
    ushort simdwidth [[threads_per_simdgroup]]) {
    using Acc = jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 1024ul, "128 sparse cells and 1024 pair contributions fit");
    threadgroup Acc scratch[3][32];
    Acc sums[3] = {Acc::zero(), Acc::zero(), Acc::zero()};
    for (uint task = tid; task < 1024; task += 256) {
        uint pair = group * 1024 + task;
        if (pair >= p[1]) break;
        uint seed_group = p[4] ? pair : pair / 2;
        uint begin = offsets[seed_group], end = offsets[seed_group + 1];
        uint4 mask = masks[seed_group];
        F inc0, inc1;
        if (p[4]) {
            F a = register_raw_inc<F>(raw_inc,p[3],4*pair), b = register_raw_inc<F>(raw_inc,p[3],4*pair+1);
            F c = register_raw_inc<F>(raw_inc,p[3],4*pair+2), d = register_raw_inc<F>(raw_inc,p[3],4*pair+3);
            inc0 = a + r1 * (b-a); inc1 = c + r1 * (d-c);
        } else { inc0 = register_raw_inc<F>(raw_inc,p[3],2*pair); inc1 = register_raw_inc<F>(raw_inc,p[3],2*pair+1); }
        Acc inner[3] = {Acc::zero(), Acc::zero(), Acc::zero()};
        while (register_any(mask)) {
            uint col = register_next(mask);
            RegisterCell<F> even = p[4] ? register_seed_pair<F>(entries,begin,end,4*pair,col,r1) : register_seed_cell<F>(entries,begin,end,2*pair,col);
            RegisterCell<F> odd = p[4] ? register_seed_pair<F>(entries,begin,end,4*pair+2,col,r1) : register_seed_cell<F>(entries,begin,end,2*pair+1,col);
            even.ra = ra_lut[even.ra_code]; even.wa = wa_lut[even.wa_code];
            odd.ra = ra_lut[odd.ra_code]; odd.wa = wa_lut[odd.wa_code];
            register_pair_evals(even,odd,inc0,inc1,p[5]!=0,inner);
        }
        F weight = e_in[pair % p[2]] * e_out[pair / p[2]];
        for (uint k=0;k<3;k++) sums[k].fmadd(weight,inner[k].reduce());
    }
    for(uint k=0;k<3;k++) {
        Acc sum = jolt::threadgroup_merge(sums[k],scratch[k],lane,simdgroup,simdgroups,simdwidth);
        if(simdgroup==0 && lane==0) partials[3*group+k]=sum.reduce();
    }
}

template <typename F>
kernel void rw_seed_bind(
    device const RegisterSeed* entries [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device const ulong2* raw_inc [[buffer(2)]],
    device const uint4* masks [[buffer(3)]],
    device const uint* out_offsets [[buffer(4)]],
    device F* values [[buffer(5)]],
    device ulong2* boundaries [[buffer(6)]],
    device uint* codes [[buffer(7)]],
    device F* inc [[buffer(8)]],
    constant uint* p [[buffer(9)]],
    constant F& r1 [[buffer(10)]],
    constant F& r2 [[buffer(11)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p[0]) return;
    uint begin=offsets[row],end=offsets[row+1], index=out_offsets[row];
    uint4 mask=masks[row];
    while(register_any(mask)) {
        uint col=register_next(mask);
        RegisterCell<F> even=register_seed_pair<F>(entries,begin,end,4*row,col,r1);
        RegisterCell<F> odd=register_seed_pair<F>(entries,begin,end,4*row+2,col,r1);
        RegisterCell<F> cell=register_bind_cell(even,odd,r2,4,2,false);
        values[index]=cell.val; boundaries[index]=ulong2(cell.prev,cell.next);
        codes[index]=(cell.wa_code<<16)|cell.ra_code; index++;
    }
    if (row < ((p[1]>>2)+uint((p[1]&3)!=0))) {
        F a=register_raw_inc<F>(raw_inc,p[1],4*row),b=register_raw_inc<F>(raw_inc,p[1],4*row+1);
        F c=register_raw_inc<F>(raw_inc,p[1],4*row+2),d=register_raw_inc<F>(raw_inc,p[1],4*row+3);
        F lo=a+r1*(b-a),hi=c+r1*(d-c);
        inc[row]=lo+r2*(hi-lo);
    }
}

template <typename F>
kernel void rw_sparse_bind(
    device const F* values [[buffer(0)]],
    device const ulong2* boundaries [[buffer(1)]],
    device const uint* codes [[buffer(2)]],
    device const F* ra [[buffer(3)]],
    device const F* wa [[buffer(4)]],
    device const uint4* masks [[buffer(5)]],
    device const uint* offsets [[buffer(6)]],
    device const F* inc [[buffer(7)]],
    device const F* ra_lut [[buffer(8)]],
    device const F* wa_lut [[buffer(9)]],
    device F* out_values [[buffer(10)]],
    device ulong2* out_boundaries [[buffer(11)]],
    device uint* out_codes [[buffer(12)]],
    device F* out_ra [[buffer(13)]],
    device F* out_wa [[buffer(14)]],
    device const uint4* out_masks [[buffer(15)]],
    device const uint* out_offsets [[buffer(16)]],
    device F* out_inc [[buffer(17)]],
    constant uint* p [[buffer(18)]],
    constant F& r [[buffer(19)]],
    uint row [[thread_position_in_grid]]) {
    if(row>=p[1]) return;
    uint4 mask=out_masks[row]; uint index=out_offsets[row];
    while(register_any(mask)) {
        uint col=register_next(mask);
        RegisterCell<F> even=register_sparse_cell(values,boundaries,codes,ra,wa,masks,offsets,ra_lut,wa_lut,p[0],2*row,col,p[3]!=0,p[4]!=0);
        RegisterCell<F> odd=register_sparse_cell(values,boundaries,codes,ra,wa,masks,offsets,ra_lut,wa_lut,p[0],2*row+1,col,p[3]!=0,p[4]!=0);
        RegisterCell<F> cell=register_bind_cell(even,odd,r,p[5],p[6],p[4]!=0);
        out_values[index]=cell.val; out_boundaries[index]=ulong2(cell.prev,cell.next);
        if(p[4]) {out_ra[index]=cell.ra;out_wa[index]=cell.wa;}
        else out_codes[index]=(cell.wa_code<<16)|cell.ra_code;
        index++;
    }
    if(row<((p[2]>>1)+(p[2]&1))) {
        F lo=inc[2*row],hi=2*row+1<p[2]?inc[2*row+1]:F::zero();
        out_inc[row]=lo+r*(hi-lo);
    }
}

template <typename F>
kernel void rw_sparse_message(
    device const F* values [[buffer(0)]],
    device const ulong2* boundaries [[buffer(1)]],
    device const uint* codes [[buffer(2)]],
    device const F* ra [[buffer(3)]],
    device const F* wa [[buffer(4)]],
    device const uint4* masks [[buffer(5)]],
    device const uint* offsets [[buffer(6)]],
    device const F* inc [[buffer(7)]],
    device const F* ra_lut [[buffer(8)]],
    device const F* wa_lut [[buffer(9)]],
    device const F* e_in [[buffer(10)]],
    device const F* e_out [[buffer(11)]],
    device F* partials [[buffer(12)]],
    constant uint* p [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort simdgroups [[simdgroups_per_threadgroup]],
    ushort simdwidth [[threads_per_simdgroup]]) {
    using Acc=jolt::Accumulator<F>;
    static_assert(Acc::CAPACITY >= 1024ul, "128 sparse cells and 1024 pair contributions fit");
    threadgroup Acc scratch[3][32];
    Acc sums[3]={Acc::zero(),Acc::zero(),Acc::zero()};
    for(uint task=tid;task<1024;task+=256) {
        uint pair=group*1024+task;if(pair>=p[1])break;
        uint4 mask=masks[2*pair];if(2*pair+1<p[0])mask|=masks[2*pair+1];
        F lo_inc=2*pair<p[3]?inc[2*pair]:F::zero(),hi_inc=2*pair+1<p[3]?inc[2*pair+1]:F::zero();
        Acc inner[3]={Acc::zero(),Acc::zero(),Acc::zero()};
        while(register_any(mask)) {
            uint col=register_next(mask);
            RegisterCell<F> even=register_sparse_cell(values,boundaries,codes,ra,wa,masks,offsets,ra_lut,wa_lut,p[0],2*pair,col,p[4]!=0,true);
            RegisterCell<F> odd=register_sparse_cell(values,boundaries,codes,ra,wa,masks,offsets,ra_lut,wa_lut,p[0],2*pair+1,col,p[4]!=0,true);
            register_pair_evals(even,odd,lo_inc,hi_inc,p[5]!=0,inner);
        }
        F weight=e_in[pair%p[2]]*e_out[pair/p[2]];
        for(uint k=0;k<3;k++)sums[k].fmadd(weight,inner[k].reduce());
    }
    for(uint k=0;k<3;k++) {
        Acc sum=jolt::threadgroup_merge(sums[k],scratch[k],lane,simdgroup,simdgroups,simdwidth);
        if(simdgroup==0&&lane==0)partials[3*group+k]=sum.reduce();
    }
}
