//! Stage-5 scan-kernel roof attribution (wave-12 lane S12).
//!
//! Factorizes the achieved IrrPhaseScan / IrrSuffixScan rate against
//! candidate roofs on production-distribution rows, X9-method:
//!
//! - `base`   — production scan+reduce vs scan-only (reduce share), per
//!   phase shape
//! - `sgs`    — production scan at 512..8192 simdgroups (thread starvation
//!   + single-level reduce tax; production uses the two-level RAF reduce)
//! - `width`  — production scan at TG widths 32..256 (packing)
//! - `floor`  — probe kernels: quiet floor (loads + identical field math,
//!   no emit machinery) and loads floor (no field ALU) — the gap between
//!   production and quiet is the flush/emit machinery
//! - `chain`  — serial fr_mont_mul chain occupancy curve (Fr twin of the
//!   X9 fq mulroof)
//! - `suffix` — the same floors for the suffix scan
//!
//! Kernel-arm A/Bs (sorted default vs scatter/grouped/eager) run through
//! the production kill switches; this rig only prices roofs.
//!
//! Usage:
//!   /usr/bin/lockf -k /tmp/jolt-metal-gpu.lock env IRR_ROOF_ROWS=<dump> \
//!     cargo bench -p jolt-eval --features metal --bench irr_roof
//! Env: IRR_ROOF_CELLS (comma filter), IRR_ROOF_ITERS (default 6),
//!      IRR_ROOF_LOG_T (random-rows fallback when no dump; default 24),
//!      IRR_ROOF_PHASES (comma list, default "1,8"),
//!      IRR_ROOF_ROWS (JOLT_IRR_DUMP_ROWS file for production rows).

#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "benchmark harness must fail loudly and report on stdout"
)]

#[cfg(target_os = "macos")]
mod macos {
    use std::time::Duration;

    use jolt_kernels::metal::testing::gpu_lock;
    use jolt_kernels::metal::{IrrPhaseScanFixture, IrrSuffixScanFixture};

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn enabled(cell: &str) -> bool {
        match std::env::var("IRR_ROOF_CELLS") {
            Ok(filter) => filter.split(',').any(|entry| entry.trim() == cell),
            Err(_) => true,
        }
    }

    fn report(label: &str, samples: &[Duration]) -> f64 {
        let ms: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1e3).collect();
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        let min = ms.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "{label:<52} mean {mean:>9.3} ms   min {min:>9.3}   n={}",
            ms.len()
        );
        mean
    }

    fn sample(iters: usize, mut f: impl FnMut() -> Duration) -> Vec<Duration> {
        let _warm = f();
        (0..iters).map(|_| f()).collect()
    }

    /// Quiet floor: identical loads + field math per row (condense mul, two
    /// products, three held adds), no flush/emit machinery. One race-free
    /// write per lane at the end.
    const QUIET_FLOOR: &str = r#"
kernel void jkx_phase_quiet_floor(
    device const uint* rows [[buffer(0)]],
    device uint* u_evals [[buffer(1)]],
    device const uint* v_prev [[buffer(2)]],
    device uint* partials [[buffer(3)]],
    constant IrrPhaseScanParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    uint sg = gid / simd_size;
    if (sg >= p.num_sgs) { return; }
    device uint* my = partials + sg * JK_IRR_RAF_CELLS * FR_LIMBS;
    Fr256 h0 = fr_zero();
    Fr256 h1 = fr_zero();
    Fr256 h2 = fr_zero();
    uint key_acc = 0u;
    uint row_start = sg * p.rows_per_sg;
    uint row_end = min(row_start + p.rows_per_sg, p.n);
    for (uint base = row_start; base < row_end; base += simd_size) {
        uint j = base + lane;
        if (j >= row_end) { continue; }
        device const uint* row = rows + j * 12u;
        ulong lo = (ulong)row[0] | ((ulong)row[1] << 32);
        ulong hi = (ulong)row[2] | ((ulong)row[3] << 32);
        bool flag = ((row[8] >> 8) & 0xFFu) != 0u;
        Fr256 u = fr_load(u_evals, j);
        if (p.do_condense != 0u) {
            u = fr_mont_mul(u, fr_load(v_prev, jk_chunk8(lo, hi, p.prev_shift)));
            fr_store(u_evals, j, u);
        }
        key_acc ^= (flag ? 256u : 0u) + jk_chunk8(lo, hi, p.suffix_len);
        ulong s_lo, s_hi;
        jk_mask128(lo, hi, p.suffix_len, s_lo, s_hi);
        Fr256 v0 = u;
        Fr256 v1;
        Fr256 v2;
        if (!flag) {
            ulong x, y;
            jk_uninterleave(s_lo, s_hi, x, y);
            v1 = fr_mont_mul(u, jk_fr_from_u64(x));
            v2 = fr_mont_mul(u, jk_fr_from_u64(y));
        } else {
            v1 = fr_mont_mul(u, jk_fr_from_u128(s_lo, s_hi));
            bool upper_ok = (p.canonical != 0u)
                && (p.upper_suffix_bits == 0u
                    || s_hi == ((1ul << p.upper_suffix_bits) - 1ul));
            v2 = upper_ok ? u : fr_zero();
        }
        h0 = fr_add(h0, v0);
        h1 = fr_add(h1, v1);
        h2 = fr_add(h2, v2);
    }
    h0.v[0] ^= key_acc;
    uint cell = lane * 3u;
    fr_store(my, cell, h0);
    fr_store(my, cell + 1u, h1);
    fr_store(my, cell + 2u, h2);
}
"#;

    /// Loads floor: identical memory traffic (row words, u load, condense
    /// store-back), zero field ALU.
    const LOADS_FLOOR: &str = r#"
kernel void jkx_phase_loads_floor(
    device const uint* rows [[buffer(0)]],
    device uint* u_evals [[buffer(1)]],
    device const uint* v_prev [[buffer(2)]],
    device uint* partials [[buffer(3)]],
    constant IrrPhaseScanParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    uint sg = gid / simd_size;
    if (sg >= p.num_sgs) { return; }
    device uint* my = partials + sg * JK_IRR_RAF_CELLS * FR_LIMBS;
    uint acc = 0u;
    uint row_start = sg * p.rows_per_sg;
    uint row_end = min(row_start + p.rows_per_sg, p.n);
    for (uint base = row_start; base < row_end; base += simd_size) {
        uint j = base + lane;
        if (j >= row_end) { continue; }
        device const uint* row = rows + j * 12u;
        acc ^= row[0] ^ row[1] ^ row[2] ^ row[3] ^ row[8];
        Fr256 u = fr_load(u_evals, j);
        for (uint i = 0u; i < FR_LIMBS; i++) { acc ^= u.v[i]; }
        if (p.do_condense != 0u) {
            fr_store(u_evals, j, u);
        }
    }
    my[lane] = acc;
}
"#;

    /// Serial fr_mont_mul chain: `p.n` dependent muls per thread,
    /// `p.num_sgs` = thread bound.
    const FR_CHAIN: &str = r#"
kernel void jkx_fr_mul_chain(
    device const uint* rows [[buffer(0)]],
    device uint* u_evals [[buffer(1)]],
    device const uint* v_prev [[buffer(2)]],
    device uint* partials [[buffer(3)]],
    constant IrrPhaseScanParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.num_sgs) { return; }
    Fr256 v = fr_load(v_prev, gid & 255u);
    Fr256 c = fr_load(v_prev, (gid * 7u + 1u) & 255u);
    for (uint i = 0u; i < p.n; i++) {
        v = fr_mont_mul(v, c);
    }
    fr_store(u_evals, gid, v);
}
"#;

    /// Suffix quiet floor: gathers + suffix MLEs + products, no emission.
    const SUFFIX_QUIET_FLOOR: &str = r#"
kernel void jkx_suffix_quiet_floor(
    device const uint* rows [[buffer(0)]],
    device const uint* u_evals [[buffer(1)]],
    device const uint* bucket_flat [[buffer(2)]],
    device const uint* sg_slot [[buffer(3)]],
    device const uint* sg_range [[buffer(4)]],
    device const uint* suffix_meta [[buffer(5)]],
    device uint* partials [[buffer(6)]],
    constant IrrSuffixScanParams& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    uint sg = gid / simd_size;
    if (sg >= p.num_sgs) { return; }
    uint slot = sg_slot[sg];
    uint count = suffix_meta[slot * 9u];
    device uint* my = partials + sg * JK_IRR_SUF_CELLS * FR_LIMBS;
    Fr256 acc = fr_zero();
    uint start = sg_range[2u * sg];
    uint end = sg_range[2u * sg + 1u];
    for (uint base = start; base < end; base += simd_size) {
        uint i = base + lane;
        if (i >= end) { continue; }
        uint j = bucket_flat[i];
        device const uint* row = rows + j * 12u;
        ulong lo = (ulong)row[0] | ((ulong)row[1] << 32);
        ulong hi = (ulong)row[2] | ((ulong)row[3] << 32);
        Fr256 u = fr_load(u_evals, j);
        ulong s_lo, s_hi;
        jk_mask128(lo, hi, p.suffix_len, s_lo, s_hi);
        for (uint s = 0u; s < count; s++) {
            uint meta = suffix_meta[slot * 9u + 1u + s];
            uint id = meta & 0xFFu;
            bool is01 = (meta & 0x100u) != 0u;
            ulong m = jk_suffix_mle(id, s_lo, s_hi, p.suffix_len);
            if (m != 0ul) {
                Fr256 v = is01 ? u : fr_mont_mul(u, jk_fr_from_u64(m));
                acc = fr_add(acc, v);
            }
        }
    }
    fr_store(my, lane, acc);
}
"#;

    /// Suffix loads floor: gathers only, zero field ALU / MLE work.
    const SUFFIX_LOADS_FLOOR: &str = r#"
kernel void jkx_suffix_loads_floor(
    device const uint* rows [[buffer(0)]],
    device const uint* u_evals [[buffer(1)]],
    device const uint* bucket_flat [[buffer(2)]],
    device const uint* sg_slot [[buffer(3)]],
    device const uint* sg_range [[buffer(4)]],
    device const uint* suffix_meta [[buffer(5)]],
    device uint* partials [[buffer(6)]],
    constant IrrSuffixScanParams& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    uint sg = gid / simd_size;
    if (sg >= p.num_sgs) { return; }
    device uint* my = partials + sg * JK_IRR_SUF_CELLS * FR_LIMBS;
    uint acc = 0u;
    uint start = sg_range[2u * sg];
    uint end = sg_range[2u * sg + 1u];
    for (uint base = start; base < end; base += simd_size) {
        uint i = base + lane;
        if (i >= end) { continue; }
        uint j = bucket_flat[i];
        device const uint* row = rows + j * 12u;
        acc ^= row[0] ^ row[1] ^ row[2] ^ row[3] ^ row[8];
        Fr256 u = fr_load(u_evals, j);
        for (uint k = 0u; k < FR_LIMBS; k++) { acc ^= u.v[k]; }
    }
    my[lane] = acc;
}
"#;

    /// Production phase-scan body with the flush swapped for `//FLUSH//`:
    /// the machinery-decomposition ladder (w17). Dead-state XOR folds keep
    /// every register live without field-add pollution.
    const PHASE_TMPL: &str = r#"
kernel void ENTRY(
    device const uint* rows [[buffer(0)]],
    device uint* u_evals [[buffer(1)]],
    device const uint* v_prev [[buffer(2)]],
    device uint* partials [[buffer(3)]],
    constant IrrPhaseScanParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    uint sg = gid / simd_size;
    if (sg >= p.num_sgs) { return; }
    device uint* my = partials + sg * JK_IRR_RAF_CELLS * FR_LIMBS;
    for (uint i = lane; i < JK_IRR_RAF_CELLS * FR_LIMBS; i += simd_size) { my[i] = 0u; }
    simdgroup_barrier(mem_flags::mem_device);
    uint held_key = 0xFFFFFFFFu;
    Fr256 h0 = fr_zero();
    Fr256 h1 = fr_zero();
    Fr256 h2 = fr_zero();
    uint acc = 0u;
    uint row_start = sg * p.rows_per_sg;
    uint row_end = min(row_start + p.rows_per_sg, p.n);
    for (uint base = row_start; base < row_end; base += simd_size) {
        uint j = base + lane;
        bool active = j < row_end;
        uint chunk = 0u;
        bool flag = false;
        Fr256 v0 = fr_zero();
        Fr256 v1 = fr_zero();
        Fr256 v2 = fr_zero();
        if (active) {
            device const uint* row = rows + j * 12u;
            ulong lo = (ulong)row[0] | ((ulong)row[1] << 32);
            ulong hi = (ulong)row[2] | ((ulong)row[3] << 32);
            flag = ((row[8] >> 8) & 0xFFu) != 0u;
            Fr256 u = fr_load(u_evals, j);
            if (p.do_condense != 0u) {
                u = fr_mont_mul(u, fr_load(v_prev, jk_chunk8(lo, hi, p.prev_shift)));
                fr_store(u_evals, j, u);
            }
            chunk = jk_chunk8(lo, hi, p.suffix_len);
            ulong s_lo, s_hi;
            jk_mask128(lo, hi, p.suffix_len, s_lo, s_hi);
            v0 = u;
            if (!flag) {
                ulong x, y;
                jk_uninterleave(s_lo, s_hi, x, y);
                v1 = fr_mont_mul(u, jk_fr_from_u64(x));
                v2 = fr_mont_mul(u, jk_fr_from_u64(y));
            } else {
                v1 = fr_mont_mul(u, jk_fr_from_u128(s_lo, s_hi));
                bool upper_ok = (p.canonical != 0u)
                    && (p.upper_suffix_bits == 0u
                        || s_hi == ((1ul << p.upper_suffix_bits) - 1ul));
                v2 = upper_ok ? u : fr_zero();
            }
        }
        uint key = (flag ? 256u : 0u) + chunk;
        bool same = active && key == held_key;
        if (same) {
            h0 = fr_add(h0, v0);
            h1 = fr_add(h1, v1);
            h2 = fr_add(h2, v2);
        }
        bool take = active && !same;
        bool flush = take && held_key != 0xFFFFFFFFu;
        if (simd_any(flush)) {
            //FLUSH//
        }
        if (take) {
            held_key = key;
            h0 = v0;
            h1 = v1;
            h2 = v2;
        }
    }
    bool have = held_key != 0xFFFFFFFFu;
    if (simd_any(have)) {
        bool flush = have;
        //FLUSH//
    }
    uint cell = lane * 3u;
    h0.v[0] ^= acc;
    fr_store(my, cell, h0);
    fr_store(my, cell + 1u, h1);
    fr_store(my, cell + 2u, h2);
}
"#;

    const FLUSH_NONE: &str = "acc ^= held_key;";
    const FLUSH_SORT: &str = r#"
            uint packed = jk_sort_key_lane(flush ? ((held_key << 5u) | lane) : 0xFFFFu, lane);
            acc ^= packed;
"#;
    const FLUSH_GATHER: &str = r#"
            uint packed = jk_sort_key_lane(flush ? ((held_key << 5u) | lane) : 0xFFFFu, lane);
            uint src = packed & 31u;
            Fr256 g0 = jk_fr_shuffle_v4(h0, src);
            Fr256 g1 = jk_fr_shuffle_v4(h1, src);
            Fr256 g2 = jk_fr_shuffle_v4(h2, src);
            for (uint i = 0u; i < FR_LIMBS; i++) { acc ^= g0.v[i] ^ g1.v[i] ^ g2.v[i]; }
"#;
    const FLUSH_SCAN: &str = r#"
            uint packed = jk_sort_key_lane(flush ? ((held_key << 5u) | lane) : 0xFFFFu, lane);
            uint src = packed & 31u;
            uint skey = packed >> 5u;
            bool valid = packed != 0xFFFFu;
            Fr256 g0 = jk_fr_shuffle_v4(h0, src);
            Fr256 g1 = jk_fr_shuffle_v4(h1, src);
            Fr256 g2 = jk_fr_shuffle_v4(h2, src);
            uint run_off; uint max_off; bool tail;
            jk_sorted_runs(skey, valid, lane, simd_size, run_off, max_off, tail);
            for (uint d = 1u; d <= max_off; d <<= 1u) {
                Fr256 p0 = jk_fr_shuffle_up_v4(g0, (ushort)d);
                Fr256 p1 = jk_fr_shuffle_up_v4(g1, (ushort)d);
                Fr256 p2 = jk_fr_shuffle_up_v4(g2, (ushort)d);
                if (run_off >= d) {
                    g0 = fr_add(g0, p0);
                    g1 = fr_add(g1, p1);
                    g2 = fr_add(g2, p2);
                }
            }
            acc ^= (uint)tail;
            for (uint i = 0u; i < FR_LIMBS; i++) { acc ^= g0.v[i] ^ g1.v[i] ^ g2.v[i]; }
"#;
    const FLUSH_SCAN5: &str = r#"
            uint packed = jk_sort_key_lane(flush ? ((held_key << 5u) | lane) : 0xFFFFu, lane);
            uint src = packed & 31u;
            uint skey = packed >> 5u;
            bool valid = packed != 0xFFFFu;
            Fr256 g0 = jk_fr_shuffle_v4(h0, src);
            Fr256 g1 = jk_fr_shuffle_v4(h1, src);
            Fr256 g2 = jk_fr_shuffle_v4(h2, src);
            for (uint d = 1u; d < simd_size; d <<= 1u) {
                uint pkey = simd_shuffle_up(skey, (ushort)d);
                Fr256 p0 = jk_fr_shuffle_up_v4(g0, (ushort)d);
                Fr256 p1 = jk_fr_shuffle_up_v4(g1, (ushort)d);
                Fr256 p2 = jk_fr_shuffle_up_v4(g2, (ushort)d);
                if (valid && lane >= d && pkey == skey) {
                    g0 = fr_add(g0, p0);
                    g1 = fr_add(g1, p1);
                    g2 = fr_add(g2, p2);
                }
            }
            uint nkey = simd_shuffle_down(skey, 1u);
            bool tail = valid && (lane == simd_size - 1u || nkey != skey);
            acc ^= (uint)tail;
            for (uint i = 0u; i < FR_LIMBS; i++) { acc ^= g0.v[i] ^ g1.v[i] ^ g2.v[i]; }
"#;
    const FLUSH_TAIL: &str = r#"
            uint packed = jk_sort_key_lane(flush ? ((held_key << 5u) | lane) : 0xFFFFu, lane);
            uint src = packed & 31u;
            uint skey = packed >> 5u;
            bool valid = packed != 0xFFFFu;
            Fr256 g0 = jk_fr_shuffle_v4(h0, src);
            Fr256 g1 = jk_fr_shuffle_v4(h1, src);
            Fr256 g2 = jk_fr_shuffle_v4(h2, src);
            uint run_off; uint max_off; bool tail;
            jk_sorted_runs(skey, valid, lane, simd_size, run_off, max_off, tail);
            for (uint d = 1u; d <= max_off; d <<= 1u) {
                Fr256 p0 = jk_fr_shuffle_up_v4(g0, (ushort)d);
                Fr256 p1 = jk_fr_shuffle_up_v4(g1, (ushort)d);
                Fr256 p2 = jk_fr_shuffle_up_v4(g2, (ushort)d);
                if (run_off >= d) {
                    g0 = fr_add(g0, p0);
                    g1 = fr_add(g1, p1);
                    g2 = fr_add(g2, p2);
                }
            }
            if (tail) {
                uint family_base = (skey >> 8u) * 3u * 256u;
                uint chunk_t = skey & 255u;
                jk_cell_add(my, family_base + chunk_t, g0);
                jk_cell_add(my, family_base + 256u + chunk_t, g1);
                jk_cell_add(my, family_base + 512u + chunk_t, g2);
            }
"#;

    fn phase_fixture(log_t: usize) -> IrrPhaseScanFixture {
        match std::env::var("IRR_ROOF_ROWS") {
            Ok(path) => {
                IrrPhaseScanFixture::from_rows_file(std::path::Path::new(&path)).expect("rows file")
            }
            Err(_) => IrrPhaseScanFixture::production_geometry(log_t).expect("phase fixture"),
        }
    }

    fn phases() -> Vec<usize> {
        std::env::var("IRR_ROOF_PHASES")
            .unwrap_or_else(|_| "1,8".to_string())
            .split(',')
            .filter_map(|entry| entry.trim().parse().ok())
            .collect()
    }

    pub fn main() {
        let _gpu = gpu_lock();
        let iters = env_usize("IRR_ROOF_ITERS", 6);
        let log_t = env_usize("IRR_ROOF_LOG_T", 24);
        let real = std::env::var("IRR_ROOF_ROWS").is_ok();
        println!(
            "== irr_roof: iters={iters} rows={} ==",
            if real { "REAL" } else { "random" }
        );

        let mut fixture = phase_fixture(log_t);

        if enabled("base") {
            println!("-- base: production scan+reduce vs scan-only --");
            for phase in phases() {
                let suffix_len = 128 - 8 * (phase + 1);
                fixture.set_phase_shape(suffix_len, phase > 0);
                let buffers = fixture.buffers().expect("buffers");
                report(
                    &format!("P{phase} scan+reduce"),
                    &sample(iters, || buffers.run_timed().expect("timed")),
                );
                report(
                    &format!("P{phase} scan only"),
                    &sample(iters, || buffers.run_timed_scan_only().expect("timed")),
                );
                report(
                    &format!("P{phase} scan only FIXED"),
                    &sample(iters, || {
                        buffers.run_timed_scan_only_fixed().expect("timed")
                    }),
                );
            }
            fixture.reset_u_evals();
        }

        if enabled("sgs") {
            println!("-- sgs: production scan at 512..8192 simdgroups (P1 shape) --");
            fixture.set_phase_shape(112, true);
            for sgs in [512usize, 1024, 2048, 4096, 8192] {
                fixture.set_simdgroups(sgs);
                let buffers = fixture.buffers().expect("buffers");
                report(
                    &format!("sgs={sgs} scan only"),
                    &sample(iters, || buffers.run_timed_scan_only().expect("timed")),
                );
                report(
                    &format!("sgs={sgs} scan+reduce"),
                    &sample(iters, || buffers.run_timed().expect("timed")),
                );
            }
            fixture.set_simdgroups(4096);
            fixture.reset_u_evals();
        }

        if enabled("width") {
            println!("-- width: production scan TG width (P1 shape) --");
            fixture.set_phase_shape(112, true);
            let buffers = fixture.buffers().expect("buffers");
            for width in [32usize, 64, 128, 256] {
                report(
                    &format!("width={width} scan only"),
                    &sample(iters, || buffers.run_timed_width(width).expect("timed")),
                );
            }
            fixture.reset_u_evals();
        }

        if enabled("mach") {
            println!("-- mach: flush-machinery decomposition ladder --");
            let ladder = [
                ("none", FLUSH_NONE),
                ("sort", FLUSH_SORT),
                ("gather", FLUSH_GATHER),
                ("scan", FLUSH_SCAN),
                ("scan5", FLUSH_SCAN5),
                ("tail", FLUSH_TAIL),
            ];
            let probes: Vec<_> = ladder
                .iter()
                .map(|(name, flush)| {
                    let source = PHASE_TMPL
                        .replace("ENTRY", &format!("jkx_phase_{name}"))
                        .replace("//FLUSH//", flush);
                    let probe = fixture
                        .compile_probe(&source, &format!("jkx_phase_{name}"))
                        .expect("mach probe");
                    println!("{name} stats (maxTG, width): {:?}", probe.stats());
                    (*name, probe)
                })
                .collect();
            for phase in phases() {
                let suffix_len = 128 - 8 * (phase + 1);
                fixture.set_phase_shape(suffix_len, phase > 0);
                let buffers = fixture.buffers().expect("buffers");
                for (name, probe) in &probes {
                    report(
                        &format!("P{phase} flush={name}"),
                        &sample(iters, || {
                            buffers.run_timed_probe(probe, 256).expect("timed")
                        }),
                    );
                }
                report(
                    &format!("P{phase} production"),
                    &sample(iters, || buffers.run_timed_scan_only().expect("timed")),
                );
            }
            fixture.reset_u_evals();
        }

        if enabled("floor") {
            println!("-- floor probes --");
            let quiet = fixture
                .compile_probe(QUIET_FLOOR, "jkx_phase_quiet_floor")
                .expect("quiet probe");
            let loads = fixture
                .compile_probe(LOADS_FLOOR, "jkx_phase_loads_floor")
                .expect("loads probe");
            println!(
                "quiet stats (maxTG, width): {:?}   loads: {:?}",
                quiet.stats(),
                loads.stats()
            );
            for phase in phases() {
                let suffix_len = 128 - 8 * (phase + 1);
                fixture.set_phase_shape(suffix_len, phase > 0);
                for sgs in [512usize, 2048, 4096] {
                    fixture.set_simdgroups(sgs);
                    let buffers = fixture.buffers().expect("buffers");
                    report(
                        &format!("P{phase} sgs={sgs} quiet floor"),
                        &sample(iters, || {
                            buffers.run_timed_probe(&quiet, 256).expect("timed")
                        }),
                    );
                    report(
                        &format!("P{phase} sgs={sgs} loads floor"),
                        &sample(iters, || {
                            buffers.run_timed_probe(&loads, 256).expect("timed")
                        }),
                    );
                }
                fixture.set_simdgroups(4096);
            }
            fixture.reset_u_evals();
        }

        if enabled("chain") {
            println!("-- fr_mont_mul serial-chain occupancy curve --");
            let chain = fixture
                .compile_probe(FR_CHAIN, "jkx_fr_mul_chain")
                .expect("chain probe");
            println!("chain stats: {:?}", chain.stats());
            let muls_per_thread = 4096u32;
            for threads in [16_384usize, 32_768, 65_536, 131_072, 262_144, 524_288] {
                let params: Vec<u32> = vec![muls_per_thread, 0, threads as u32, 0, 0, 0, 0, 0];
                let samples = sample(iters, || {
                    fixture
                        .buffers()
                        .expect("buffers")
                        .run_timed_probe_threads(&chain, &params, threads, 256)
                        .expect("timed")
                });
                let mean = report(&format!("chain threads={threads}"), &samples);
                let gmul = (threads as f64 * muls_per_thread as f64) / (mean * 1e-3) / 1e9;
                println!("{:>52} {gmul:.2} Gmul/s", "->");
            }
        }

        if enabled("suffix") {
            println!("-- suffix floors --");
            let mut sfx = match std::env::var("IRR_ROOF_ROWS") {
                Ok(path) => IrrSuffixScanFixture::from_rows_file(std::path::Path::new(&path))
                    .expect("suffix rows"),
                Err(_) => IrrSuffixScanFixture::production_geometry(log_t).expect("suffix fixture"),
            };
            let quiet = fixture
                .compile_probe(SUFFIX_QUIET_FLOOR, "jkx_suffix_quiet_floor")
                .expect("suffix quiet");
            let loads = fixture
                .compile_probe(SUFFIX_LOADS_FLOOR, "jkx_suffix_loads_floor")
                .expect("suffix loads");
            let buffers = sfx.buffers().expect("suffix buffers");
            report(
                "suffix scan+reduce",
                &sample(iters, || buffers.run_timed().expect("timed")),
            );
            report(
                "suffix scan only",
                &sample(iters, || buffers.run_timed_scan_only().expect("timed")),
            );
            report(
                "suffix scan only FIXED",
                &sample(iters, || {
                    buffers.run_timed_scan_only_fixed().expect("timed")
                }),
            );
            report(
                "suffix quiet floor",
                &sample(iters, || {
                    buffers.run_timed_probe(&quiet, 256).expect("timed")
                }),
            );
            report(
                "suffix loads floor",
                &sample(iters, || {
                    buffers.run_timed_probe(&loads, 256).expect("timed")
                }),
            );
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos::main();
}

#[cfg(not(target_os = "macos"))]
#[expect(clippy::print_stderr, reason = "benchmark harness must fail loudly")]
fn main() {
    eprintln!("irr_roof requires macOS");
}
