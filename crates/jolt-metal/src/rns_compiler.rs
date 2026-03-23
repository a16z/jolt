//! RNS-Montgomery kernel compiler: generates MSL kernels for sumcheck reduce.
//!
//! Elements are stored in RNS-Montgomery form (`ã = a·M mod r`) decomposed
//! across 18 pseudo-Mersenne primes (9 primary B + 9 secondary B'). Each thread
//! reads all 18 residues of a pair and performs the full RNS-Montgomery multiply
//! inline — including Garner base extension from B → B'.
//!
//! # Architecture
//!
//! Input buffers: SoA layout per polynomial:
//! `[B_0: n u32s | B_1: | ... | B_8: | B'_0: | ... | B'_8: ]`
//!
//! Each thread processes one pair via grid-stride loop. Per pair:
//! 1. Read lo/hi in both bases (18 + 18 u32)
//! 2. Compute diff = hi - lo (component-wise in both bases)
//! 3. For each eval point t: cur(t) via incremental addition
//! 4. RNS-Montgomery multiply: weight × cur
//! 5. Accumulate result in B' accumulators (9 per eval point)
//!
//! Output: `num_groups × BASIS_SIZE × num_evals` u32 partial sums (B' only).
//! CPU reconstructs via Garner CRT on B' and reduces mod r.
//!
//! # Register pressure
//!
//! Per thread with D=4: ~108 u32 → 2.3 simdgroups/EU occupancy.
//! (vs CIOS: 280-500 u32 → 0.5-0.9 simdgroups/EU)

use std::fmt::Write;
use std::marker::PhantomData;

use jolt_field::Field;
use jolt_ir::{KernelDescriptor, KernelShape};
use metal::CompileOptions;

use crate::rns::{
    self, b_mod_bp, bp_mod_b, garner_inv_b, garner_inv_bp, m_inv_bp, neg_r_inv_b, r_mod_bp,
    BASIS_SIZE,
};
use crate::shaders::{build_source, make_pipeline, SHADER_RNS};

/// Threadgroup size for RNS-Montgomery reduce kernels.
pub(crate) const RNS_REDUCE_GROUP_SIZE: usize = 256;

/// Maximum threadgroups per RNS-Montgomery reduce dispatch.
pub(crate) const RNS_MAX_REDUCE_GROUPS: usize = 256;

const SIMD_SIZE: usize = 32;

/// Compiled RNS-Montgomery reduce kernel.
pub struct RnsKernel<F: Field> {
    pub pipeline_l2h: metal::ComputePipelineState,
    pub pipeline_h2l: metal::ComputePipelineState,
    pub pipeline_l2h_unw: metal::ComputePipelineState,
    pub pipeline_h2l_unw: metal::ComputePipelineState,
    pub num_evals: usize,
    pub num_inputs: usize,
    pub _marker: PhantomData<F>,
}

#[derive(Clone, Copy)]
enum RnsVariant {
    LowToHigh,
    HighToLow,
}

impl RnsVariant {
    fn function_name(self, weighted: bool) -> &'static str {
        match (self, weighted) {
            (Self::LowToHigh, true) => "rns_reduce_l2h",
            (Self::HighToLow, true) => "rns_reduce_h2l",
            (Self::LowToHigh, false) => "rns_reduce_l2h_unw",
            (Self::HighToLow, false) => "rns_reduce_h2l_unw",
        }
    }
}

/// Compile an RNS-Montgomery reduce kernel from a KernelDescriptor.
///
/// Only supports ProductSum with P=1 (single input per product group).
pub fn compile_rns<F: Field>(
    device: &metal::Device,
    descriptor: &KernelDescriptor,
) -> RnsKernel<F> {
    let num_inputs = descriptor.num_inputs();
    let num_evals = descriptor.num_evals();

    let (d, p) = match &descriptor.shape {
        KernelShape::ProductSum {
            num_inputs_per_product,
            num_products,
        } => (*num_inputs_per_product, *num_products),
        _ => panic!("RNS compiler only supports ProductSum shape"),
    };

    assert_eq!(
        p, 1,
        "RNS compiler requires P=1 (no inter-element products). Got P={p}"
    );

    // Force initialization of precomputed constants
    let _ = neg_r_inv_b();
    let _ = m_inv_bp();
    let _ = r_mod_bp();
    let _ = garner_inv_b();
    let _ = b_mod_bp();
    let _ = garner_inv_bp();
    let _ = bp_mod_b();

    let constants_msl = generate_rns_mont_constants();

    let mut msl = String::with_capacity(65536);
    msl.push_str(&constants_msl);
    msl.push('\n');

    for variant in [RnsVariant::LowToHigh, RnsVariant::HighToLow] {
        msl.push_str(&generate_rns_mont_reduce_kernel(
            num_inputs, num_evals, d, variant, true,
        ));
        msl.push('\n');
        msl.push_str(&generate_rns_mont_reduce_kernel(
            num_inputs, num_evals, d, variant, false,
        ));
        msl.push('\n');
    }

    let source = build_source(&[SHADER_RNS, &msl]);
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(&source, &options)
        .unwrap_or_else(|e| panic!("RNS-Montgomery reduce kernel MSL compilation failed: {e}"));

    RnsKernel {
        pipeline_l2h: make_pipeline(device, &library, "rns_reduce_l2h"),
        pipeline_h2l: make_pipeline(device, &library, "rns_reduce_h2l"),
        pipeline_l2h_unw: make_pipeline(device, &library, "rns_reduce_l2h_unw"),
        pipeline_h2l_unw: make_pipeline(device, &library, "rns_reduce_h2l_unw"),
        num_evals,
        num_inputs,
        _marker: PhantomData,
    }
}

/// Compiled RNS-Montgomery bind kernel (interpolate pairs).
///
/// Independent of D — the same bind kernel works for all reduce shapes.
/// Computes `result[i] = lo[i] + challenge × (hi[i] - lo[i])` in RNS-Montgomery form
/// with one multiply + one B'→B base extension per element.
pub struct RnsBindKernel {
    pub pipeline_l2h: metal::ComputePipelineState,
    pub pipeline_h2l: metal::ComputePipelineState,
}

/// SAFETY: MTLComputePipelineState is immutable after creation.
unsafe impl Send for RnsBindKernel {}
/// SAFETY: See above.
unsafe impl Sync for RnsBindKernel {}

/// Compile the RNS-Montgomery bind kernel.
pub fn compile_rns_bind(device: &metal::Device) -> RnsBindKernel {
    let _ = neg_r_inv_b();
    let _ = m_inv_bp();
    let _ = r_mod_bp();
    let _ = garner_inv_b();
    let _ = b_mod_bp();
    let _ = garner_inv_bp();
    let _ = bp_mod_b();

    let constants_msl = generate_rns_mont_constants();

    let mut msl = String::with_capacity(32768);
    msl.push_str(&constants_msl);
    msl.push('\n');
    msl.push_str(&generate_rns_bind_kernel(RnsVariant::LowToHigh));
    msl.push('\n');
    msl.push_str(&generate_rns_bind_kernel(RnsVariant::HighToLow));
    msl.push('\n');

    let source = build_source(&[SHADER_RNS, &msl]);
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(&source, &options)
        .unwrap_or_else(|e| panic!("RNS bind kernel MSL compilation failed: {e}"));

    RnsBindKernel {
        pipeline_l2h: make_pipeline(device, &library, "rns_bind_l2h"),
        pipeline_h2l: make_pipeline(device, &library, "rns_bind_h2l"),
    }
}

/// Generate an RNS-Montgomery bind kernel (pairwise interpolation).
///
/// Each thread computes one `result = lo + challenge × (hi - lo)` in full
/// RNS-Montgomery form (both B and B' bases in the output).
fn generate_rns_bind_kernel(variant: RnsVariant) -> String {
    let bs = BASIS_SIZE;
    let np = rns::NUM_PRIMES;
    let is_h2l = matches!(variant, RnsVariant::HighToLow);
    let fname = if is_h2l {
        "rns_bind_h2l"
    } else {
        "rns_bind_l2h"
    };

    let mut s = String::with_capacity(16384);

    let _ = writeln!(s, "kernel void {fname}(");
    let _ = writeln!(s, "    device const uint* data [[buffer(0)]],");
    let _ = writeln!(s, "    device const uint* challenge [[buffer(1)]],");
    let _ = writeln!(s, "    device uint* output [[buffer(2)]],");
    let _ = writeln!(s, "    device const uint* params [[buffer(3)]],");
    let _ = writeln!(s, "    uint tid [[thread_position_in_grid]]");
    let _ = writeln!(s, ") {{");
    let _ = writeln!(s, "    uint n_pairs = params[0];");
    let _ = writeln!(s, "    if (tid >= n_pairs) return;");
    s.push('\n');

    // Read challenge residues (18 u32)
    for r in 0..np {
        let basis = if r < bs { "b" } else { "bp" };
        let idx = if r < bs { r } else { r - bs };
        let _ = writeln!(s, "    uint ch_{basis}{idx} = challenge[{r}u];");
    }
    s.push('\n');

    // Read lo/hi pair
    for r in 0..np {
        let basis = if r < bs { "b" } else { "bp" };
        let idx = if r < bs { r } else { r - bs };
        if is_h2l {
            let _ = writeln!(
                s,
                "    uint lo_{basis}{idx} = data[{r}u * n_pairs * 2u + tid];"
            );
            let _ = writeln!(
                s,
                "    uint hi_{basis}{idx} = data[{r}u * n_pairs * 2u + tid + n_pairs];"
            );
        } else {
            let _ = writeln!(
                s,
                "    uint lo_{basis}{idx} = data[{r}u * n_pairs * 2u + 2u * tid];"
            );
            let _ = writeln!(
                s,
                "    uint hi_{basis}{idx} = data[{r}u * n_pairs * 2u + 2u * tid + 1u];"
            );
        }
    }
    s.push('\n');

    // diff = hi - lo in both bases
    for i in 0..bs {
        let _ = writeln!(
            s,
            "    uint diff_b{i} = rns_sub(hi_b{i}, lo_b{i}, B_PRIMES[{i}]);"
        );
    }
    for j in 0..bs {
        let _ = writeln!(
            s,
            "    uint diff_bp{j} = rns_sub(hi_bp{j}, lo_bp{j}, BP_PRIMES[{j}]);"
        );
    }
    s.push('\n');

    // RNS-Montgomery multiply: challenge × diff → prod (B' only)
    emit_rns_mont_multiply(&mut s, "ch_b", "ch_bp", "diff_b", "diff_bp", "prod", bs);
    s.push('\n');

    // Base extend product from B' → B
    emit_base_extend_bp_to_b(&mut s, "prod", bs);
    s.push('\n');

    // result = lo + product (both bases)
    for i in 0..bs {
        let _ = writeln!(
            s,
            "    uint res_b{i} = rns_add(lo_b{i}, prod_b{i}, B_PRIMES[{i}]);"
        );
    }
    for j in 0..bs {
        let _ = writeln!(
            s,
            "    uint res_bp{j} = rns_add(lo_bp{j}, prod_bp{j}, BP_PRIMES[{j}]);"
        );
    }
    s.push('\n');

    // Write result: n_pairs elements in SoA layout
    for r in 0..np {
        let basis = if r < bs { "b" } else { "bp" };
        let idx = if r < bs { r } else { r - bs };
        let _ = writeln!(s, "    output[{r}u * n_pairs + tid] = res_{basis}{idx};");
    }

    let _ = writeln!(s, "}}");
    s
}

/// Generate MSL constant declarations for RNS-Montgomery precomputed values.
fn generate_rns_mont_constants() -> String {
    let neg_r_inv = neg_r_inv_b();
    let m_inv = m_inv_bp();
    let r_mod = r_mod_bp();
    let garner = garner_inv_b();
    let b_mod = b_mod_bp();
    let garner_bp = garner_inv_bp();
    let bp_mod = bp_mod_b();
    let bs = BASIS_SIZE;

    let mut s = String::with_capacity(4096);

    let _ = writeln!(s, "// RNS-Montgomery precomputed constants");
    let _ = writeln!(s, "constant constexpr uint MONT_NEG_R_INV_B[{bs}] = {{");
    for (i, &v) in neg_r_inv.iter().enumerate() {
        let comma = if i + 1 < bs { "," } else { "" };
        let _ = write!(s, "    {v}u{comma}");
        if (i + 1) % 5 == 0 || i + 1 == bs {
            s.push('\n');
        }
    }
    let _ = writeln!(s, "}};");

    let _ = writeln!(s, "constant constexpr uint MONT_M_INV_BP[{bs}] = {{");
    for (i, &v) in m_inv.iter().enumerate() {
        let comma = if i + 1 < bs { "," } else { "" };
        let _ = write!(s, "    {v}u{comma}");
        if (i + 1) % 5 == 0 || i + 1 == bs {
            s.push('\n');
        }
    }
    let _ = writeln!(s, "}};");

    let _ = writeln!(s, "constant constexpr uint MONT_R_MOD_BP[{bs}] = {{");
    for (i, &v) in r_mod.iter().enumerate() {
        let comma = if i + 1 < bs { "," } else { "" };
        let _ = write!(s, "    {v}u{comma}");
        if (i + 1) % 5 == 0 || i + 1 == bs {
            s.push('\n');
        }
    }
    let _ = writeln!(s, "}};");

    let _ = writeln!(
        s,
        "constant constexpr uint MONT_GARNER_INV_B[{bs}][{bs}] = {{"
    );
    for (i, row) in garner.iter().enumerate() {
        let _ = write!(s, "    {{");
        for (j, &val) in row.iter().enumerate() {
            let comma = if j + 1 < bs { "," } else { "" };
            let _ = write!(s, "{val}{comma}");
        }
        let comma = if i + 1 < bs { "," } else { "" };
        let _ = writeln!(s, "}}{comma}");
    }
    let _ = writeln!(s, "}};");

    let _ = writeln!(s, "constant constexpr uint MONT_B_MOD_BP[{bs}][{bs}] = {{");
    for (i, row) in b_mod.iter().enumerate() {
        let _ = write!(s, "    {{");
        for (j, &val) in row.iter().enumerate() {
            let comma = if j + 1 < bs { "," } else { "" };
            let _ = write!(s, "{val}{comma}");
        }
        let comma = if i + 1 < bs { "," } else { "" };
        let _ = writeln!(s, "}}{comma}");
    }
    let _ = writeln!(s, "}};");

    // B' → B extension constants (for chaining multiplies)
    let _ = writeln!(
        s,
        "constant constexpr uint MONT_GARNER_INV_BP[{bs}][{bs}] = {{"
    );
    for (i, row) in garner_bp.iter().enumerate() {
        let _ = write!(s, "    {{");
        for (j, &val) in row.iter().enumerate() {
            let comma = if j + 1 < bs { "," } else { "" };
            let _ = write!(s, "{val}{comma}");
        }
        let comma = if i + 1 < bs { "," } else { "" };
        let _ = writeln!(s, "}}{comma}");
    }
    let _ = writeln!(s, "}};");

    let _ = writeln!(s, "constant constexpr uint MONT_BP_MOD_B[{bs}][{bs}] = {{");
    for (j, row) in bp_mod.iter().enumerate() {
        let _ = write!(s, "    {{");
        for (i, &val) in row.iter().enumerate() {
            let comma = if i + 1 < bs { "," } else { "" };
            let _ = write!(s, "{val}{comma}");
        }
        let comma = if j + 1 < bs { "," } else { "" };
        let _ = writeln!(s, "}}{comma}");
    }
    let _ = writeln!(s, "}};");

    s
}

/// Generate an RNS-Montgomery reduce kernel.
///
/// Each thread reads all 18 residues per pair, does RNS-Montgomery multiplication
/// (weight × interpolated), and accumulates in B' basis.
fn generate_rns_mont_reduce_kernel(
    num_inputs: usize,
    num_evals: usize,
    d: usize,
    variant: RnsVariant,
    weighted: bool,
) -> String {
    let gs = RNS_REDUCE_GROUP_SIZE;
    let num_simdgroups = gs / SIMD_SIZE;
    let fname = variant.function_name(weighted);
    let is_h2l = matches!(variant, RnsVariant::HighToLow);
    let bs = BASIS_SIZE;
    let np = rns::NUM_PRIMES;

    let mut s = String::with_capacity(32768);

    // Kernel signature
    let _ = writeln!(s, "kernel void {fname}(");
    for k in 0..num_inputs {
        let _ = writeln!(s, "    device const uint* input_{k} [[buffer({k})]],");
    }
    let mut next_buf = num_inputs;
    if weighted {
        let _ = writeln!(s, "    device const uint* weights [[buffer({next_buf})]],");
        next_buf += 1;
    }
    let _ = writeln!(s, "    device uint* partials [[buffer({next_buf})]],");
    next_buf += 1;
    let _ = writeln!(s, "    device const uint* params [[buffer({next_buf})]],");
    let _ = writeln!(s, "    uint gid [[threadgroup_position_in_grid]],");
    let _ = writeln!(s, "    uint lid [[thread_position_in_threadgroup]],");
    let _ = writeln!(s, "    uint num_groups [[threadgroups_per_grid]],");
    let _ = writeln!(s, "    uint simd_lane [[thread_index_in_simdgroup]],");
    let _ = writeln!(s, "    uint simd_id [[simdgroup_index_in_threadgroup]]");
    let _ = writeln!(s, ") {{");

    let _ = writeln!(s, "    uint n_pairs = params[0];");
    s.push('\n');

    // Shared memory for tree reduction: num_evals × BASIS_SIZE × num_simdgroups
    // (we need B' accumulators for each eval point)
    let sh_size = num_evals * bs * num_simdgroups;
    let _ = writeln!(s, "    threadgroup uint sh[{sh_size}];");
    s.push('\n');

    // Per-thread accumulators: num_evals × BASIS_SIZE u32 (B' only)
    for d_idx in 0..num_evals {
        for j in 0..bs {
            let _ = writeln!(s, "    uint acc_{d_idx}_{j} = 0;");
        }
    }
    s.push('\n');

    // Grid-stride loop over pairs
    let _ = writeln!(
        s,
        "    for (uint i = gid * {gs}u + lid; i < n_pairs; i += num_groups * {gs}u) {{"
    );

    // Read all 18 residues of lo/hi for input_0 (P=1 so only one input)
    // SoA layout: input_k[r * n_elements + element_index]
    // For pairs: n_elements = 2 * n_pairs (L2H interleaved) or same with H2L split
    for k in 0..num_inputs {
        for r in 0..np {
            let basis = if r < bs { "b" } else { "bp" };
            let idx = if r < bs { r } else { r - bs };
            if is_h2l {
                let _ = writeln!(
                    s,
                    "        uint lo_{k}_{basis}{idx} = input_{k}[{r}u * n_pairs * 2u + i];"
                );
                let _ = writeln!(
                    s,
                    "        uint hi_{k}_{basis}{idx} = input_{k}[{r}u * n_pairs * 2u + i + n_pairs];"
                );
            } else {
                let _ = writeln!(
                    s,
                    "        uint lo_{k}_{basis}{idx} = input_{k}[{r}u * n_pairs * 2u + 2u * i];"
                );
                let _ = writeln!(
                    s,
                    "        uint hi_{k}_{basis}{idx} = input_{k}[{r}u * n_pairs * 2u + 2u * i + 1u];"
                );
            }
        }
    }
    s.push('\n');

    // Read weight (all 18 residues)
    if weighted {
        for r in 0..np {
            let basis = if r < bs { "b" } else { "bp" };
            let idx = if r < bs { r } else { r - bs };
            let _ = writeln!(
                s,
                "        uint w_{basis}{idx} = weights[{r}u * n_pairs + i];"
            );
        }
        s.push('\n');
    }

    // Compute diff = hi - lo in both bases
    for k in 0..num_inputs {
        for i in 0..bs {
            let _ = writeln!(
                s,
                "        uint diff_{k}_b{i} = rns_sub(hi_{k}_b{i}, lo_{k}_b{i}, B_PRIMES[{i}]);"
            );
        }
        for j in 0..bs {
            let _ = writeln!(
                s,
                "        uint diff_{k}_bp{j} = rns_sub(hi_{k}_bp{j}, lo_{k}_bp{j}, BP_PRIMES[{j}]);"
            );
        }
    }
    s.push('\n');

    // Initialize cur = hi (t=1 evaluation point)
    for k in 0..num_inputs {
        for i in 0..bs {
            let _ = writeln!(s, "        uint cur_{k}_b{i} = hi_{k}_b{i};");
        }
        for j in 0..bs {
            let _ = writeln!(s, "        uint cur_{k}_bp{j} = hi_{k}_bp{j};");
        }
    }
    s.push('\n');

    // Evaluate at each grid point and accumulate
    // t=1: cur = hi
    emit_rns_mont_eval_and_accumulate(&mut s, 0, num_inputs, weighted, bs);

    // t=2..D-1: cur += diff, then evaluate
    for t in 2..d {
        let eval_idx = t - 1;
        // Advance cur
        for k in 0..num_inputs {
            for i in 0..bs {
                let _ = writeln!(
                    s,
                    "        cur_{k}_b{i} = rns_add(cur_{k}_b{i}, diff_{k}_b{i}, B_PRIMES[{i}]);"
                );
            }
            for j in 0..bs {
                let _ = writeln!(
                    s,
                    "        cur_{k}_bp{j} = rns_add(cur_{k}_bp{j}, diff_{k}_bp{j}, BP_PRIMES[{j}]);"
                );
            }
        }
        emit_rns_mont_eval_and_accumulate(&mut s, eval_idx, num_inputs, weighted, bs);
    }

    // t=∞: leading coefficient = diff
    let last_eval = d - 1;
    // Set cur = diff for the last eval
    for k in 0..num_inputs {
        for i in 0..bs {
            let _ = writeln!(s, "        cur_{k}_b{i} = diff_{k}_b{i};");
        }
        for j in 0..bs {
            let _ = writeln!(s, "        cur_{k}_bp{j} = diff_{k}_bp{j};");
        }
    }
    emit_rns_mont_eval_and_accumulate(&mut s, last_eval, num_inputs, weighted, bs);

    let _ = writeln!(s, "    }}"); // close grid-stride loop
    s.push('\n');

    // Simdgroup reduction via modular add (rns_add keeps values < p, no u32 overflow)
    let half_simd = SIMD_SIZE / 2;
    let _ = writeln!(
        s,
        "    for (ushort off = {half_simd}u; off > 0u; off >>= 1u) {{"
    );
    for d_idx in 0..num_evals {
        for j in 0..bs {
            let _ = writeln!(
                s,
                "        acc_{d_idx}_{j} = rns_add(acc_{d_idx}_{j}, simd_shuffle_xor(acc_{d_idx}_{j}, off), BP_PRIMES[{j}]);"
            );
        }
    }
    let _ = writeln!(s, "    }}");
    s.push('\n');

    // Lane 0 writes to shared memory
    let _ = writeln!(s, "    if (simd_lane == 0u) {{");
    for d_idx in 0..num_evals {
        for j in 0..bs {
            let sh_base = (d_idx * bs + j) * num_simdgroups;
            let _ = writeln!(s, "        sh[{sh_base}u + simd_id] = acc_{d_idx}_{j};");
        }
    }
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);");
    s.push('\n');

    // Tree reduction over simdgroup partials (modular add keeps values < p)
    let mut stride = num_simdgroups / 2;
    while stride > 0 {
        let _ = writeln!(s, "    if (lid < {stride}u) {{");
        for d_idx in 0..num_evals {
            for j in 0..bs {
                let sh_base = (d_idx * bs + j) * num_simdgroups;
                let _ = writeln!(
                    s,
                    "        sh[{sh_base}u + lid] = rns_add(sh[{sh_base}u + lid], sh[{sh_base}u + lid + {stride}u], BP_PRIMES[{j}]);"
                );
            }
        }
        let _ = writeln!(s, "    }}");
        if stride > 1 {
            let _ = writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);");
        }
        stride /= 2;
    }
    s.push('\n');

    // Write output
    let _ = writeln!(s, "    if (lid == 0u) {{");
    for d_idx in 0..num_evals {
        for j in 0..bs {
            let sh_base = (d_idx * bs + j) * num_simdgroups;
            let _ = writeln!(
                s,
                "        partials[{j}u * num_groups * {num_evals}u + gid * {num_evals}u + {d_idx}u] = sh[{sh_base}u];"
            );
        }
    }
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "}}");

    s
}

/// Emit an RNS-Montgomery multiply of (a_b, a_bp) × (b_b, b_bp) → result in B' only.
/// Variables are named with the given prefix. Result is stored in `{prefix}_bp{j}`.
fn emit_rns_mont_multiply(
    s: &mut String,
    a_b_prefix: &str,
    a_bp_prefix: &str,
    b_b_prefix: &str,
    b_bp_prefix: &str,
    result_prefix: &str,
    bs: usize,
) {
    // Step 1: component-wise product in B
    for i in 0..bs {
        let _ = writeln!(
            s,
            "            uint t_b{i}_{result_prefix} = rns_mul({a_b_prefix}{i}, {b_b_prefix}{i}, B_PRIMES[{i}], B_C[{i}]);"
        );
    }
    // Step 1b: component-wise product in B'
    for j in 0..bs {
        let _ = writeln!(
            s,
            "            uint t_bp{j}_{result_prefix} = rns_mul({a_bp_prefix}{j}, {b_bp_prefix}{j}, BP_PRIMES[{j}], BP_C[{j}]);"
        );
    }
    // Step 2: q_b = t_b × neg_r_inv mod B
    for i in 0..bs {
        let _ = writeln!(
            s,
            "            uint q_b{i}_{result_prefix} = rns_mul(t_b{i}_{result_prefix}, MONT_NEG_R_INV_B[{i}], B_PRIMES[{i}], B_C[{i}]);"
        );
    }
    // Step 3: base extend q from B → B' via Garner
    for i in 0..bs {
        let _ = writeln!(
            s,
            "            uint gv{i}_{result_prefix} = q_b{i}_{result_prefix};"
        );
    }
    for i in 1..bs {
        for j in 0..i {
            let _ = writeln!(
                s,
                "            gv{i}_{result_prefix} = rns_mul(rns_sub(gv{i}_{result_prefix}, gv{j}_{result_prefix}, B_PRIMES[{i}]), MONT_GARNER_INV_B[{i}][{j}], B_PRIMES[{i}], B_C[{i}]);"
            );
        }
    }
    let last = bs - 1;
    for j in 0..bs {
        let _ = writeln!(
            s,
            "            uint q_bp{j}_{result_prefix} = gv{last}_{result_prefix};"
        );
        for i in (0..last).rev() {
            let _ = writeln!(
                s,
                "            q_bp{j}_{result_prefix} = rns_add(rns_mul(q_bp{j}_{result_prefix}, MONT_B_MOD_BP[{i}][{j}], BP_PRIMES[{j}], BP_C[{j}]), gv{i}_{result_prefix}, BP_PRIMES[{j}]);"
            );
        }
    }
    // Step 4: combine in B'
    for j in 0..bs {
        let _ = writeln!(
            s,
            "            uint qr{j}_{result_prefix} = rns_mul(q_bp{j}_{result_prefix}, MONT_R_MOD_BP[{j}], BP_PRIMES[{j}], BP_C[{j}]);"
        );
        let _ = writeln!(
            s,
            "            uint {result_prefix}_bp{j} = rns_mul(rns_add(t_bp{j}_{result_prefix}, qr{j}_{result_prefix}, BP_PRIMES[{j}]), MONT_M_INV_BP[{j}], BP_PRIMES[{j}], BP_C[{j}]);"
        );
    }
}

/// Emit B' → B base extension (Garner CRT).
/// Input variables: `{prefix}_bp{j}`. Output: `{prefix}_b{i}`.
fn emit_base_extend_bp_to_b(s: &mut String, prefix: &str, bs: usize) {
    // Garner forward pass on B' residues
    for j in 0..bs {
        let _ = writeln!(s, "            uint ev{j}_{prefix} = {prefix}_bp{j};");
    }
    for i in 1..bs {
        for j in 0..i {
            let _ = writeln!(
                s,
                "            ev{i}_{prefix} = rns_mul(rns_sub(ev{i}_{prefix}, ev{j}_{prefix}, BP_PRIMES[{i}]), MONT_GARNER_INV_BP[{i}][{j}], BP_PRIMES[{i}], BP_C[{i}]);"
            );
        }
    }
    // Horner backward evaluation at each m_i
    let last = bs - 1;
    for i in 0..bs {
        let _ = writeln!(s, "            uint {prefix}_b{i} = ev{last}_{prefix};");
        for k in (0..last).rev() {
            let _ = writeln!(
                s,
                "            {prefix}_b{i} = rns_add(rns_mul({prefix}_b{i}, MONT_BP_MOD_B[{k}][{i}], B_PRIMES[{i}], B_C[{i}]), ev{k}_{prefix}, B_PRIMES[{i}]);"
            );
        }
    }
}

/// Emit code for one evaluation point: compute product of D inputs,
/// multiply by weight, accumulate into B' accumulators.
///
/// Uses uniquely-named prefixes at each chain step (`p0`, `p1`, ...)
/// so that D>2 chained multiplies don't collide.
fn emit_rns_mont_eval_and_accumulate(
    s: &mut String,
    eval_idx: usize,
    num_inputs: usize,
    weighted: bool,
    bs: usize,
) {
    let _ = writeln!(s, "        {{ // Eval {eval_idx}");

    if num_inputs == 1 && !weighted {
        for j in 0..bs {
            let _ = writeln!(
                s,
                "            acc_{eval_idx}_{j} = rns_add(acc_{eval_idx}_{j}, cur_0_bp{j}, BP_PRIMES[{j}]);"
            );
        }
        let _ = writeln!(s, "        }}");
        s.push('\n');
        return;
    }

    // Compute Π_k cur_k. Track the prefix holding the running product.
    let product_prefix = if num_inputs == 1 {
        "cur_0".to_string()
    } else {
        // cur_0 × cur_1 → p0 (result in B' only)
        emit_rns_mont_multiply(s, "cur_0_b", "cur_0_bp", "cur_1_b", "cur_1_bp", "p0", bs);
        let mut cur_prefix = "p0".to_string();

        // Chain: p_{k-1} × cur_k → p_k  for k=2..num_inputs-1
        for k in 2..num_inputs {
            let next_prefix = format!("p{}", k - 1);
            emit_base_extend_bp_to_b(s, &cur_prefix, bs);
            emit_rns_mont_multiply(
                s,
                &format!("{cur_prefix}_b"),
                &format!("{cur_prefix}_bp"),
                &format!("cur_{k}_b"),
                &format!("cur_{k}_bp"),
                &next_prefix,
                bs,
            );
            cur_prefix = next_prefix;
        }
        cur_prefix
    };

    // Multiply by weight if needed
    let final_prefix = if weighted {
        if num_inputs > 1 {
            emit_base_extend_bp_to_b(s, &product_prefix, bs);
        }
        emit_rns_mont_multiply(
            s,
            "w_b",
            "w_bp",
            &format!("{product_prefix}_b"),
            &format!("{product_prefix}_bp"),
            "wres",
            bs,
        );
        "wres".to_string()
    } else {
        product_prefix
    };

    // Accumulate in B'
    for j in 0..bs {
        let _ = writeln!(
            s,
            "            acc_{eval_idx}_{j} = rns_add(acc_{eval_idx}_{j}, {final_prefix}_bp{j}, BP_PRIMES[{j}]);"
        );
    }

    let _ = writeln!(s, "        }}");
    s.push('\n');
}
