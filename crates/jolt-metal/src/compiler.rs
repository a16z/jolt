//! Kernel compiler: `jolt-ir::KernelDescriptor` → MSL source → `MetalKernel`.
//!
//! Two compilation paths:
//!
//! - **ProductSum**: Generates a fully-unrolled MSL kernel evaluating
//!   `Σ_g Π_j p_{g,j}(t)` on the Toom-Cook grid `{1, 2, ..., D-1, ∞}`.
//!   Uses incremental interpolation (`cur += diff` per grid step) to avoid
//!   expensive `fr_mul` for small integer grid scalars.
//!
//! - **Custom**: Walks the `jolt-ir::Expr` DAG via `ExprVisitor`, emitting
//!   MSL assignments in SSA form. Challenge values are read from a runtime
//!   `device const Fr* challenges` buffer — not baked as MSL constants. This
//!   makes pipeline compilation deterministic per kernel shape, enabling AOT
//!   caching.
//!   Uses incremental interpolation for the standard grid `{0, 2, ..., degree}`
//!   (skipping `t=1`, which is derived from the sumcheck claim).
//!
//! Five kernel variants are compiled from each descriptor:
//! - **L2H**: Weighted, pairs `(buf[2i], buf[2i+1])` — interleaved
//! - **H2L**: Weighted, pairs `(buf[i], buf[i+half])` — split-half
//! - **Tensor**: Weighted, L2H with two-buffer weight: `outer[i>>log] * inner[i&mask]`
//! - **L2H_unw**: Unweighted L2H — no weight buffer, no weight multiply
//! - **H2L_unw**: Unweighted H2L — no weight buffer, no weight multiply
//!
//! All reduce kernels use `WideAcc` (576-bit accumulators) for deferred
//! modular reduction: products/sums are accumulated as wide integers and
//! reduced to `Fr` once per thread at the end of the grid-stride loop.
//! The simdgroup + shared-memory tree reduction then operates on `Fr` values.

use std::fmt::Write;
use std::sync::Arc;

use jolt_ir::{Expr, ExprVisitor, KernelDescriptor, KernelShape, Var};
use metal::CompileOptions;

use crate::kernel::CachedPipelines;
use crate::shaders::{build_source_with_mode, make_pipeline, SHADER_BN254_FR, SHADER_WIDE_ACC};

/// Controls Metal shader compilation strategy.
#[derive(Clone, Copy, Debug, Default)]
pub enum CompileMode {
    /// Full LLVM inlining — maximum GPU throughput, slow compilation.
    #[default]
    Performance,
    /// Noinline on field arithmetic — fast compilation, minor GPU overhead.
    /// Reduces D=8 kernel compilation from ~3 minutes to seconds.
    FastCompile,
}

/// Threadgroup size for reduce kernels. Power of 2.
pub(crate) const REDUCE_GROUP_SIZE: usize = 256;

/// Maximum threadgroups per reduce dispatch.
pub(crate) const MAX_REDUCE_GROUPS: usize = 256;

/// Apple GPU SIMD width (32 threads per simdgroup on M-series).
const SIMD_SIZE: usize = 32;

/// WideAcc limb count (18 × u32 = 576 bits). Must match `ACC_LIMBS` in wide_accumulator.metal.
const ACC_LIMBS: usize = 18;

/// ProductSum D threshold for split-pass kernel generation. Kernels with
/// `num_inputs_per_product >= SPLIT_PASS_THRESHOLD` use multi-pass streaming
/// to reduce register pressure and improve GPU occupancy.
///
/// Currently disabled (set above any realistic D). Benchmarking showed the 4×
/// bandwidth re-read cost outweighs occupancy gains — single-pass is faster.
const SPLIT_PASS_THRESHOLD: usize = 1024;

/// Grid points processed per pass in split-pass mode. Each pass only needs
/// `chunk * 18` registers for WideAcc instead of `D * 18`.
const SPLIT_PASS_CHUNK: usize = 2;

/// Which pair-reading and weight strategy a kernel uses.
#[derive(Clone, Copy)]
enum KernelVariant {
    /// Interleaved pairs: `lo = buf[2i], hi = buf[2i+1]`, single weight buffer.
    LowToHigh,
    /// Split-half pairs: `lo = buf[i], hi = buf[i + n_pairs]`, single weight buffer.
    HighToLow,
    /// Interleaved pairs with tensor weight: `w = outer[i>>log] * inner[i&mask]`.
    Tensor,
}

impl KernelVariant {
    fn function_name(self, weighted: bool) -> &'static str {
        match (self, weighted) {
            (Self::LowToHigh, true) => "reduce_kernel_l2h",
            (Self::HighToLow, true) => "reduce_kernel_h2l",
            (Self::Tensor, true) => "reduce_kernel_tensor",
            (Self::LowToHigh, false) => "reduce_kernel_l2h_unw",
            (Self::HighToLow, false) => "reduce_kernel_h2l_unw",
            (Self::Tensor, false) => unreachable!("tensor has no unweighted variant"),
        }
    }
}

/// Output of MSL generation — the full source string plus kernel metadata.
pub(crate) struct GeneratedMsl {
    pub source: String,
    pub num_inputs: usize,
    pub num_evals: usize,
    /// Whether the kernel signature includes a `challenges` buffer parameter.
    pub has_challenges: bool,
}

/// Generate the full MSL source for a kernel descriptor.
///
/// This is the deterministic, challenge-independent MSL generation. The
/// returned source can be hashed to produce a cache key.
pub(crate) fn generate_msl(descriptor: &KernelDescriptor, mode: CompileMode) -> GeneratedMsl {
    let num_inputs = descriptor.num_inputs();
    let num_evals = descriptor.num_evals();

    let use_split_pass = matches!(
        &descriptor.shape,
        KernelShape::ProductSum { num_inputs_per_product, .. }
            if *num_inputs_per_product >= SPLIT_PASS_THRESHOLD
    );

    let mut msl = String::with_capacity(65536);
    let mut has_challenges = false;

    if use_split_pass {
        let (d, p) = match &descriptor.shape {
            KernelShape::ProductSum {
                num_inputs_per_product,
                num_products,
            } => (*num_inputs_per_product, *num_products),
            _ => unreachable!(),
        };

        for variant in [
            KernelVariant::LowToHigh,
            KernelVariant::HighToLow,
            KernelVariant::Tensor,
        ] {
            msl.push_str(&generate_split_pass_reduce_kernel(
                num_inputs, num_evals, d, p, variant, true,
            ));
            msl.push('\n');
        }
        for variant in [KernelVariant::LowToHigh, KernelVariant::HighToLow] {
            msl.push_str(&generate_split_pass_reduce_kernel(
                num_inputs, num_evals, d, p, variant, false,
            ));
            msl.push('\n');
        }
    } else {
        // Single-pass path for small D and non-ProductSum shapes.
        let (eval_body_weighted, eval_body_unweighted, weight_folded) = match &descriptor.shape {
            KernelShape::ProductSum {
                num_inputs_per_product,
                num_products,
            } => {
                let d = *num_inputs_per_product;
                let p = *num_products;
                if d == 8 {
                    // Toom-Cook balanced binary splitting: 30 fr_mul vs 58
                    // naive. No weight folding — same body for weighted
                    // and unweighted variants.
                    let body = generate_toom_cook_body(d, p);
                    (body.clone(), body, false)
                } else {
                    let fold = d > 2 * p;
                    let body = generate_product_sum_body(d, p, fold);
                    let body_unw = if fold {
                        generate_product_sum_body(d, p, false)
                    } else {
                        body.clone()
                    };
                    (body, body_unw, fold)
                }
            }
            KernelShape::EqProduct => {
                let body = r"
        evals[0] = fr_mul(lo[0], lo[1]);
        Fr a2 = fr_sub(fr_add(hi[0], hi[0]), lo[0]);
        Fr b2 = fr_sub(fr_add(hi[1], hi[1]), lo[1]);
        evals[1] = fr_mul(a2, b2);"
                    .to_string();
                (body.clone(), body, false)
            }
            KernelShape::HammingBooleanity => {
                let body = r"
        Fr d_eq = fr_sub(hi[0], lo[0]);
        Fr d_h = fr_sub(hi[1], lo[1]);
        evals[0] = fr_mul(fr_mul(lo[0], lo[1]), fr_sub(lo[1], fr_one()));
        Fr eq_val = fr_add(hi[0], d_eq);
        Fr h_val = fr_add(hi[1], d_h);
        evals[1] = fr_mul(fr_mul(eq_val, h_val), fr_sub(h_val, fr_one()));
        eq_val = fr_add(eq_val, d_eq);
        h_val = fr_add(h_val, d_h);
        evals[2] = fr_mul(fr_mul(eq_val, h_val), fr_sub(h_val, fr_one()));"
                    .to_string();
                (body.clone(), body, false)
            }
            KernelShape::Custom { expr, num_inputs } => {
                let body = generate_custom_body(expr, *num_inputs, descriptor.degree);
                (body.clone(), body, false)
            }
        };

        // Only emit the challenges buffer parameter if the eval body references it.
        // Custom kernels without Var::Challenge nodes don't need the buffer.
        has_challenges = eval_body_weighted.contains("challenges[");

        for variant in [
            KernelVariant::LowToHigh,
            KernelVariant::HighToLow,
            KernelVariant::Tensor,
        ] {
            msl.push_str(&generate_reduce_kernel(
                num_inputs,
                num_evals,
                &eval_body_weighted,
                variant,
                weight_folded,
                true,
                has_challenges,
            ));
            msl.push('\n');
        }
        for variant in [KernelVariant::LowToHigh, KernelVariant::HighToLow] {
            msl.push_str(&generate_reduce_kernel(
                num_inputs,
                num_evals,
                &eval_body_unweighted,
                variant,
                false,
                false,
                has_challenges,
            ));
            msl.push('\n');
        }
    }

    let noinline = matches!(mode, CompileMode::FastCompile);
    let source = build_source_with_mode(&[SHADER_BN254_FR, SHADER_WIDE_ACC, &msl], noinline);

    GeneratedMsl {
        source,
        num_inputs,
        num_evals,
        has_challenges,
    }
}

/// Compile a generated MSL source into pipeline states.
pub(crate) fn compile_msl(device: &metal::Device, msl: &GeneratedMsl) -> Arc<CachedPipelines> {
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(&msl.source, &options)
        .unwrap_or_else(|e| panic!("reduce kernel MSL compilation failed: {e}"));

    Arc::new(CachedPipelines {
        pipeline_l2h: make_pipeline(device, &library, "reduce_kernel_l2h"),
        pipeline_h2l: make_pipeline(device, &library, "reduce_kernel_h2l"),
        pipeline_tensor: make_pipeline(device, &library, "reduce_kernel_tensor"),
        pipeline_l2h_unw: make_pipeline(device, &library, "reduce_kernel_l2h_unw"),
        pipeline_h2l_unw: make_pipeline(device, &library, "reduce_kernel_h2l_unw"),
        num_evals: msl.num_evals,
        num_inputs: msl.num_inputs,
        has_challenges: msl.has_challenges,
    })
}

/// How the per-thread accumulation works for a given kernel configuration.
#[derive(Clone, Copy)]
enum AccumulationStrategy {
    /// `acc_fmadd(wide_acc, w, eval)` — weight multiply inside accumulator.
    /// Result after `acc_reduce` is already in Montgomery form.
    WeightedFmadd,
    /// `acc_add_fr(wide_acc, eval)` — no weight multiply (weight is either
    /// baked into the eval body or implicitly one).
    /// Result after `acc_reduce` is in standard form; needs `fr_to_mont`.
    DirectAdd,
}

/// Generate a reduce kernel for a specific pair-reading variant.
///
/// All variants share the same eval body, accumulation, simdgroup reduction,
/// and shared-memory tree. They differ only in:
/// - Kernel function name and buffer signature
/// - Pair reading pattern (interleaved vs split-half)
/// - Weight computation (single buffer, tensor product, or none)
/// - Accumulation strategy (WideAcc fmadd vs WideAcc add)
fn generate_reduce_kernel(
    num_inputs: usize,
    num_evals: usize,
    eval_body: &str,
    variant: KernelVariant,
    weight_folded: bool,
    weighted: bool,
    has_challenges: bool,
) -> String {
    let gs = REDUCE_GROUP_SIZE;
    let num_simdgroups = gs / SIMD_SIZE;
    let fname = variant.function_name(weighted);
    let is_tensor = matches!(variant, KernelVariant::Tensor);

    // Determine accumulation strategy.
    // - Unweighted: always DirectAdd (no weight to multiply)
    // - Weighted + folded: DirectAdd (weight baked into eval body)
    // - Weighted + not folded: WeightedFmadd (explicit w * eval)
    let strategy = if !weighted || weight_folded {
        AccumulationStrategy::DirectAdd
    } else {
        AccumulationStrategy::WeightedFmadd
    };

    let mut s = String::with_capacity(8192);

    // Kernel signature
    let _ = writeln!(s, "kernel void {fname}(");
    for k in 0..num_inputs {
        let _ = writeln!(s, "    device const Fr* input_{k} [[buffer({k})]],");
    }

    let mut next_buf = num_inputs;
    if weighted {
        if is_tensor {
            let _ = writeln!(
                s,
                "    device const Fr* outer_weights [[buffer({next_buf})]],",
            );
            next_buf += 1;
            let _ = writeln!(
                s,
                "    device const Fr* inner_weights [[buffer({next_buf})]],",
            );
            next_buf += 1;
        } else {
            let _ = writeln!(s, "    device const Fr* weights [[buffer({next_buf})]],",);
            next_buf += 1;
        }
    }
    if has_challenges {
        let _ = writeln!(s, "    device const Fr* challenges [[buffer({next_buf})]],",);
        next_buf += 1;
    }
    let _ = writeln!(s, "    device Fr* partials [[buffer({next_buf})]],");
    next_buf += 1;
    let _ = writeln!(s, "    device const uint* params [[buffer({next_buf})]],",);
    let _ = writeln!(s, "    uint gid [[threadgroup_position_in_grid]],");
    let _ = writeln!(s, "    uint lid [[thread_position_in_threadgroup]],");
    let _ = writeln!(s, "    uint num_groups [[threadgroups_per_grid]],");
    let _ = writeln!(s, "    uint simd_lane [[thread_index_in_simdgroup]],");
    let _ = writeln!(s, "    uint simd_id [[simdgroup_index_in_threadgroup]]");
    let _ = writeln!(s, ") {{");

    // Read params
    let _ = writeln!(s, "    uint n_pairs = params[0];");
    if is_tensor {
        let _ = writeln!(s, "    uint inner_log = params[1];");
        let _ = writeln!(s, "    uint inner_mask = params[2];");
    }
    s.push('\n');

    // Per-thread WideAcc accumulators
    let _ = writeln!(s, "    WideAcc wide_acc[{num_evals}];");
    for d in 0..num_evals {
        let _ = writeln!(s, "    wide_acc[{d}] = acc_zero();");
    }
    s.push('\n');

    // Grid-stride loop
    let _ = writeln!(
        s,
        "    for (uint i = gid * {gs}u + lid; i < n_pairs; i += num_groups * {gs}u) {{"
    );

    // Read pairs
    let _ = writeln!(s, "        Fr lo[{num_inputs}], hi[{num_inputs}];");
    if matches!(variant, KernelVariant::HighToLow) {
        for k in 0..num_inputs {
            let _ = writeln!(
                s,
                "        lo[{k}] = input_{k}[i]; hi[{k}] = input_{k}[i + n_pairs];"
            );
        }
    } else {
        for k in 0..num_inputs {
            let _ = writeln!(
                s,
                "        lo[{k}] = input_{k}[2u * i]; hi[{k}] = input_{k}[2u * i + 1u];"
            );
        }
    }
    s.push('\n');

    // Read weight (weighted variants only)
    if weighted {
        if is_tensor {
            let _ = writeln!(
                s,
                "        Fr w = fr_mul(outer_weights[i >> inner_log], inner_weights[i & inner_mask]);"
            );
        } else {
            let _ = writeln!(s, "        Fr w = weights[i];");
        }
        s.push('\n');
    }

    // Kernel-specific evaluation
    let _ = writeln!(s, "        Fr evals[{num_evals}];");
    let _ = write!(s, "{eval_body}");
    s.push('\n');

    // Accumulation into WideAcc
    match strategy {
        AccumulationStrategy::WeightedFmadd => {
            for d in 0..num_evals {
                let _ = writeln!(s, "        acc_fmadd(wide_acc[{d}], w, evals[{d}]);");
            }
        }
        AccumulationStrategy::DirectAdd => {
            if weight_folded && weighted {
                // Weight already folded into eval body — evals are weighted.
                for d in 0..num_evals {
                    let _ = writeln!(s, "        acc_add_fr(wide_acc[{d}], evals[{d}]);");
                }
            } else {
                // Unweighted — evals are raw.
                for d in 0..num_evals {
                    let _ = writeln!(s, "        acc_add_fr(wide_acc[{d}], evals[{d}]);");
                }
            }
        }
    }
    let _ = writeln!(s, "    }}");
    s.push('\n');

    // Shuffle the 18-limb WideAcc across the simdgroup using acc_merge.
    // Only lane 0 of each simdgroup calls the expensive acc_reduce (8 CIOS
    // rounds). This saves 31 acc_reduce calls per simdgroup (248 per
    // threadgroup of 256 threads).
    let half_simd = SIMD_SIZE / 2;
    let _ = writeln!(
        s,
        "    for (ushort _off = {half_simd}u; _off > 0u; _off >>= 1u) {{"
    );
    for d in 0..num_evals {
        let _ = writeln!(s, "        {{ WideAcc _o;");
        for l in 0..ACC_LIMBS {
            let _ = writeln!(
                s,
                "        _o.limbs[{l}] = simd_shuffle_down(wide_acc[{d}].limbs[{l}], _off);"
            );
        }
        let _ = writeln!(s, "        acc_merge(wide_acc[{d}], _o); }}");
    }
    let _ = writeln!(s, "    }}");
    s.push('\n');

    // Lane 0 of each simdgroup reduces WideAcc → Fr and writes to shared memory
    let needs_to_mont = matches!(strategy, AccumulationStrategy::DirectAdd);
    let sh_size = num_evals * num_simdgroups;
    let _ = writeln!(s, "    threadgroup Fr sh[{sh_size}];");
    let _ = writeln!(s, "    if (simd_lane == 0u) {{");
    for d in 0..num_evals {
        if needs_to_mont {
            let _ = writeln!(
                s,
                "        sh[{}u + simd_id] = fr_to_mont(acc_reduce(wide_acc[{d}]));",
                d * num_simdgroups
            );
        } else {
            let _ = writeln!(
                s,
                "        sh[{}u + simd_id] = acc_reduce(wide_acc[{d}]);",
                d * num_simdgroups
            );
        }
    }
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);");
    s.push('\n');

    // Tree reduction over simdgroup partials (8 → 1).
    // All threads must reach each threadgroup_barrier — never put barriers
    // inside divergent branches. Only the active lanes do the fr_add.
    let mut stride = num_simdgroups / 2;
    while stride > 0 {
        let _ = writeln!(s, "    if (lid < {stride}u) {{");
        for d in 0..num_evals {
            let base = d * num_simdgroups;
            let _ = writeln!(
                s,
                "        sh[{base}u + lid] = fr_add(sh[{base}u + lid], sh[{base}u + lid + {stride}u]);"
            );
        }
        let _ = writeln!(s, "    }}");
        if stride > 1 {
            let _ = writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);");
        }
        stride /= 2;
    }
    s.push('\n');

    // Write final partials
    let _ = writeln!(s, "    if (lid == 0u) {{");
    for d in 0..num_evals {
        let _ = writeln!(
            s,
            "        partials[gid * {num_evals}u + {d}u] = sh[{}u];",
            d * num_simdgroups
        );
    }
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "}}");

    s
}

/// Generate a split-pass reduce kernel for high-D ProductSum (D ≥ SPLIT_PASS_THRESHOLD).
///
/// Instead of loading all D lo/hi arrays at once (which requires ~500 registers
/// for D=8 and kills GPU occupancy), processes grid points in passes of
/// `SPLIT_PASS_CHUNK`. Each pass:
/// 1. Streams through factors one at a time (lo/hi scoped per-factor)
/// 2. Accumulates into only `chunk` WideAcc accumulators (not D)
/// 3. Performs simdgroup + tree reduction
/// 4. Writes `chunk` partial results
///
/// Register pressure drops from ~500 to ~130 per thread, improving occupancy
/// from ~0.5 to ~2 simdgroups per EU (4× improvement).
///
/// Tradeoff: each factor's lo/hi is re-loaded once per pass (D/chunk passes
/// total). For D=8, this is 4× the bandwidth. Since D=8 is compute-bound,
/// the occupancy improvement dominates.
fn generate_split_pass_reduce_kernel(
    num_inputs: usize,
    num_evals: usize,
    d: usize,
    p: usize,
    variant: KernelVariant,
    weighted: bool,
) -> String {
    let gs = REDUCE_GROUP_SIZE;
    let num_simdgroups = gs / SIMD_SIZE;
    let fname = variant.function_name(weighted);
    let is_tensor = matches!(variant, KernelVariant::Tensor);
    let chunk = SPLIT_PASS_CHUNK;
    let num_passes = num_evals.div_ceil(chunk);

    // No weight folding in split-pass mode: weighted always uses acc_fmadd.
    let strategy = if weighted {
        AccumulationStrategy::WeightedFmadd
    } else {
        AccumulationStrategy::DirectAdd
    };
    let needs_to_mont = matches!(strategy, AccumulationStrategy::DirectAdd);

    let mut s = String::with_capacity(65536);

    // Kernel signature (identical to single-pass)
    let _ = writeln!(s, "kernel void {fname}(");
    for k in 0..num_inputs {
        let _ = writeln!(s, "    device const Fr* input_{k} [[buffer({k})]],");
    }
    let mut next_buf = num_inputs;
    if weighted {
        if is_tensor {
            let _ = writeln!(
                s,
                "    device const Fr* outer_weights [[buffer({next_buf})]],",
            );
            next_buf += 1;
            let _ = writeln!(
                s,
                "    device const Fr* inner_weights [[buffer({next_buf})]],",
            );
            next_buf += 1;
        } else {
            let _ = writeln!(s, "    device const Fr* weights [[buffer({next_buf})]],");
            next_buf += 1;
        }
    }
    let _ = writeln!(s, "    device Fr* partials [[buffer({next_buf})]],");
    next_buf += 1;
    let _ = writeln!(s, "    device const uint* params [[buffer({next_buf})]],");
    let _ = writeln!(s, "    uint gid [[threadgroup_position_in_grid]],");
    let _ = writeln!(s, "    uint lid [[thread_position_in_threadgroup]],");
    let _ = writeln!(s, "    uint num_groups [[threadgroups_per_grid]],");
    let _ = writeln!(s, "    uint simd_lane [[thread_index_in_simdgroup]],");
    let _ = writeln!(s, "    uint simd_id [[simdgroup_index_in_threadgroup]]");
    let _ = writeln!(s, ") {{");

    let _ = writeln!(s, "    uint n_pairs = params[0];");
    if is_tensor {
        let _ = writeln!(s, "    uint inner_log = params[1];");
        let _ = writeln!(s, "    uint inner_mask = params[2];");
    }
    s.push('\n');

    // Shared memory reused across passes (only `chunk * num_simdgroups` slots).
    let sh_size = chunk * num_simdgroups;
    let _ = writeln!(s, "    threadgroup Fr sh[{sh_size}];");
    s.push('\n');

    // Per-pass codegen
    for pass_idx in 0..num_passes {
        let base = pass_idx * chunk;
        let chunk_size = chunk.min(num_evals - base);

        let _ = writeln!(
            s,
            "    {{ // Pass {pass_idx}: eval[{base}..{}]",
            base + chunk_size - 1
        );

        // Per-pass WideAcc (scoped to this block → freed between passes)
        for c in 0..chunk_size {
            let _ = writeln!(s, "    WideAcc wa_{c} = acc_zero();");
        }
        s.push('\n');

        // Grid-stride loop
        let _ = writeln!(
            s,
            "    for (uint i = gid * {gs}u + lid; i < n_pairs; i += num_groups * {gs}u) {{"
        );

        // Read weight
        if weighted {
            if is_tensor {
                let _ = writeln!(
                    s,
                    "        Fr w = fr_mul(outer_weights[i >> inner_log], inner_weights[i & inner_mask]);"
                );
            } else {
                let _ = writeln!(s, "        Fr w = weights[i];");
            }
        }

        // Sum accumulators for P > 1
        if p > 1 {
            for c in 0..chunk_size {
                let _ = writeln!(s, "        Fr sum_{c} = fr_zero();");
            }
        }

        // For each product group g
        for g in 0..p {
            if p > 1 {
                let _ = writeln!(s, "        {{");
            }

            // Product accumulators (set by first factor, multiplied by rest)
            for c in 0..chunk_size {
                let _ = writeln!(s, "        Fr prod_{c};");
            }

            // Stream through factors — only one lo/hi/diff live at a time
            for j in 0..d {
                let input_idx = g * d + j;
                let _ = writeln!(s, "        {{");

                // Load pair
                if matches!(variant, KernelVariant::HighToLow) {
                    let _ = writeln!(s, "            Fr lo = input_{input_idx}[i];");
                    let _ = writeln!(s, "            Fr hi = input_{input_idx}[i + n_pairs];");
                } else {
                    let _ = writeln!(s, "            Fr lo = input_{input_idx}[2u * i];");
                    let _ = writeln!(s, "            Fr hi = input_{input_idx}[2u * i + 1u];");
                }
                let _ = writeln!(s, "            Fr diff = fr_sub(hi, lo);");

                // Compute val at each grid point in this chunk
                for c in 0..chunk_size {
                    let eval_idx = base + c;
                    if eval_idx == d - 1 {
                        // t = ∞: leading coefficient
                        let _ = writeln!(s, "            Fr val_{c} = diff;");
                    } else if eval_idx == 0 {
                        // t = 1: val = hi
                        let _ = writeln!(s, "            Fr val_{c} = hi;");
                    } else if c > 0 && (base + c - 1) < d - 1 {
                        // Incremental: val_c = val_{c-1} + diff
                        let _ =
                            writeln!(s, "            Fr val_{c} = fr_add(val_{}, diff);", c - 1);
                    } else {
                        // From scratch: hi + eval_idx * diff
                        let _ = writeln!(s, "            Fr val_{c} = hi;");
                        for _ in 0..eval_idx {
                            let _ = writeln!(s, "            val_{c} = fr_add(val_{c}, diff);");
                        }
                    }
                }

                // First factor: assign. Subsequent: unreduced multiply.
                for c in 0..chunk_size {
                    if j == 0 {
                        let _ = writeln!(s, "            prod_{c} = val_{c};");
                    } else {
                        let _ = writeln!(
                            s,
                            "            prod_{c} = fr_mul_unreduced(prod_{c}, val_{c});"
                        );
                    }
                }

                let _ = writeln!(s, "        }}");
            }

            // Accumulate product into sum (P > 1) — reduce before fr_add
            if p > 1 {
                for c in 0..chunk_size {
                    let _ = writeln!(s, "        sum_{c} = fr_add(sum_{c}, fr_reduce(prod_{c}));");
                }
                let _ = writeln!(s, "        }}");
            }
        }

        // Accumulate into WideAcc
        let acc_var = if p > 1 { "sum" } else { "prod" };
        match strategy {
            AccumulationStrategy::WeightedFmadd => {
                for c in 0..chunk_size {
                    let _ = writeln!(s, "        acc_fmadd(wa_{c}, w, {acc_var}_{c});");
                }
            }
            AccumulationStrategy::DirectAdd => {
                for c in 0..chunk_size {
                    let _ = writeln!(s, "        acc_add_fr(wa_{c}, {acc_var}_{c});");
                }
            }
        }
        let _ = writeln!(s, "    }}"); // close grid-stride loop
        s.push('\n');

        // Simdgroup WideAcc reduction (same as single-pass)
        let half_simd = SIMD_SIZE / 2;
        let _ = writeln!(
            s,
            "    for (ushort _off = {half_simd}u; _off > 0u; _off >>= 1u) {{"
        );
        for c in 0..chunk_size {
            let _ = writeln!(s, "        {{ WideAcc _o;");
            for l in 0..ACC_LIMBS {
                let _ = writeln!(
                    s,
                    "        _o.limbs[{l}] = simd_shuffle_down(wa_{c}.limbs[{l}], _off);"
                );
            }
            let _ = writeln!(s, "        acc_merge(wa_{c}, _o); }}");
        }
        let _ = writeln!(s, "    }}");
        s.push('\n');

        // Lane 0: acc_reduce → shared memory
        let _ = writeln!(s, "    if (simd_lane == 0u) {{");
        for c in 0..chunk_size {
            let sh_base = c * num_simdgroups;
            if needs_to_mont {
                let _ = writeln!(
                    s,
                    "        sh[{sh_base}u + simd_id] = fr_to_mont(acc_reduce(wa_{c}));",
                );
            } else {
                let _ = writeln!(s, "        sh[{sh_base}u + simd_id] = acc_reduce(wa_{c});",);
            }
        }
        let _ = writeln!(s, "    }}");
        let _ = writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);");
        s.push('\n');

        // Tree reduction over simdgroup partials.
        // All threads must reach each barrier — no outer `if (lid < N)` guard.
        let mut stride = num_simdgroups / 2;
        while stride > 0 {
            let _ = writeln!(s, "    if (lid < {stride}u) {{");
            for c in 0..chunk_size {
                let sh_base = c * num_simdgroups;
                let _ = writeln!(
                    s,
                    "        sh[{sh_base}u + lid] = fr_add(sh[{sh_base}u + lid], sh[{sh_base}u + lid + {stride}u]);"
                );
            }
            let _ = writeln!(s, "    }}");
            if stride > 1 {
                let _ = writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
            stride /= 2;
        }
        s.push('\n');

        // Write partials for this pass's eval indices
        let _ = writeln!(s, "    if (lid == 0u) {{");
        for c in 0..chunk_size {
            let sh_base = c * num_simdgroups;
            let eval_out = base + c;
            let _ = writeln!(
                s,
                "        partials[gid * {num_evals}u + {eval_out}u] = sh[{sh_base}u];"
            );
        }
        let _ = writeln!(s, "    }}");

        let _ = writeln!(s, "    }}"); // close pass block

        // Barrier before next pass (shared memory reuse)
        if pass_idx < num_passes - 1 {
            let _ = writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);");
        }
        s.push('\n');
    }

    let _ = writeln!(s, "}}");

    s
}

/// Generate Toom-Cook MSL evaluation body for ProductSum D=4 or D=8.
///
/// Uses balanced binary splitting to reduce field multiplications:
/// - D=4: 2x2 split → 10 fr_mul (vs 12 naive)
/// - D=8: 4x4 split → 30 fr_mul (vs 56 naive)
///
/// Extrapolation uses only fr_add/fr_sub (no mul_u64 needed on GPU).
/// Weight is NOT folded into the eval body — the caller uses
/// `AccumulationStrategy::WeightedFmadd` for weighted kernels.
fn generate_toom_cook_body(d: usize, p: usize) -> String {
    match d {
        4 => generate_toom_cook_d4(p),
        8 => generate_toom_cook_d8(p),
        _ => unreachable!("Toom-Cook only implemented for D=4 and D=8"),
    }
}

/// Emit MSL for the degree-2 sub-product of two linear polynomials.
///
/// Given inputs at indices `i0` and `i1` in the `lo`/`hi` arrays, produces
/// three sub-product evaluations `{prefix}_1`, `{prefix}_2`, `{prefix}_inf`
/// at points {1, 2, ∞}. Uses 3 fr_mul.
fn emit_eval_linear_prod_2(
    s: &mut String,
    prefix: &str,
    lo0: &str,
    hi0: &str,
    lo1: &str,
    hi1: &str,
) {
    let _ = writeln!(s, "        Fr {prefix}_s0 = fr_sub({hi0}, {lo0});");
    let _ = writeln!(s, "        Fr {prefix}_v2_0 = fr_add({prefix}_s0, {hi0});");
    let _ = writeln!(s, "        Fr {prefix}_s1 = fr_sub({hi1}, {lo1});");
    let _ = writeln!(s, "        Fr {prefix}_v2_1 = fr_add({prefix}_s1, {hi1});");
    let _ = writeln!(s, "        Fr {prefix}_1 = fr_mul({hi0}, {hi1});");
    let _ = writeln!(
        s,
        "        Fr {prefix}_2 = fr_mul({prefix}_v2_0, {prefix}_v2_1);"
    );
    let _ = writeln!(
        s,
        "        Fr {prefix}_inf = fr_mul({prefix}_s0, {prefix}_s1);"
    );
}

/// Emit MSL for `ex2` extrapolation: `result = 2*(f1 + finf) - f0`.
fn emit_ex2(s: &mut String, result: &str, f0: &str, f1: &str, finf: &str) {
    let _ = writeln!(s, "        Fr {result}_sum = fr_add({f1}, {finf});");
    let _ = writeln!(
        s,
        "        Fr {result} = fr_sub(fr_add({result}_sum, {result}_sum), {f0});"
    );
}

/// Emit MSL for `ex4_2`: returns two extrapolated values (f4, f5) from
/// 4 consecutive evaluations `f[0..4]` and `6*f_inf`.
fn emit_ex4_2(s: &mut String, r4: &str, r5: &str, f: [&str; 4], finf6: &str) {
    // f4 = 2*(2*(finf6 + f[3] - f[2] + f[1]) - f[2]) - f[0]
    let _ = writeln!(s, "        Fr {r4}_f3m2 = fr_sub({}, {});", f[3], f[2]);
    let _ = writeln!(
        s,
        "        Fr {r4}_t = fr_add(fr_add({finf6}, {r4}_f3m2), {});",
        f[1]
    );
    let _ = writeln!(s, "        {r4}_t = fr_add({r4}_t, {r4}_t);");
    let _ = writeln!(s, "        {r4}_t = fr_sub({r4}_t, {});", f[2]);
    let _ = writeln!(s, "        {r4}_t = fr_add({r4}_t, {r4}_t);");
    let _ = writeln!(s, "        Fr {r4} = fr_sub({r4}_t, {});", f[0]);

    // f5 = 2*(2*(f4 - f3m2 + finf6) - f[3]) - f[1]
    let _ = writeln!(
        s,
        "        Fr {r5}_t = fr_add(fr_sub({r4}, {r4}_f3m2), {finf6});"
    );
    let _ = writeln!(s, "        {r5}_t = fr_add({r5}_t, {r5}_t);");
    let _ = writeln!(s, "        {r5}_t = fr_sub({r5}_t, {});", f[3]);
    let _ = writeln!(s, "        {r5}_t = fr_add({r5}_t, {r5}_t);");
    let _ = writeln!(s, "        Fr {r5} = fr_sub({r5}_t, {});", f[1]);
}

/// Emit MSL for `ex4`: returns one extrapolated value from
/// 4 consecutive evaluations and `6*f_inf`.
fn emit_ex4(s: &mut String, result: &str, f0: &str, f1: &str, f2: &str, f3: &str, finf6: &str) {
    // result = 2*(2*(finf6 + f3 - f2 + f1) - f2) - f0
    let _ = writeln!(
        s,
        "        Fr {result}_t = fr_add(fr_add({finf6}, fr_sub({f3}, {f2})), {f1});"
    );
    let _ = writeln!(s, "        {result}_t = fr_add({result}_t, {result}_t);");
    let _ = writeln!(s, "        {result}_t = fr_sub({result}_t, {f2});");
    let _ = writeln!(s, "        {result}_t = fr_add({result}_t, {result}_t);");
    let _ = writeln!(s, "        Fr {result} = fr_sub({result}_t, {f0});");
}

/// Generate Toom-Cook D=4 MSL eval body for P product groups.
///
/// Algorithm (per product group of 4 inputs):
/// 1. Split into (input0, input1) and (input2, input3)
/// 2. Evaluate each pair-product at {1, 2, ∞} — 3 fr_mul each
/// 3. Extrapolate both to point 3 via ex2 — 0 fr_mul
/// 4. Point-wise multiply — 4 fr_mul
///
/// Total: 10 fr_mul per group (vs 12 naive).
fn generate_toom_cook_d4(p: usize) -> String {
    let mut s = String::with_capacity(4096);

    if p > 1 {
        for d in 0..4 {
            let _ = writeln!(s, "        evals[{d}] = fr_zero();");
        }
    }

    for g in 0..p {
        let base = g * 4;
        let _ = writeln!(s, "        {{ // Product group {g}");

        // Half A: sub-product of inputs [base+0, base+1]
        emit_eval_linear_prod_2(
            &mut s,
            "a",
            &format!("lo[{base}]"),
            &format!("hi[{base}]"),
            &format!("lo[{}]", base + 1),
            &format!("hi[{}]", base + 1),
        );
        emit_ex2(&mut s, "a_3", "a_1", "a_2", "a_inf");

        // Half B: sub-product of inputs [base+2, base+3]
        emit_eval_linear_prod_2(
            &mut s,
            "b",
            &format!("lo[{}]", base + 2),
            &format!("hi[{}]", base + 2),
            &format!("lo[{}]", base + 3),
            &format!("hi[{}]", base + 3),
        );
        emit_ex2(&mut s, "b_3", "b_1", "b_2", "b_inf");

        // Point-wise multiply
        if p == 1 {
            let _ = writeln!(s, "        evals[0] = fr_mul(a_1, b_1);");
            let _ = writeln!(s, "        evals[1] = fr_mul(a_2, b_2);");
            let _ = writeln!(s, "        evals[2] = fr_mul(a_3, b_3);");
            let _ = writeln!(s, "        evals[3] = fr_mul(a_inf, b_inf);");
        } else {
            let _ = writeln!(s, "        evals[0] = fr_add(evals[0], fr_mul(a_1, b_1));");
            let _ = writeln!(s, "        evals[1] = fr_add(evals[1], fr_mul(a_2, b_2));");
            let _ = writeln!(s, "        evals[2] = fr_add(evals[2], fr_mul(a_3, b_3));");
            let _ = writeln!(
                s,
                "        evals[3] = fr_add(evals[3], fr_mul(a_inf, b_inf));"
            );
        }
        let _ = writeln!(s, "        }}");
    }

    s
}

/// Generate Toom-Cook D=8 MSL eval body for P product groups.
///
/// Algorithm (per product group of 8 inputs):
/// 1. `eval_linear_prod_4_internal(inputs[0..4])` → (a1..a4, a_inf) — 11 fr_mul
/// 2. Extrapolate a to {5, 6, 7} via ex4_2 + ex4 — 0 fr_mul
/// 3. Same for inputs[4..8] → (b1..b7, b_inf) — 11 fr_mul
/// 4. Point-wise multiply — 8 fr_mul
///
/// Total: 30 fr_mul per group (vs 56 naive).
fn generate_toom_cook_d8(p: usize) -> String {
    let mut s = String::with_capacity(16384);

    if p > 1 {
        for d in 0..8 {
            let _ = writeln!(s, "        evals[{d}] = fr_zero();");
        }
    }

    for g in 0..p {
        let base = g * 8;
        let _ = writeln!(s, "        {{ // Product group {g}");

        // ---- Half A: eval_linear_prod_4_internal(inputs[base..base+4]) ----
        // Sub-pair (input0, input1) → (ar1, ar2, ar_inf)
        emit_eval_linear_prod_2(
            &mut s,
            "ar",
            &format!("lo[{base}]"),
            &format!("hi[{base}]"),
            &format!("lo[{}]", base + 1),
            &format!("hi[{}]", base + 1),
        );
        // Extrapolate: ar3 = ex2([ar1, ar2], ar_inf), ar4 = ex2([ar2, ar3], ar_inf)
        emit_ex2(&mut s, "ar_3", "ar_1", "ar_2", "ar_inf");
        emit_ex2(&mut s, "ar_4", "ar_2", "ar_3", "ar_inf");

        // Sub-pair (input2, input3) → (br1, br2, br_inf)
        emit_eval_linear_prod_2(
            &mut s,
            "br",
            &format!("lo[{}]", base + 2),
            &format!("hi[{}]", base + 2),
            &format!("lo[{}]", base + 3),
            &format!("hi[{}]", base + 3),
        );
        emit_ex2(&mut s, "br_3", "br_1", "br_2", "br_inf");
        emit_ex2(&mut s, "br_4", "br_2", "br_3", "br_inf");

        // Point-wise: a_k = ar_k * br_k for k in {1,2,3,4,inf}
        let _ = writeln!(s, "        Fr a_1 = fr_mul(ar_1, br_1);");
        let _ = writeln!(s, "        Fr a_2 = fr_mul(ar_2, br_2);");
        let _ = writeln!(s, "        Fr a_3 = fr_mul(ar_3, br_3);");
        let _ = writeln!(s, "        Fr a_4 = fr_mul(ar_4, br_4);");
        let _ = writeln!(s, "        Fr a_inf = fr_mul(ar_inf, br_inf);");

        // Extrapolate a to {5, 6, 7}: a_inf6 = 6*a_inf via adds
        let _ = writeln!(s, "        Fr a_inf2 = fr_add(a_inf, a_inf);");
        let _ = writeln!(s, "        Fr a_inf3 = fr_add(a_inf2, a_inf);");
        let _ = writeln!(s, "        Fr a_inf6 = fr_add(a_inf3, a_inf3);");
        emit_ex4_2(&mut s, "a_5", "a_6", ["a_1", "a_2", "a_3", "a_4"], "a_inf6");
        emit_ex4(&mut s, "a_7", "a_3", "a_4", "a_5", "a_6", "a_inf6");

        // ---- Half B: eval_linear_prod_4_internal(inputs[base+4..base+8]) ----
        emit_eval_linear_prod_2(
            &mut s,
            "cr",
            &format!("lo[{}]", base + 4),
            &format!("hi[{}]", base + 4),
            &format!("lo[{}]", base + 5),
            &format!("hi[{}]", base + 5),
        );
        emit_ex2(&mut s, "cr_3", "cr_1", "cr_2", "cr_inf");
        emit_ex2(&mut s, "cr_4", "cr_2", "cr_3", "cr_inf");

        emit_eval_linear_prod_2(
            &mut s,
            "dr",
            &format!("lo[{}]", base + 6),
            &format!("hi[{}]", base + 6),
            &format!("lo[{}]", base + 7),
            &format!("hi[{}]", base + 7),
        );
        emit_ex2(&mut s, "dr_3", "dr_1", "dr_2", "dr_inf");
        emit_ex2(&mut s, "dr_4", "dr_2", "dr_3", "dr_inf");

        // Point-wise: b_k = cr_k * dr_k for k in {1,2,3,4,inf}
        let _ = writeln!(s, "        Fr b_1 = fr_mul(cr_1, dr_1);");
        let _ = writeln!(s, "        Fr b_2 = fr_mul(cr_2, dr_2);");
        let _ = writeln!(s, "        Fr b_3 = fr_mul(cr_3, dr_3);");
        let _ = writeln!(s, "        Fr b_4 = fr_mul(cr_4, dr_4);");
        let _ = writeln!(s, "        Fr b_inf = fr_mul(cr_inf, dr_inf);");

        // Extrapolate b to {5, 6, 7}
        let _ = writeln!(s, "        Fr b_inf2 = fr_add(b_inf, b_inf);");
        let _ = writeln!(s, "        Fr b_inf3 = fr_add(b_inf2, b_inf);");
        let _ = writeln!(s, "        Fr b_inf6 = fr_add(b_inf3, b_inf3);");
        emit_ex4_2(&mut s, "b_5", "b_6", ["b_1", "b_2", "b_3", "b_4"], "b_inf6");
        emit_ex4(&mut s, "b_7", "b_3", "b_4", "b_5", "b_6", "b_inf6");

        // ---- Final point-wise multiply ----
        if p == 1 {
            let _ = writeln!(s, "        evals[0] = fr_mul(a_1, b_1);");
            let _ = writeln!(s, "        evals[1] = fr_mul(a_2, b_2);");
            let _ = writeln!(s, "        evals[2] = fr_mul(a_3, b_3);");
            let _ = writeln!(s, "        evals[3] = fr_mul(a_4, b_4);");
            let _ = writeln!(s, "        evals[4] = fr_mul(a_5, b_5);");
            let _ = writeln!(s, "        evals[5] = fr_mul(a_6, b_6);");
            let _ = writeln!(s, "        evals[6] = fr_mul(a_7, b_7);");
            let _ = writeln!(s, "        evals[7] = fr_mul(a_inf, b_inf);");
        } else {
            let _ = writeln!(s, "        evals[0] = fr_add(evals[0], fr_mul(a_1, b_1));");
            let _ = writeln!(s, "        evals[1] = fr_add(evals[1], fr_mul(a_2, b_2));");
            let _ = writeln!(s, "        evals[2] = fr_add(evals[2], fr_mul(a_3, b_3));");
            let _ = writeln!(s, "        evals[3] = fr_add(evals[3], fr_mul(a_4, b_4));");
            let _ = writeln!(s, "        evals[4] = fr_add(evals[4], fr_mul(a_5, b_5));");
            let _ = writeln!(s, "        evals[5] = fr_add(evals[5], fr_mul(a_6, b_6));");
            let _ = writeln!(s, "        evals[6] = fr_add(evals[6], fr_mul(a_7, b_7));");
            let _ = writeln!(
                s,
                "        evals[7] = fr_add(evals[7], fr_mul(a_inf, b_inf));"
            );
        }
        let _ = writeln!(s, "        }}");
    }

    s
}

/// Generate MSL evaluation body for ProductSum shape (single-pass, all evals at once).
///
/// Uses incremental interpolation: instead of `lo[k] + t * diff[k]` (which
/// requires a full `fr_mul` per input per grid point), we maintain `cur[k]`
/// starting at `hi[k]` and add `diff[k]` for each successive grid point.
/// This replaces `(D-2) * K` expensive `fr_mul` with cheap `fr_add`.
///
/// When `weight_folded`, the first factor of each product group is
/// pre-multiplied by `w`, so `evals[d]` are already weighted. This saves
/// `num_evals - 2*num_products` fr_mul per pair by folding the weight into
/// the product chain instead of multiplying each eval separately.
fn generate_product_sum_body(
    num_inputs_per_product: usize,
    num_products: usize,
    weight_folded: bool,
) -> String {
    let d = num_inputs_per_product;
    let p = num_products;
    let k = d * p;
    let mut s = String::with_capacity(2048);

    // Precompute differences. When weight-folded, the first index of each
    // product group (g*d) gets w baked in: diff[g*d] = w * (hi[g*d] - lo[g*d]).
    let _ = writeln!(s, "        Fr diff[{k}];");
    for i in 0..k {
        if weight_folded && i % d == 0 {
            let _ = writeln!(
                s,
                "        diff[{i}] = fr_mul(w, fr_sub(hi[{i}], lo[{i}]));"
            );
        } else {
            let _ = writeln!(s, "        diff[{i}] = fr_sub(hi[{i}], lo[{i}]);");
        }
    }

    // Interpolated values, starting at hi (t=1 point).
    // When weight-folded: cur[g*d] = w * hi[g*d].
    let _ = writeln!(s, "        Fr cur[{k}];");
    for i in 0..k {
        if weight_folded && i % d == 0 {
            let _ = writeln!(s, "        cur[{i}] = fr_mul(w, hi[{i}]);");
        } else {
            let _ = writeln!(s, "        cur[{i}] = hi[{i}];");
        }
    }
    s.push('\n');

    // t=1: product of cur values (with weight baked into first factor if folded)
    emit_product_sum(&mut s, d, p, "cur", 0);

    // t=2, ..., D-1: increment cur by diff, then compute product.
    // The incremental update w*cur[g*d] += w*diff[g*d] correctly maintains
    // the weighted first factor across grid points.
    for t in 2..d {
        let eval_idx = t - 1;
        for i in 0..k {
            let _ = writeln!(s, "        cur[{i}] = fr_add(cur[{i}], diff[{i}]);");
        }
        emit_product_sum(&mut s, d, p, "cur", eval_idx);
    }

    // t=inf: product of diffs (leading coefficient).
    // diff[g*d] already contains w*diff if weight-folded.
    emit_product_sum(&mut s, d, p, "diff", d - 1);

    s
}

/// Emit `evals[idx] = Σ_g Π_j arr[g*D+j]` where `arr` is "cur", "hi", or "diff".
///
/// Uses `fr_mul_unreduced` for intermediate products in the chain (skips the
/// conditional modulus subtraction after each CIOS). The unreduced result
/// (in [0, 2r)) is safe as CIOS input since BN254's 4r²/R < 2r. An explicit
/// `fr_reduce` before `fr_add` brings the final product back to [0, r).
/// Saves `(D-2)` `fr_reduce` calls per product group per grid point.
fn emit_product_sum(s: &mut String, d: usize, p: usize, arr: &str, eval_idx: usize) {
    let _ = writeln!(s, "        {{ Fr sum = fr_zero();");
    for g in 0..p {
        let base = g * d;
        let _ = writeln!(s, "          {{ Fr prod = {arr}[{base}];");
        for j in 1..d {
            let _ = writeln!(
                s,
                "            prod = fr_mul_unreduced(prod, {arr}[{}]);",
                base + j
            );
        }
        let _ = writeln!(s, "            sum = fr_add(sum, fr_reduce(prod)); }}");
    }
    let _ = writeln!(s, "          evals[{eval_idx}] = sum; }}");
}

/// Generate MSL evaluation body for Custom expression shape.
///
/// Uses incremental interpolation: maintains `cur_k` starting at `lo[k]` and
/// adding `diff_k` for each successive grid point. Grid is `{0, 2, 3, ..., degree}`
/// (skipping `t=1`), so slot 0 maps to `t=0` and slot `k>=1` maps to `t=k+1`.
///
/// Challenge values are read from a `device const Fr* challenges` buffer at
/// runtime rather than baked as MSL constants. This makes the kernel shape
/// deterministic at compile time, enabling AOT pipeline caching.
fn generate_custom_body(expr: &Expr, num_inputs: usize, degree: usize) -> String {
    // Grid: {0, 2, 3, ..., degree} — `degree` evaluations, skipping t=1
    let num_evals = degree;
    let mut s = String::with_capacity(2048);

    // Precompute diffs
    for k in 0..num_inputs {
        let _ = writeln!(s, "        Fr diff_{k} = fr_sub(hi[{k}], lo[{k}]);");
    }

    // Running interpolated values: start at lo, add diff each step
    for k in 0..num_inputs {
        let _ = writeln!(s, "        Fr cur_{k} = lo[{k}];");
    }
    s.push('\n');

    // Build the grid: slot 0 → t=0, slot k≥1 → t=k+1
    let grid: Vec<usize> = (0..num_evals)
        .map(|slot| if slot == 0 { 0 } else { slot + 1 })
        .collect();

    let mut prev_t = 0usize;
    for (slot, &t) in grid.iter().enumerate() {
        // Advance cur_k by diff_k for each step from prev_t to t
        let steps = t - prev_t;
        for _ in 0..steps {
            for k in 0..num_inputs {
                let _ = writeln!(s, "        cur_{k} = fr_add(cur_{k}, diff_{k});");
            }
        }
        prev_t = t;

        let _ = writeln!(s, "        {{ // t = {t}");

        // Walk expression tree — challenges read from buffer at runtime
        let mut visitor = MslCodeGen {
            code: String::new(),
            next_id: 0,
        };
        let root_name = expr.visit(&mut visitor);
        let _ = write!(s, "{}", visitor.code);
        let _ = writeln!(s, "            evals[{slot}] = {root_name};");
        let _ = writeln!(s, "        }}");
    }

    s
}

/// ExprVisitor that emits MSL assignments in SSA form.
///
/// Challenge variables emit `challenges[id]` (buffer reads), not baked constants.
/// This makes generated MSL deterministic per kernel shape, enabling AOT compilation.
struct MslCodeGen {
    code: String,
    next_id: usize,
}

impl MslCodeGen {
    fn fresh_var(&mut self) -> String {
        let name = format!("e{}", self.next_id);
        self.next_id += 1;
        name
    }
}

impl ExprVisitor for MslCodeGen {
    type Output = String;

    fn visit_constant(&mut self, val: i128) -> String {
        let name = self.fresh_var();
        // SoP coefficients in Jolt are always in {-1, 0, 1}. We fast-path
        // those and fall back to fr_from_u64 for small positive/negative values.
        match val {
            0 => {
                let _ = writeln!(self.code, "            Fr {name} = fr_zero();");
            }
            1 => {
                let _ = writeln!(self.code, "            Fr {name} = fr_one();");
            }
            -1 => {
                let _ = writeln!(self.code, "            Fr {name} = fr_neg(fr_one());");
            }
            v if v > 0 => {
                assert!(
                    v <= i128::from(u64::MAX),
                    "MSL constant {v} exceeds u64 range"
                );
                let _ = writeln!(
                    self.code,
                    "            Fr {name} = fr_from_u64((ulong){v});"
                );
            }
            v => {
                let abs = -v;
                assert!(
                    abs <= i128::from(u64::MAX),
                    "MSL constant {v} exceeds u64 range"
                );
                let _ = writeln!(
                    self.code,
                    "            Fr {name} = fr_neg(fr_from_u64((ulong){abs}));"
                );
            }
        }
        name
    }

    fn visit_var(&mut self, var: Var) -> String {
        match var {
            Var::Opening(id) => format!("cur_{id}"),
            Var::Challenge(id) => format!("challenges[{id}]"),
        }
    }

    fn visit_neg(&mut self, inner: String) -> String {
        let name = self.fresh_var();
        let _ = writeln!(self.code, "            Fr {name} = fr_neg({inner});");
        name
    }

    fn visit_add(&mut self, lhs: String, rhs: String) -> String {
        let name = self.fresh_var();
        let _ = writeln!(self.code, "            Fr {name} = fr_add({lhs}, {rhs});");
        name
    }

    fn visit_sub(&mut self, lhs: String, rhs: String) -> String {
        let name = self.fresh_var();
        let _ = writeln!(self.code, "            Fr {name} = fr_sub({lhs}, {rhs});");
        name
    }

    fn visit_mul(&mut self, lhs: String, rhs: String) -> String {
        let name = self.fresh_var();
        let _ = writeln!(self.code, "            Fr {name} = fr_mul({lhs}, {rhs});");
        name
    }
}
