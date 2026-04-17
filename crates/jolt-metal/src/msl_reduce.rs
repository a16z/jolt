//! MSL reduce kernel codegen: `Formula` → MSL source → `CompiledPipeline`.
//!
//! Two compilation paths:
//!
//! - **ProductSum**: Generates a fully-unrolled MSL kernel evaluating
//!   `Σ_g Π_j p_{g,j}(t)` on the Toom-Cook grid `{1, 2, ..., D-1, ∞}`.
//!   Uses incremental interpolation (`cur += diff` per grid step) to avoid
//!   expensive `fr_mul` for small integer grid scalars.
//!
//! - **Custom**: Walks the `Formula` terms, emitting
//!   MSL assignments in SSA form. Challenge values are read from a runtime
//!   `device const Fr* challenges` buffer — not baked as MSL constants. This
//!   makes pipeline compilation deterministic per kernel shape, enabling AOT
//!   caching.
//!   Uses incremental interpolation for the contiguous grid `{0, 1, 2, ..., degree}`.
//!
//! Each `generate_msl` call emits exactly one kernel variant (Tensor,
//! LowToHigh, or HighToLow), determined by `KernelVariant::from_spec`.
//!
//! Single-pass reduce kernels accumulate directly into `Fr` values via
//! `fr_add(fr_acc, fr_reduce(eval))` (unweighted) or
//! `fr_add(fr_acc, fr_mul(w, eval))` (weighted). This uses N limbs per
//! accumulator instead of 2N+2 (`WideAcc`), cutting register pressure
//! roughly in half and improving GPU occupancy at high D.

use std::sync::Arc;

use jolt_compiler::kernel_spec::Iteration;
use jolt_compiler::{Factor, Formula, ProductTerm};
use jolt_compute::BindingOrder;
use metal::CompileOptions;

use crate::field_params::MslFieldParams;
use crate::kernel::CompiledPipeline;
use crate::msl_writer::{msl, Msl};
use crate::pipeline::{build_source_with_preamble, make_pipeline};

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

/// Which pair-reading and weight strategy a kernel uses.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelVariant {
    /// Interleaved pairs: `lo = buf[2i], hi = buf[2i+1]`, no weight buffer.
    LowToHigh,
    /// Split-half pairs: `lo = buf[i], hi = buf[i + n_pairs]`, no weight buffer.
    HighToLow,
    /// Interleaved pairs with tensor weight: `w = outer[i>>log] * inner[i&mask]`.
    Tensor,
    /// Sparse index-based pairs: `lo = buf[pair_index[2i]]`, `hi = buf[pair_index[2i+1]]`.
    /// Missing entries (sentinel `0xFFFFFFFF`) default to zero.
    Sparse,
}

impl KernelVariant {
    pub fn from_spec(iteration: &Iteration, binding_order: BindingOrder) -> Self {
        match iteration {
            Iteration::DenseTensor => Self::Tensor,
            Iteration::Sparse => Self::Sparse,
            Iteration::Dense => match binding_order {
                BindingOrder::LowToHigh => Self::LowToHigh,
                BindingOrder::HighToLow => Self::HighToLow,
            },
            Iteration::Domain { .. } => {
                panic!("Domain iteration not yet supported on Metal — use CpuBackend")
            }
            Iteration::Gruen => {
                panic!("Gruen iteration not yet supported on Metal — use CpuBackend")
            }
        }
    }

    const FUNCTION_NAME: &'static str = "reduce_kernel";
}

/// Output of MSL generation — the full source string plus kernel metadata.
pub struct GeneratedMsl {
    pub source: String,
    pub num_inputs: usize,
    pub num_evals: usize,
    /// Whether the kernel signature includes a `challenges` buffer parameter.
    pub has_challenges: bool,
}

struct EvalBody {
    msl: String,
    reads_inline: bool,
    fuse_accumulate: bool,
    weight_folded: bool,
    has_challenges: bool,
}

/// Generate the MSL source for a single kernel variant of a composition formula.
///
/// Only emits the variant needed for the given `iteration` + `binding_order`,
/// cutting shader compilation time to ~1/3.
pub fn generate_msl(
    formula: &Formula,
    variant: KernelVariant,
    mode: CompileMode,
    field_config: &MslFieldParams,
    device_config: &crate::config::MetalDeviceConfig,
) -> GeneratedMsl {
    let num_inputs = formula.num_inputs;
    let degree = formula.degree();
    let is_weighted = matches!(variant, KernelVariant::Tensor);
    let is_h2l = matches!(variant, KernelVariant::HighToLow);
    let n_limbs = field_config.n_limbs;

    let is_sparse = matches!(variant, KernelVariant::Sparse);
    let num_evals = degree;

    // The D=8 deferred path reads pairs inline with hardcoded positional
    // indexing — incompatible with sparse index-based reads.
    let body = if let Some((d, p)) = formula.as_product_sum() {
        if d == 8 && p == 1 && !is_sparse {
            generate_toom_cook_d8_deferred(p, is_weighted, is_h2l)
        } else if d == 4 || d == 8 {
            if is_weighted {
                generate_toom_cook_body_weighted(d, p)
            } else {
                generate_toom_cook_body(d, p)
            }
        } else {
            let fold = is_weighted && d > 2 * p;
            generate_product_sum_body(d, p, fold)
        }
    } else {
        generate_formula_body(formula, degree)
    };

    let kernel = generate_reduce_kernel(
        num_inputs,
        num_evals,
        &body,
        variant,
        n_limbs,
        device_config,
    );

    let noinline = matches!(mode, CompileMode::FastCompile);
    let source = build_source_with_preamble(&field_config.msl_preamble, &[&kernel], noinline);

    GeneratedMsl {
        source,
        num_inputs,
        num_evals,
        has_challenges: body.has_challenges,
    }
}

/// Compile a generated MSL source into a single pipeline state.
pub(crate) fn compile_msl(device: &metal::Device, msl: &GeneratedMsl) -> Arc<CompiledPipeline> {
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(&msl.source, &options)
        .unwrap_or_else(|e| panic!("reduce kernel MSL compilation failed: {e}"));

    Arc::new(CompiledPipeline {
        pipeline: make_pipeline(device, &library, KernelVariant::FUNCTION_NAME),
        num_evals: msl.num_evals,
        num_inputs: msl.num_inputs,
        has_challenges: msl.has_challenges,
    })
}

/// Generate a reduce kernel for a specific pair-reading variant.
///
/// All variants share the same eval body, accumulation, simdgroup reduction,
/// and shared-memory tree. They differ only in:
/// - Kernel function name and buffer signature
/// - Pair reading pattern (interleaved vs split-half)
/// - Weight computation (single buffer, tensor product, or none)
/// - Accumulation strategy (Fr direct vs WideAcc fmadd)
///
/// For unweighted and weight-folded kernels, accumulation uses Fr accumulators
/// instead of WideAcc. This reduces register pressure by `num_evals × (ACC_LIMBS - N)`
/// u32 registers per thread (80 for BN254 D=8), improving GPU occupancy.
/// The simdgroup shuffle also shrinks from ACC_LIMBS to N limbs per eval.
///
/// When `fuse_accumulate` is true, the eval body has already fused the
/// final multiply + reduce + accumulate into `acc_d` directly. The evals
/// array declaration and accumulation loop are skipped, saving `num_evals × N`
/// registers (64 for BN254 D=8).
fn generate_reduce_kernel(
    num_inputs: usize,
    num_evals: usize,
    body: &EvalBody,
    variant: KernelVariant,
    n_limbs: usize,
    device_config: &crate::config::MetalDeviceConfig,
) -> String {
    let gs = device_config.reduce_group_size;
    let num_simdgroups = device_config.num_simdgroups();
    let fname = KernelVariant::FUNCTION_NAME;
    let is_tensor = matches!(variant, KernelVariant::Tensor);
    let weighted = matches!(variant, KernelVariant::Tensor);

    // All single-pass kernels use Fr accumulators (no WideAcc).
    // This saves ACC_LIMBS - N registers per eval per thread.
    // For weighted + not folded: accumulation uses fr_mul(w, eval) + fr_add.
    // For unweighted / weight-folded: accumulation uses fr_reduce(eval) + fr_add.
    let needs_weight_mul = weighted && !body.weight_folded;

    let mut w = Msl::new(8192);

    // Kernel signature
    msl!(w, "kernel void {fname}(");
    w.push();
    for k in 0..num_inputs {
        msl!(w, "device const Fr* input_{k} [[buffer({k})]],");
    }

    let is_sparse = matches!(variant, KernelVariant::Sparse);
    let mut next_buf = num_inputs;
    if is_sparse {
        msl!(w, "device const uint* pair_index [[buffer({next_buf})]],");
        next_buf += 1;
    } else if weighted {
        if is_tensor {
            msl!(w, "device const Fr* outer_weights [[buffer({next_buf})]],");
            next_buf += 1;
            msl!(w, "device const Fr* inner_weights [[buffer({next_buf})]],");
            next_buf += 1;
        } else {
            msl!(w, "device const Fr* weights [[buffer({next_buf})]],");
            next_buf += 1;
        }
    }
    if body.has_challenges {
        msl!(w, "device const Fr* challenges [[buffer({next_buf})]],");
        next_buf += 1;
    }
    msl!(w, "device Fr* partials [[buffer({next_buf})]],");
    next_buf += 1;
    msl!(w, "device const uint* params [[buffer({next_buf})]],");
    w.line("uint gid [[threadgroup_position_in_grid]],");
    w.line("uint lid [[thread_position_in_threadgroup]],");
    w.line("uint num_groups [[threadgroups_per_grid]],");
    w.line("uint simd_lane [[thread_index_in_simdgroup]],");
    w.line("uint simd_id [[simdgroup_index_in_threadgroup]]");
    w.pop();
    w.line(") {");

    w.push();

    // Read params
    w.line("uint n_pairs = params[0];");
    if is_tensor {
        w.line("uint inner_log = params[1];");
        w.line("uint inner_mask = params[2];");
    }
    w.blank();

    // Individual Fr accumulators (not arrays) for independent register scheduling.
    // N limbs per eval vs ACC_LIMBS with WideAcc — saves 80 registers for BN254 D=8.
    for d in 0..num_evals {
        msl!(w, "Fr acc_{d} = fr_zero();");
    }
    w.blank();

    // Grid-stride loop
    msl!(
        w,
        "for (uint i = gid * {gs}u + lid; i < n_pairs; i += num_groups * {gs}u) {{"
    );
    w.push();

    // Read pairs into individual scalars (not arrays) so the Metal
    // compiler can track per-element register lifetimes independently.
    // When reads_inline, the body handles its own reads with deferred timing.
    if !body.reads_inline {
        if is_sparse {
            // Read pair indices, then load values with sentinel check.
            w.line("uint _lo_i = pair_index[2u * i];");
            w.line("uint _hi_i = pair_index[2u * i + 1u];");
            for k in 0..num_inputs {
                msl!(
                    w,
                    "Fr lo_{k} = (_lo_i != 0xFFFFFFFFu) ? input_{k}[_lo_i] : fr_zero();"
                );
                msl!(
                    w,
                    "Fr hi_{k} = (_hi_i != 0xFFFFFFFFu) ? input_{k}[_hi_i] : fr_zero();"
                );
            }
        } else if matches!(variant, KernelVariant::HighToLow) {
            for k in 0..num_inputs {
                msl!(
                    w,
                    "Fr lo_{k} = input_{k}[i], hi_{k} = input_{k}[i + n_pairs];"
                );
            }
        } else {
            for k in 0..num_inputs {
                msl!(
                    w,
                    "Fr lo_{k} = input_{k}[2u * i], hi_{k} = input_{k}[2u * i + 1u];"
                );
            }
        }
        w.blank();
    }

    // Read weight (weighted variants only) — before eval body so
    // weight-folded bodies can reference `w`.
    if weighted {
        if is_tensor {
            w.line("Fr w = fr_mul(outer_weights[i >> inner_log], inner_weights[i & inner_mask]);");
        } else {
            w.line("Fr w = weights[i];");
        }
        w.blank();
    }

    if body.fuse_accumulate {
        // Body directly accumulates into acc_d — no evals array needed.
        w.raw(&body.msl);
    } else {
        // Scalar decomposition: replace `evals[d]` array with individual
        // `eval_d` scalars so the Metal compiler tracks per-element
        // register lifetimes independently (same technique as lo/hi).
        let mut decomposed_body = body.msl.clone();
        for d in 0..num_evals {
            decomposed_body = decomposed_body.replace(&format!("evals[{d}]"), &format!("eval_{d}"));
        }

        for d in 0..num_evals {
            msl!(w, "Fr eval_{d};");
        }
        w.raw(&decomposed_body);
        w.blank();

        if needs_weight_mul {
            for d in 0..num_evals {
                msl!(w, "acc_{d} = fr_add(acc_{d}, fr_mul(w, eval_{d}));");
            }
        } else {
            for d in 0..num_evals {
                msl!(w, "acc_{d} = fr_add(acc_{d}, fr_reduce(eval_{d}));");
            }
        }
    }
    w.pop();
    w.line("}");
    w.blank();

    // Simdgroup reduction: shuffle N limbs per eval, merge with fr_add.
    // For BN254 D=8: 8×8×5 = 320 shuffles (vs 8×18×5 = 720 with old WideAcc).
    let half_simd = device_config.simd_size / 2;
    msl!(
        w,
        "for (ushort _off = {half_simd}u; _off > 0u; _off >>= 1u) {{"
    );
    w.push();
    for d in 0..num_evals {
        w.line("{ Fr _o;");
        for l in 0..n_limbs {
            msl!(
                w,
                "_o.limbs[{l}] = simd_shuffle_down(acc_{d}.limbs[{l}], _off);"
            );
        }
        msl!(w, "acc_{d} = fr_add(acc_{d}, _o); }}");
    }
    w.pop();
    w.line("}");
    w.blank();

    // Lane 0 of each simdgroup writes Fr directly to shared memory.
    let sh_size = num_evals * num_simdgroups;
    msl!(w, "threadgroup Fr sh[{sh_size}];");
    w.line("if (simd_lane == 0u) {");
    w.push();
    for d in 0..num_evals {
        msl!(w, "sh[{}u + simd_id] = acc_{d};", d * num_simdgroups);
    }
    w.pop();
    w.line("}");
    w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    w.blank();

    // Tree reduction over simdgroup partials (8 → 1).
    // All threads must reach each threadgroup_barrier — never put barriers
    // inside divergent branches. Only the active lanes do the fr_add.
    let mut stride = num_simdgroups / 2;
    while stride > 0 {
        msl!(w, "if (lid < {stride}u) {{");
        w.push();
        for d in 0..num_evals {
            let base = d * num_simdgroups;
            msl!(
                w,
                "sh[{base}u + lid] = fr_add(sh[{base}u + lid], sh[{base}u + lid + {stride}u]);"
            );
        }
        w.pop();
        w.line("}");
        if stride > 1 {
            w.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        }
        stride /= 2;
    }
    w.blank();

    // Write final partials
    w.line("if (lid == 0u) {");
    w.push();
    for d in 0..num_evals {
        msl!(
            w,
            "partials[gid * {num_evals}u + {d}u] = sh[{}u];",
            d * num_simdgroups
        );
    }
    w.pop();
    w.line("}");

    w.pop();
    w.line("}");

    w.into_string()
}

/// Generate Toom-Cook MSL evaluation body for ProductSum D=4 or D=8.
///
/// Uses balanced binary splitting to reduce field multiplications:
/// - D=4: 2x2 split → 10 fr_mul (vs 12 naive)
/// - D=8: 4x4 split → 30 fr_mul (vs 56 naive)
///
/// Extrapolation uses only fr_add/fr_sub (no mul_u64 needed on GPU).
fn generate_toom_cook_body(d: usize, p: usize) -> EvalBody {
    EvalBody {
        msl: generate_toom_cook_body_msl(d, p),
        reads_inline: false,
        fuse_accumulate: false,
        weight_folded: false,
        has_challenges: false,
    }
}

fn generate_toom_cook_body_msl(d: usize, p: usize) -> String {
    match d {
        4 => generate_toom_cook_d4(p),
        8 => generate_toom_cook_d8(p),
        _ => unreachable!("Toom-Cook only implemented for D=4 and D=8"),
    }
}

/// Weight-folded Toom-Cook body: bake `w` into the first factor of the first
/// product group (lo_0/hi_0). All subsequent Toom-Cook evaluations carry the
/// weight through the tree. Costs 2 `fr_mul` (for `w*lo_0`, `w*hi_0`)
/// instead of D `fr_mul(w, eval)` in the accumulation — saves D-2 multiplies.
fn generate_toom_cook_body_weighted(d: usize, p: usize) -> EvalBody {
    let mut w = Msl::new_at(16384, 2);
    // Fold weight into first factor of first product group
    w.line("lo_0 = fr_mul(w, lo_0);");
    w.line("hi_0 = fr_mul(w, hi_0);");
    let mut msl_str = w.into_string();
    msl_str.push_str(&generate_toom_cook_body_msl(d, p));
    EvalBody {
        msl: msl_str,
        reads_inline: false,
        fuse_accumulate: false,
        weight_folded: true,
        has_challenges: false,
    }
}

/// Toom-Cook D=8 P=1 with deferred pair reads and fused accumulation.
///
/// Reads lo/hi for inputs 0-3 (half A), computes half A's degree-4
/// sub-product evaluations at {1..7, inf}, then reads lo/hi for
/// inputs 4-7 (half B). By the time half B is read, lo_0..lo_3
/// and hi_0..hi_3 are dead — saving 64 registers at peak liveness.
///
/// The final point-wise multiplies accumulate directly into `acc_d`
/// (fused eval-accumulate), eliminating the `evals[8]` intermediate
/// array and saving 64 registers (8 Fr × 8 limbs) at peak.
///
/// `weighted`: if true, prepends weight fold (`lo_0 = fr_mul(w, lo_0)`)
/// `h2l`: if true, uses H2L indexing (`input_k[i]`/`input_k[i + n_pairs]`)
fn generate_toom_cook_d8_deferred(p: usize, weighted: bool, h2l: bool) -> EvalBody {
    assert_eq!(p, 1, "deferred reads only implemented for P=1");
    let mut w = Msl::new_at(16384, 2);

    // Read half A inputs (0-3)
    for k in 0..4 {
        if h2l {
            msl!(
                w,
                "Fr lo_{k} = input_{k}[i], hi_{k} = input_{k}[i + n_pairs];"
            );
        } else {
            msl!(
                w,
                "Fr lo_{k} = input_{k}[2u * i], hi_{k} = input_{k}[2u * i + 1u];"
            );
        }
    }

    // Weight fold into first factor
    if weighted {
        w.line("lo_0 = fr_mul(w, lo_0);");
        w.line("hi_0 = fr_mul(w, hi_0);");
    }

    // Half A: inputs 0-3
    // Sub-pair (input0, input1) → (ar1, ar2, ar_inf)
    emit_eval_linear_prod_2(&mut w, "ar", "lo_0", "hi_0", "lo_1", "hi_1");
    emit_ex2(&mut w, "ar_3", "ar_1", "ar_2", "ar_inf");
    emit_ex2(&mut w, "ar_4", "ar_2", "ar_3", "ar_inf");

    // Sub-pair (input2, input3) → (br1, br2, br_inf)
    emit_eval_linear_prod_2(&mut w, "br", "lo_2", "hi_2", "lo_3", "hi_3");
    emit_ex2(&mut w, "br_3", "br_1", "br_2", "br_inf");
    emit_ex2(&mut w, "br_4", "br_2", "br_3", "br_inf");

    // Point-wise: a_k = ar_k * br_k for k in {1,2,3,4,inf}
    w.line("Fr a_1 = fr_mul(ar_1, br_1);");
    w.line("Fr a_2 = fr_mul(ar_2, br_2);");
    w.line("Fr a_3 = fr_mul(ar_3, br_3);");
    w.line("Fr a_4 = fr_mul(ar_4, br_4);");
    w.line("Fr a_inf = fr_mul(ar_inf, br_inf);");

    // Extrapolate a to {5, 6, 7}
    w.line("Fr a_inf2 = fr_add(a_inf, a_inf);");
    w.line("Fr a_inf3 = fr_add(a_inf2, a_inf);");
    w.line("Fr a_inf6 = fr_add(a_inf3, a_inf3);");
    emit_ex4_2(&mut w, "a_5", "a_6", ["a_1", "a_2", "a_3", "a_4"], "a_inf6");
    emit_ex4(&mut w, "a_7", "a_3", "a_4", "a_5", "a_6", "a_inf6");

    // Deferred read: half B inputs (4-7)
    // lo_0..lo_3, hi_0..hi_3 are dead — their registers are available.
    for k in 4..8 {
        if h2l {
            msl!(
                w,
                "Fr lo_{k} = input_{k}[i], hi_{k} = input_{k}[i + n_pairs];"
            );
        } else {
            msl!(
                w,
                "Fr lo_{k} = input_{k}[2u * i], hi_{k} = input_{k}[2u * i + 1u];"
            );
        }
    }

    // Half B: inputs 4-7
    emit_eval_linear_prod_2(&mut w, "cr", "lo_4", "hi_4", "lo_5", "hi_5");
    emit_ex2(&mut w, "cr_3", "cr_1", "cr_2", "cr_inf");
    emit_ex2(&mut w, "cr_4", "cr_2", "cr_3", "cr_inf");

    emit_eval_linear_prod_2(&mut w, "dr", "lo_6", "hi_6", "lo_7", "hi_7");
    emit_ex2(&mut w, "dr_3", "dr_1", "dr_2", "dr_inf");
    emit_ex2(&mut w, "dr_4", "dr_2", "dr_3", "dr_inf");

    // Point-wise: b_k = cr_k * dr_k for k in {1,2,3,4,inf}
    w.line("Fr b_1 = fr_mul(cr_1, dr_1);");
    w.line("Fr b_2 = fr_mul(cr_2, dr_2);");
    w.line("Fr b_3 = fr_mul(cr_3, dr_3);");
    w.line("Fr b_4 = fr_mul(cr_4, dr_4);");
    w.line("Fr b_inf = fr_mul(cr_inf, dr_inf);");

    // Extrapolate b to {5, 6, 7}
    w.line("Fr b_inf2 = fr_add(b_inf, b_inf);");
    w.line("Fr b_inf3 = fr_add(b_inf2, b_inf);");
    w.line("Fr b_inf6 = fr_add(b_inf3, b_inf3);");
    emit_ex4_2(&mut w, "b_5", "b_6", ["b_1", "b_2", "b_3", "b_4"], "b_inf6");
    emit_ex4(&mut w, "b_7", "b_3", "b_4", "b_5", "b_6", "b_inf6");

    // Fused eval-accumulate: point-wise multiply → fr_reduce → fr_add
    // directly into accumulators. No evals[] array needed — saves 64 registers.
    for (d, a_name, b_name) in [
        (0, "a_1", "b_1"),
        (1, "a_2", "b_2"),
        (2, "a_3", "b_3"),
        (3, "a_4", "b_4"),
        (4, "a_5", "b_5"),
        (5, "a_6", "b_6"),
        (6, "a_7", "b_7"),
        (7, "a_inf", "b_inf"),
    ] {
        msl!(
            w,
            "acc_{d} = fr_add(acc_{d}, fr_reduce(fr_mul_unreduced({a_name}, {b_name})));"
        );
    }

    EvalBody {
        msl: w.into_string(),
        reads_inline: true,
        fuse_accumulate: true,
        weight_folded: weighted,
        has_challenges: false,
    }
}

/// Emit MSL for the degree-2 sub-product of two linear polynomials.
///
/// Given inputs at indices `i0` and `i1` in the `lo`/`hi` arrays, produces
/// three sub-product evaluations `{prefix}_1`, `{prefix}_2`, `{prefix}_inf`
/// at points {1, 2, ∞}. Uses 3 fr_mul.
fn emit_eval_linear_prod_2(w: &mut Msl, prefix: &str, lo0: &str, hi0: &str, lo1: &str, hi1: &str) {
    msl!(w, "Fr {prefix}_s0 = fr_sub({hi0}, {lo0});");
    msl!(w, "Fr {prefix}_v2_0 = fr_add({prefix}_s0, {hi0});");
    msl!(w, "Fr {prefix}_s1 = fr_sub({hi1}, {lo1});");
    msl!(w, "Fr {prefix}_v2_1 = fr_add({prefix}_s1, {hi1});");
    msl!(w, "Fr {prefix}_1 = fr_mul({hi0}, {hi1});");
    msl!(w, "Fr {prefix}_2 = fr_mul({prefix}_v2_0, {prefix}_v2_1);");
    msl!(w, "Fr {prefix}_inf = fr_mul({prefix}_s0, {prefix}_s1);");
}

/// Emit MSL for `ex2` extrapolation: `result = 2*(f1 + find) - f0`.
fn emit_ex2(w: &mut Msl, result: &str, f0: &str, f1: &str, find: &str) {
    msl!(w, "Fr {result}_sum = fr_add({f1}, {find});");
    msl!(
        w,
        "Fr {result} = fr_sub(fr_add({result}_sum, {result}_sum), {f0});"
    );
}

/// Emit MSL for `ex4_2`: returns two extrapolated values (f4, f5) from
/// 4 consecutive evaluations `f[0..4]` and `6*f_inf`.
fn emit_ex4_2(w: &mut Msl, r4: &str, r5: &str, f: [&str; 4], find6: &str) {
    // f4 = 2*(2*(find6 + f[3] - f[2] + f[1]) - f[2]) - f[0]
    msl!(w, "Fr {r4}_f3m2 = fr_sub({}, {});", f[3], f[2]);
    msl!(
        w,
        "Fr {r4}_t = fr_add(fr_add({find6}, {r4}_f3m2), {});",
        f[1]
    );
    msl!(w, "{r4}_t = fr_add({r4}_t, {r4}_t);");
    msl!(w, "{r4}_t = fr_sub({r4}_t, {});", f[2]);
    msl!(w, "{r4}_t = fr_add({r4}_t, {r4}_t);");
    msl!(w, "Fr {r4} = fr_sub({r4}_t, {});", f[0]);

    // f5 = 2*(2*(f4 - f3m2 + find6) - f[3]) - f[1]
    msl!(w, "Fr {r5}_t = fr_add(fr_sub({r4}, {r4}_f3m2), {find6});");
    msl!(w, "{r5}_t = fr_add({r5}_t, {r5}_t);");
    msl!(w, "{r5}_t = fr_sub({r5}_t, {});", f[3]);
    msl!(w, "{r5}_t = fr_add({r5}_t, {r5}_t);");
    msl!(w, "Fr {r5} = fr_sub({r5}_t, {});", f[1]);
}

/// Emit MSL for `ex4`: returns one extrapolated value from
/// 4 consecutive evaluations and `6*f_inf`.
fn emit_ex4(w: &mut Msl, result: &str, f0: &str, f1: &str, f2: &str, f3: &str, find6: &str) {
    // result = 2*(2*(find6 + f3 - f2 + f1) - f2) - f0
    msl!(
        w,
        "Fr {result}_t = fr_add(fr_add({find6}, fr_sub({f3}, {f2})), {f1});"
    );
    msl!(w, "{result}_t = fr_add({result}_t, {result}_t);");
    msl!(w, "{result}_t = fr_sub({result}_t, {f2});");
    msl!(w, "{result}_t = fr_add({result}_t, {result}_t);");
    msl!(w, "Fr {result} = fr_sub({result}_t, {f0});");
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
    let mut w = Msl::new_at(4096, 2);

    if p > 1 {
        for d in 0..4 {
            msl!(w, "evals[{d}] = fr_zero();");
        }
    }

    for g in 0..p {
        let base = g * 4;
        msl!(w, "{{ // Product group {g}");

        // Half A: sub-product of inputs [base+0, base+1]
        emit_eval_linear_prod_2(
            &mut w,
            "a",
            &format!("lo_{base}"),
            &format!("hi_{base}"),
            &format!("lo_{}", base + 1),
            &format!("hi_{}", base + 1),
        );
        emit_ex2(&mut w, "a_3", "a_1", "a_2", "a_inf");

        // Half B: sub-product of inputs [base+2, base+3]
        emit_eval_linear_prod_2(
            &mut w,
            "b",
            &format!("lo_{}", base + 2),
            &format!("hi_{}", base + 2),
            &format!("lo_{}", base + 3),
            &format!("hi_{}", base + 3),
        );
        emit_ex2(&mut w, "b_3", "b_1", "b_2", "b_inf");

        // Point-wise multiply
        if p == 1 {
            w.line("evals[0] = fr_mul_unreduced(a_1, b_1);");
            w.line("evals[1] = fr_mul_unreduced(a_2, b_2);");
            w.line("evals[2] = fr_mul_unreduced(a_3, b_3);");
            w.line("evals[3] = fr_mul_unreduced(a_inf, b_inf);");
        } else {
            w.line("evals[0] = fr_add(evals[0], fr_mul(a_1, b_1));");
            w.line("evals[1] = fr_add(evals[1], fr_mul(a_2, b_2));");
            w.line("evals[2] = fr_add(evals[2], fr_mul(a_3, b_3));");
            w.line("evals[3] = fr_add(evals[3], fr_mul(a_inf, b_inf));");
        }
        w.line("}");
    }

    w.into_string()
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
    let mut w = Msl::new_at(16384, 2);

    if p > 1 {
        for d in 0..8 {
            msl!(w, "evals[{d}] = fr_zero();");
        }
    }

    for g in 0..p {
        let base = g * 8;
        msl!(w, "{{ // Product group {g}");

        // Half A: eval_linear_prod_4_internal(inputs[base..base+4])
        // Sub-pair (input0, input1) → (ar1, ar2, ar_inf)
        emit_eval_linear_prod_2(
            &mut w,
            "ar",
            &format!("lo_{base}"),
            &format!("hi_{base}"),
            &format!("lo_{}", base + 1),
            &format!("hi_{}", base + 1),
        );
        // Extrapolate: ar3 = ex2([ar1, ar2], ar_inf), ar4 = ex2([ar2, ar3], ar_inf)
        emit_ex2(&mut w, "ar_3", "ar_1", "ar_2", "ar_inf");
        emit_ex2(&mut w, "ar_4", "ar_2", "ar_3", "ar_inf");

        // Sub-pair (input2, input3) → (br1, br2, br_inf)
        emit_eval_linear_prod_2(
            &mut w,
            "br",
            &format!("lo_{}", base + 2),
            &format!("hi_{}", base + 2),
            &format!("lo_{}", base + 3),
            &format!("hi_{}", base + 3),
        );
        emit_ex2(&mut w, "br_3", "br_1", "br_2", "br_inf");
        emit_ex2(&mut w, "br_4", "br_2", "br_3", "br_inf");

        // Point-wise: a_k = ar_k * br_k for k in {1,2,3,4,inf}
        w.line("Fr a_1 = fr_mul(ar_1, br_1);");
        w.line("Fr a_2 = fr_mul(ar_2, br_2);");
        w.line("Fr a_3 = fr_mul(ar_3, br_3);");
        w.line("Fr a_4 = fr_mul(ar_4, br_4);");
        w.line("Fr a_inf = fr_mul(ar_inf, br_inf);");

        // Extrapolate a to {5, 6, 7}: a_inf6 = 6*a_inf via adds
        w.line("Fr a_inf2 = fr_add(a_inf, a_inf);");
        w.line("Fr a_inf3 = fr_add(a_inf2, a_inf);");
        w.line("Fr a_inf6 = fr_add(a_inf3, a_inf3);");
        emit_ex4_2(&mut w, "a_5", "a_6", ["a_1", "a_2", "a_3", "a_4"], "a_inf6");
        emit_ex4(&mut w, "a_7", "a_3", "a_4", "a_5", "a_6", "a_inf6");

        // Half B: eval_linear_prod_4_internal(inputs[base+4..base+8])
        emit_eval_linear_prod_2(
            &mut w,
            "cr",
            &format!("lo_{}", base + 4),
            &format!("hi_{}", base + 4),
            &format!("lo_{}", base + 5),
            &format!("hi_{}", base + 5),
        );
        emit_ex2(&mut w, "cr_3", "cr_1", "cr_2", "cr_inf");
        emit_ex2(&mut w, "cr_4", "cr_2", "cr_3", "cr_inf");

        emit_eval_linear_prod_2(
            &mut w,
            "dr",
            &format!("lo_{}", base + 6),
            &format!("hi_{}", base + 6),
            &format!("lo_{}", base + 7),
            &format!("hi_{}", base + 7),
        );
        emit_ex2(&mut w, "dr_3", "dr_1", "dr_2", "dr_inf");
        emit_ex2(&mut w, "dr_4", "dr_2", "dr_3", "dr_inf");

        // Point-wise: b_k = cr_k * dr_k for k in {1,2,3,4,inf}
        w.line("Fr b_1 = fr_mul(cr_1, dr_1);");
        w.line("Fr b_2 = fr_mul(cr_2, dr_2);");
        w.line("Fr b_3 = fr_mul(cr_3, dr_3);");
        w.line("Fr b_4 = fr_mul(cr_4, dr_4);");
        w.line("Fr b_inf = fr_mul(cr_inf, dr_inf);");

        // Extrapolate b to {5, 6, 7}
        w.line("Fr b_inf2 = fr_add(b_inf, b_inf);");
        w.line("Fr b_inf3 = fr_add(b_inf2, b_inf);");
        w.line("Fr b_inf6 = fr_add(b_inf3, b_inf3);");
        emit_ex4_2(&mut w, "b_5", "b_6", ["b_1", "b_2", "b_3", "b_4"], "b_inf6");
        emit_ex4(&mut w, "b_7", "b_3", "b_4", "b_5", "b_6", "b_inf6");

        // Final point-wise multiply
        if p == 1 {
            w.line("evals[0] = fr_mul_unreduced(a_1, b_1);");
            w.line("evals[1] = fr_mul_unreduced(a_2, b_2);");
            w.line("evals[2] = fr_mul_unreduced(a_3, b_3);");
            w.line("evals[3] = fr_mul_unreduced(a_4, b_4);");
            w.line("evals[4] = fr_mul_unreduced(a_5, b_5);");
            w.line("evals[5] = fr_mul_unreduced(a_6, b_6);");
            w.line("evals[6] = fr_mul_unreduced(a_7, b_7);");
            w.line("evals[7] = fr_mul_unreduced(a_inf, b_inf);");
        } else {
            w.line("evals[0] = fr_add(evals[0], fr_mul(a_1, b_1));");
            w.line("evals[1] = fr_add(evals[1], fr_mul(a_2, b_2));");
            w.line("evals[2] = fr_add(evals[2], fr_mul(a_3, b_3));");
            w.line("evals[3] = fr_add(evals[3], fr_mul(a_4, b_4));");
            w.line("evals[4] = fr_add(evals[4], fr_mul(a_5, b_5));");
            w.line("evals[5] = fr_add(evals[5], fr_mul(a_6, b_6));");
            w.line("evals[6] = fr_add(evals[6], fr_mul(a_7, b_7));");
            w.line("evals[7] = fr_add(evals[7], fr_mul(a_inf, b_inf));");
        }
        w.line("}");
    }

    w.into_string()
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
) -> EvalBody {
    let d = num_inputs_per_product;
    let p = num_products;
    let k = d * p;
    let mut w = Msl::new_at(2048, 2);

    // Precompute differences. When weight-folded, the first index of each
    // product group (g*d) gets w baked in: diff_{g*d} = w * (hi_{g*d} - lo_{g*d}).
    for i in 0..k {
        if weight_folded && i % d == 0 {
            msl!(w, "Fr diff_{i} = fr_mul(w, fr_sub(hi_{i}, lo_{i}));");
        } else {
            msl!(w, "Fr diff_{i} = fr_sub(hi_{i}, lo_{i});");
        }
    }

    // Interpolated values, starting at hi (t=1 point).
    // When weight-folded: cur_{g*d} = w * hi_{g*d}.
    for i in 0..k {
        if weight_folded && i % d == 0 {
            msl!(w, "Fr cur_{i} = fr_mul(w, hi_{i});");
        } else {
            msl!(w, "Fr cur_{i} = hi_{i};");
        }
    }
    w.blank();

    // t=1: product of cur values (with weight baked into first factor if folded)
    emit_product_sum(&mut w, d, p, "cur", 0);

    // t=2, ..., D-1: increment cur by diff, then compute product.
    // The incremental update w*cur_{g*d} += w*diff_{g*d} correctly maintains
    // the weighted first factor across grid points.
    for t in 2..d {
        let eval_idx = t - 1;
        for i in 0..k {
            msl!(w, "cur_{i} = fr_add(cur_{i}, diff_{i});");
        }
        emit_product_sum(&mut w, d, p, "cur", eval_idx);
    }

    // t=inf: product of diffs (leading coefficient).
    // diff_{g*d} already contains w*diff if weight-folded.
    emit_product_sum(&mut w, d, p, "diff", d - 1);

    let msl_str = w.into_string();
    let has_challenges = msl_str.contains("challenges[");
    EvalBody {
        msl: msl_str,
        reads_inline: false,
        fuse_accumulate: false,
        weight_folded,
        has_challenges,
    }
}

/// Emit `evals[idx] = Σ_g Π_j arr[g*D+j]` where `arr` is "cur", "hi", or "diff".
///
/// Uses `fr_mul_unreduced` for intermediate products in the chain (skips the
/// conditional modulus subtraction after each CIOS). The unreduced result
/// (in [0, 2r)) is safe as CIOS input since BN254's 4r²/R < 2r. An explicit
/// `fr_reduce` before `fr_add` brings the final product back to [0, r).
/// Saves `(D-2)` `fr_reduce` calls per product group per grid point.
fn emit_product_sum(w: &mut Msl, d: usize, p: usize, arr: &str, eval_idx: usize) {
    w.line("{ Fr sum = fr_zero();");
    w.push();
    for g in 0..p {
        let base = g * d;
        msl!(w, "{{ Fr prod = {arr}_{base};");
        w.push();
        for j in 1..d {
            msl!(w, "prod = fr_mul_unreduced(prod, {arr}_{});", base + j);
        }
        msl!(w, "sum = fr_add(sum, fr_reduce(prod)); }}");
        w.pop();
    }
    msl!(w, "evals[{eval_idx}] = sum; }}");
    w.pop();
}

/// Generate MSL evaluation body from a [`Formula`].
///
/// Uses incremental interpolation: maintains `cur_k` starting at `lo[k]` and
/// adding `diff_k` for each successive grid point. Grid is `{0, 1, ..., degree-1}`,
/// matching `KernelSpec.num_evals = degree`.
///
/// Challenge values are read from a `device const Fr* challenges` buffer at
/// runtime rather than baked as MSL constants. This makes the kernel shape
/// deterministic at compile time, enabling AOT pipeline caching.
fn generate_formula_body(formula: &Formula, degree: usize) -> EvalBody {
    let num_inputs = formula.num_inputs;
    let num_evals = degree;
    let mut w = Msl::new_at(2048, 2);

    // Precompute diffs
    for k in 0..num_inputs {
        msl!(w, "Fr diff_{k} = fr_sub(hi_{k}, lo_{k});");
    }

    // Running interpolated values: start at lo, add diff each step
    for k in 0..num_inputs {
        msl!(w, "Fr cur_{k} = lo_{k};");
    }
    w.blank();

    // Grid {0, 1, ..., degree-1}: matches KernelSpec.num_evals = degree.
    let grid: Vec<usize> = (0..num_evals).collect();

    let mut prev_t = 0usize;
    for (slot, &t) in grid.iter().enumerate() {
        // Advance cur_k by diff_k for each step from prev_t to t
        let steps = t - prev_t;
        for _ in 0..steps {
            for k in 0..num_inputs {
                msl!(w, "cur_{k} = fr_add(cur_{k}, diff_{k});");
            }
        }
        prev_t = t;

        msl!(w, "{{ // t = {t}");
        w.push();

        // Emit sum-of-products evaluation from the formula terms
        let mut term_names = Vec::with_capacity(formula.terms.len());
        for (ti, term) in formula.terms.iter().enumerate() {
            let term_name = emit_term_msl(&mut w, term, ti);
            term_names.push(term_name);
        }

        // Sum all terms
        if term_names.is_empty() {
            msl!(w, "evals[{slot}] = fr_zero();");
        } else {
            let mut acc = term_names[0].clone();
            for tn in &term_names[1..] {
                msl!(w, "Fr sum_{slot}_{tn} = fr_add({acc}, {tn});");
                acc = format!("sum_{slot}_{tn}");
            }
            msl!(w, "evals[{slot}] = {acc};");
        }

        w.pop();
        w.line("}");
    }

    let has_challenges = formula
        .terms
        .iter()
        .any(|t| t.factors.iter().any(|f| matches!(f, Factor::Challenge(_))));

    EvalBody {
        msl: w.into_string(),
        reads_inline: false,
        fuse_accumulate: false,
        weight_folded: false,
        has_challenges,
    }
}

/// Emit MSL for a single [`ProductTerm`]: `coefficient × Π factors`.
///
/// Returns the MSL variable name holding the term's value.
fn emit_term_msl(w: &mut Msl, term: &ProductTerm, idx: usize) -> String {
    // Start with coefficient
    let coeff_name = format!("c_{idx}");
    match term.coefficient {
        0 => {
            msl!(w, "Fr {coeff_name} = fr_zero();");
            return coeff_name;
        }
        1 => {
            msl!(w, "Fr {coeff_name} = fr_one();");
        }
        -1 => {
            msl!(w, "Fr {coeff_name} = fr_neg(fr_one());");
        }
        v if v > 0 => {
            assert!(
                v <= i128::from(u64::MAX),
                "MSL constant {v} exceeds u64 range"
            );
            msl!(w, "Fr {coeff_name} = fr_from_u64((ulong){v});");
        }
        v => {
            let abs = -v;
            assert!(
                abs <= i128::from(u64::MAX),
                "MSL constant {v} exceeds u64 range"
            );
            msl!(w, "Fr {coeff_name} = fr_neg(fr_from_u64((ulong){abs}));");
        }
    }

    if term.factors.is_empty() {
        return coeff_name;
    }

    // Multiply coefficient by each factor
    let mut acc = coeff_name;
    for (fi, factor) in term.factors.iter().enumerate() {
        let factor_ref = match factor {
            Factor::Input(id) => format!("cur_{id}"),
            Factor::Challenge(id) => format!("challenges[{id}]"),
        };
        let prod_name = format!("t_{idx}_{fi}");
        msl!(w, "Fr {prod_name} = fr_mul({acc}, {factor_ref});");
        acc = prod_name;
    }

    acc
}
