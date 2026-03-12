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
//!   MSL assignments in SSA form. Challenge values are baked as constants.
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
use std::marker::PhantomData;

use jolt_field::Field;
use jolt_ir::{Expr, ExprVisitor, KernelDescriptor, KernelShape, Var};
use metal::CompileOptions;

use crate::kernel::MetalKernel;
use crate::shaders::{build_source, make_pipeline, SHADER_BN254_FR, SHADER_WIDE_ACC};

/// Threadgroup size for reduce kernels. Power of 2.
pub(crate) const REDUCE_GROUP_SIZE: usize = 256;

/// Maximum threadgroups per reduce dispatch.
pub(crate) const MAX_REDUCE_GROUPS: usize = 256;

/// Apple GPU SIMD width (32 threads per simdgroup on M-series).
const SIMD_SIZE: usize = 32;

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

/// Compile a `KernelDescriptor` into five Metal compute pipelines.
pub fn compile<F: Field>(device: &metal::Device, descriptor: &KernelDescriptor) -> MetalKernel<F> {
    compile_with_challenges(device, descriptor, &[])
}

/// Compile with baked challenge values for Custom expression kernels.
pub fn compile_with_challenges<F: Field>(
    device: &metal::Device,
    descriptor: &KernelDescriptor,
    challenges: &[F],
) -> MetalKernel<F> {
    let num_inputs = descriptor.num_inputs();
    let num_evals = descriptor.num_evals();

    // Generate eval bodies. When weight_folded=true for ProductSum, the
    // weighted body references `w` (the weight variable), so we need a
    // separate unfolded body for the unweighted kernel variants.
    let (eval_body_weighted, eval_body_unweighted, weight_folded) = match &descriptor.shape {
        KernelShape::ProductSum {
            num_inputs_per_product,
            num_products,
        } => {
            let d = *num_inputs_per_product;
            let p = *num_products;
            // Fold weight into first product factor when it saves multiplies:
            // costs 2*p extra muls (w*hi[g*d], w*diff[g*d]), saves d accumulation muls.
            let fold = d > 2 * p;
            let body = generate_product_sum_body(d, p, fold);
            let body_unw = if fold {
                generate_product_sum_body(d, p, false)
            } else {
                body.clone()
            };
            (body, body_unw, fold)
        }
        KernelShape::EqProduct => {
            // eq * g: out = [P(0), P(2)] — t=1 derived from claim
            let body = r"
        evals[0] = fr_mul(lo[0], lo[1]);
        Fr a2 = fr_sub(fr_add(hi[0], hi[0]), lo[0]);
        Fr b2 = fr_sub(fr_add(hi[1], hi[1]), lo[1]);
        evals[1] = fr_mul(a2, b2);"
                .to_string();
            (body.clone(), body, false)
        }
        KernelShape::HammingBooleanity => {
            // eq * h * (h-1): out = [P(0), P(2), P(3)] — t=1 derived from claim
            // Incremental interpolation: eq_val = lo[0] + t*d_eq, h_val = lo[1] + t*d_h
            // Grid: {0, 2, 3} (skipping t=1). Start cur at t=2 (= hi + diff).
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
            let body = generate_custom_body::<F>(expr, *num_inputs, descriptor.degree, challenges);
            (body.clone(), body, false)
        }
    };

    let mut msl = String::with_capacity(65536);

    // Generate 3 weighted variants (L2H, H2L, Tensor)
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
        ));
        msl.push('\n');
    }

    // Generate 2 unweighted variants (L2H, H2L) — use unfolded body
    for variant in [KernelVariant::LowToHigh, KernelVariant::HighToLow] {
        msl.push_str(&generate_reduce_kernel(
            num_inputs,
            num_evals,
            &eval_body_unweighted,
            variant,
            false,
            false,
        ));
        msl.push('\n');
    }

    let source = build_source(&[SHADER_BN254_FR, SHADER_WIDE_ACC, &msl]);
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(&source, &options)
        .unwrap_or_else(|e| panic!("reduce kernel MSL compilation failed: {e}"));

    MetalKernel {
        pipeline_l2h: make_pipeline(device, &library, "reduce_kernel_l2h"),
        pipeline_h2l: make_pipeline(device, &library, "reduce_kernel_h2l"),
        pipeline_tensor: make_pipeline(device, &library, "reduce_kernel_tensor"),
        pipeline_l2h_unw: make_pipeline(device, &library, "reduce_kernel_l2h_unw"),
        pipeline_h2l_unw: make_pipeline(device, &library, "reduce_kernel_h2l_unw"),
        num_evals,
        num_inputs,
        _marker: PhantomData,
    }
}

/// Convert a field element to an MSL `Fr` aggregate initializer.
///
/// Produces `{ { l0u, l1u, ..., l7u } }` matching the `struct Fr { uint limbs[8]; }` layout.
/// SAFETY: Relies on identical byte layout between CPU `[u64; 4]` and Metal `[u32; 8]`
/// on little-endian ARM64.
fn field_to_msl_literal<F: Field>(val: &F) -> String {
    debug_assert_eq!(std::mem::size_of::<F>(), 32);
    // SAFETY: F is 32 bytes (BN254 Fr in Montgomery form). On LE ARM64,
    // [u64; 4] and [u32; 8] have identical byte layout.
    let limbs: [u32; 8] = unsafe { std::ptr::read(std::ptr::from_ref(val).cast()) };
    format!(
        "{{ {{ {} }} }}",
        limbs
            .iter()
            .map(|l| format!("{l}u"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn i128_to_field<F: Field>(val: i128) -> F {
    if val >= 0 {
        F::from_u64(val as u64)
    } else {
        -F::from_u64((-val) as u64)
    }
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

    // Reduce WideAcc to Fr
    let _ = writeln!(s, "    Fr acc[{num_evals}];");
    let needs_to_mont = matches!(strategy, AccumulationStrategy::DirectAdd);
    if needs_to_mont {
        for d in 0..num_evals {
            let _ = writeln!(s, "    acc[{d}] = fr_to_mont(acc_reduce(wide_acc[{d}]));");
        }
    } else {
        for d in 0..num_evals {
            let _ = writeln!(s, "    acc[{d}] = acc_reduce(wide_acc[{d}]);");
        }
    }
    s.push('\n');

    // Simdgroup reduction via simd_shuffle_down (no barriers needed)
    let half_simd = SIMD_SIZE / 2;
    let _ = writeln!(
        s,
        "    for (ushort _off = {half_simd}u; _off > 0u; _off >>= 1u) {{"
    );
    for d in 0..num_evals {
        let _ = writeln!(s, "        {{ Fr _o;");
        for l in 0..8 {
            let _ = writeln!(
                s,
                "        _o.limbs[{l}] = simd_shuffle_down(acc[{d}].limbs[{l}], _off);"
            );
        }
        let _ = writeln!(s, "        acc[{d}] = fr_add(acc[{d}], _o); }}");
    }
    let _ = writeln!(s, "    }}");
    s.push('\n');

    // Lane 0 of each simdgroup writes to shared memory
    let sh_size = num_evals * num_simdgroups;
    let _ = writeln!(s, "    threadgroup Fr sh[{sh_size}];");
    let _ = writeln!(s, "    if (simd_lane == 0u) {{");
    for d in 0..num_evals {
        let _ = writeln!(
            s,
            "        sh[{}u + simd_id] = acc[{d}];",
            d * num_simdgroups
        );
    }
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    threadgroup_barrier(mem_flags::mem_threadgroup);");
    s.push('\n');

    // Tree reduction over simdgroup partials
    let _ = writeln!(s, "    if (lid < {num_simdgroups}u) {{");
    let mut stride = num_simdgroups / 2;
    while stride > 0 {
        let _ = writeln!(s, "        if (lid < {stride}u) {{");
        for d in 0..num_evals {
            let base = d * num_simdgroups;
            let _ = writeln!(
                s,
                "            sh[{base}u + lid] = fr_add(sh[{base}u + lid], sh[{base}u + lid + {stride}u]);"
            );
        }
        let _ = writeln!(s, "        }}");
        if stride > 1 {
            let _ = writeln!(
                s,
                "        threadgroup_barrier(mem_flags::mem_threadgroup);"
            );
        }
        stride /= 2;
    }
    let _ = writeln!(s, "    }}");
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
fn emit_product_sum(s: &mut String, d: usize, p: usize, arr: &str, eval_idx: usize) {
    let _ = writeln!(s, "        {{ Fr sum = fr_zero();");
    for g in 0..p {
        let base = g * d;
        let _ = writeln!(s, "          {{ Fr prod = {arr}[{base}];");
        for j in 1..d {
            let _ = writeln!(s, "            prod = fr_mul(prod, {arr}[{}]);", base + j);
        }
        let _ = writeln!(s, "            sum = fr_add(sum, prod); }}");
    }
    let _ = writeln!(s, "          evals[{eval_idx}] = sum; }}");
}

/// Generate MSL evaluation body for Custom expression shape.
///
/// Uses incremental interpolation: maintains `cur_k` starting at `lo[k]` and
/// adding `diff_k` for each successive grid point. Grid is `{0, 2, 3, ..., degree}`
/// (skipping `t=1`), so slot 0 maps to `t=0` and slot `k>=1` maps to `t=k+1`.
fn generate_custom_body<F: Field>(
    expr: &Expr,
    num_inputs: usize,
    degree: usize,
    challenges: &[F],
) -> String {
    // Grid: {0, 2, 3, ..., degree} — `degree` evaluations, skipping t=1
    let num_evals = degree;
    let mut s = String::with_capacity(2048);

    // Precompute diffs
    for k in 0..num_inputs {
        let _ = writeln!(s, "        Fr diff_{k} = fr_sub(hi[{k}], lo[{k}]);");
    }

    // Baked challenge constants
    for (id, val) in challenges.iter().enumerate() {
        let literal = field_to_msl_literal(val);
        let _ = writeln!(s, "        Fr chal_{id} = {literal};");
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

        // Walk expression tree (visitor emits cur_{id} for openings)
        let mut visitor = MslCodeGen {
            code: String::new(),
            next_id: 0,
            _marker: PhantomData::<F>,
        };
        let root_name = expr.visit(&mut visitor);
        let _ = write!(s, "{}", visitor.code);
        let _ = writeln!(s, "            evals[{slot}] = {root_name};");
        let _ = writeln!(s, "        }}");
    }

    s
}

/// ExprVisitor that emits MSL assignments in SSA form.
struct MslCodeGen<F: Field> {
    code: String,
    next_id: usize,
    _marker: PhantomData<F>,
}

impl<F: Field> MslCodeGen<F> {
    fn fresh_var(&mut self) -> String {
        let name = format!("e{}", self.next_id);
        self.next_id += 1;
        name
    }
}

impl<F: Field> ExprVisitor for MslCodeGen<F> {
    type Output = String;

    fn visit_constant(&mut self, val: i128) -> String {
        let name = self.fresh_var();
        let f_val = i128_to_field::<F>(val);
        let literal = field_to_msl_literal(&f_val);
        let _ = writeln!(self.code, "            Fr {name} = {literal};");
        name
    }

    fn visit_var(&mut self, var: Var) -> String {
        match var {
            Var::Opening(id) => format!("cur_{id}"),
            Var::Challenge(id) => format!("chal_{id}"),
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
