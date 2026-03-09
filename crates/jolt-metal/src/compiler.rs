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
//!   Evaluates on the standard grid `{0, 1, ..., degree}`.
//!
//! The outer reduce kernel uses simdgroup shuffles for the first 32-element
//! reduction (no shared memory barriers), then a small shared memory tree
//! reduction for the remaining 8 simdgroup partial sums (3 barriers total
//! vs D × log2(256) in the naive approach).

use std::fmt::Write;
use std::marker::PhantomData;

use jolt_field::Field;
use jolt_ir::{Expr, ExprVisitor, KernelDescriptor, KernelShape, Var};
use metal::CompileOptions;

use crate::kernel::MetalKernel;
use crate::shaders::{build_source, make_pipeline, SHADER_BN254_FR};

/// Threadgroup size for reduce kernels. Power of 2.
pub(crate) const REDUCE_GROUP_SIZE: usize = 256;

/// Maximum threadgroups per reduce dispatch.
pub(crate) const MAX_REDUCE_GROUPS: usize = 256;

/// Apple GPU SIMD width (32 threads per simdgroup on M-series).
const SIMD_SIZE: usize = 32;

/// Compile a `KernelDescriptor` into a Metal compute pipeline.
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

    let eval_body = match &descriptor.shape {
        KernelShape::ProductSum {
            num_inputs_per_product,
            num_products,
        } => generate_product_sum_body(*num_inputs_per_product, *num_products),
        KernelShape::Custom { expr, num_inputs } => {
            generate_custom_body::<F>(expr, *num_inputs, descriptor.degree, challenges)
        }
    };

    let msl = generate_reduce_kernel(num_inputs, num_evals, &eval_body);
    let source = build_source(&[SHADER_BN254_FR, &msl]);
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(&source, &options)
        .unwrap_or_else(|e| panic!("reduce kernel MSL compilation failed: {e}"));
    let pipeline = make_pipeline(device, &library, "reduce_kernel");

    MetalKernel {
        pipeline,
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

/// Generate the outer reduce kernel MSL that wraps a kernel-specific eval body.
///
/// Uses a two-phase reduction:
/// 1. **Simdgroup shuffle**: Each 32-thread simdgroup reduces via `simd_shuffle_down`
///    (5 rounds, no barriers, no shared memory).
/// 2. **Inter-simdgroup tree**: The 8 simdgroup partial sums are reduced via shared
///    memory with only 3 barrier synchronizations (log2(8)), processing all D eval
///    dimensions simultaneously.
///
/// Buffer layout:
/// - `buffer(0)` .. `buffer(K-1)`: K input buffers
/// - `buffer(K)`: weights
/// - `buffer(K+1)`: partials output (`num_groups * num_evals` elements)
/// - `buffer(K+2)`: params `[n_pairs, half_n, order_flag]`
fn generate_reduce_kernel(num_inputs: usize, num_evals: usize, eval_body: &str) -> String {
    let gs = REDUCE_GROUP_SIZE;
    let num_simdgroups = gs / SIMD_SIZE;
    let mut s = String::with_capacity(8192);

    // Kernel signature with simdgroup metadata
    let _ = writeln!(s, "kernel void reduce_kernel(");
    for k in 0..num_inputs {
        let _ = writeln!(s, "    device const Fr* input_{k} [[buffer({k})]],");
    }
    let _ = writeln!(s, "    device const Fr* weights [[buffer({num_inputs})]],");
    let _ = writeln!(s, "    device Fr* partials [[buffer({})]],", num_inputs + 1);
    let _ = writeln!(
        s,
        "    device const uint* params [[buffer({})]],",
        num_inputs + 2
    );
    let _ = writeln!(s, "    uint gid [[threadgroup_position_in_grid]],");
    let _ = writeln!(s, "    uint lid [[thread_position_in_threadgroup]],");
    let _ = writeln!(s, "    uint num_groups [[threadgroups_per_grid]],");
    let _ = writeln!(s, "    uint simd_lane [[thread_index_in_simdgroup]],");
    let _ = writeln!(s, "    uint simd_id [[simdgroup_index_in_threadgroup]]");
    let _ = writeln!(s, ") {{");

    let _ = writeln!(s, "    uint n_pairs = params[0];");
    let _ = writeln!(s, "    uint half_n = params[1];");
    let _ = writeln!(s, "    uint order = params[2];");
    s.push('\n');

    // Per-thread accumulators
    let _ = writeln!(s, "    Fr acc[{num_evals}];");
    for d in 0..num_evals {
        let _ = writeln!(s, "    acc[{d}] = fr_zero();");
    }
    s.push('\n');

    // Grid-stride loop
    let _ = writeln!(
        s,
        "    for (uint i = gid * {gs}u + lid; i < n_pairs; i += num_groups * {gs}u) {{"
    );

    // Read pairs from all inputs
    let _ = writeln!(s, "        Fr lo[{num_inputs}], hi[{num_inputs}];");
    let _ = writeln!(s, "        if (order == 0u) {{");
    for k in 0..num_inputs {
        let _ = writeln!(
            s,
            "            lo[{k}] = input_{k}[2u * i]; hi[{k}] = input_{k}[2u * i + 1u];"
        );
    }
    let _ = writeln!(s, "        }} else {{");
    for k in 0..num_inputs {
        let _ = writeln!(
            s,
            "            lo[{k}] = input_{k}[i]; hi[{k}] = input_{k}[i + half_n];"
        );
    }
    let _ = writeln!(s, "        }}");
    s.push('\n');

    let _ = writeln!(s, "        Fr w = weights[i];");
    s.push('\n');

    // Kernel-specific evaluation
    let _ = writeln!(s, "        Fr evals[{num_evals}];");
    let _ = write!(s, "{eval_body}");
    s.push('\n');

    // Weighted accumulation
    for d in 0..num_evals {
        let _ = writeln!(
            s,
            "        acc[{d}] = fr_add(acc[{d}], fr_mul(w, evals[{d}]));"
        );
    }
    let _ = writeln!(s, "    }}");
    s.push('\n');

    // Phase 1: Simdgroup reduction via simd_shuffle_down (no barriers)
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

    // Phase 2: Lane 0 of each simdgroup writes to shared memory
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

    // Phase 3: Tree reduction over simdgroup partials (first num_simdgroups threads)
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
            let _ = writeln!(s, "        threadgroup_barrier(mem_flags::mem_threadgroup);");
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

/// Generate MSL evaluation body for ProductSum shape.
///
/// Uses incremental interpolation: instead of `lo[k] + t * diff[k]` (which
/// requires a full `fr_mul` per input per grid point), we maintain `cur[k]`
/// starting at `hi[k]` and add `diff[k]` for each successive grid point.
/// This replaces `(D-2) * K` expensive `fr_mul` with cheap `fr_add`.
fn generate_product_sum_body(num_inputs_per_product: usize, num_products: usize) -> String {
    let d = num_inputs_per_product;
    let p = num_products;
    let k = d * p;
    let mut s = String::with_capacity(2048);

    // Precompute differences
    let _ = writeln!(s, "        Fr diff[{k}];");
    for i in 0..k {
        let _ = writeln!(s, "        diff[{i}] = fr_sub(hi[{i}], lo[{i}]);");
    }

    // Interpolated values, starting at hi (t=1 point)
    let _ = writeln!(s, "        Fr cur[{k}];");
    for i in 0..k {
        let _ = writeln!(s, "        cur[{i}] = hi[{i}];");
    }
    s.push('\n');

    // t=1: product of hi values (cur == hi at this point)
    emit_product_sum(&mut s, d, p, "cur", 0);

    // t=2, ..., D-1: increment cur by diff, then compute product
    for t in 2..d {
        let eval_idx = t - 1;
        for i in 0..k {
            let _ = writeln!(s, "        cur[{i}] = fr_add(cur[{i}], diff[{i}]);");
        }
        emit_product_sum(&mut s, d, p, "cur", eval_idx);
    }

    // t=∞: product of diffs (leading coefficient)
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
            let _ = writeln!(
                s,
                "            prod = fr_mul(prod, {arr}[{}]);",
                base + j
            );
        }
        let _ = writeln!(s, "            sum = fr_add(sum, prod); }}");
    }
    let _ = writeln!(s, "          evals[{eval_idx}] = sum; }}");
}

/// Generate MSL evaluation body for Custom expression shape.
///
/// Evaluates the expression at grid `{0, 1, ..., degree}`, producing
/// `degree + 1` evaluation values. Challenge values are baked as MSL constants.
fn generate_custom_body<F: Field>(
    expr: &Expr,
    num_inputs: usize,
    degree: usize,
    challenges: &[F],
) -> String {
    let num_evals = degree + 1;
    let mut s = String::with_capacity(2048);

    // Precompute diffs for openings
    for k in 0..num_inputs {
        let _ = writeln!(s, "        Fr diff_{k} = fr_sub(hi[{k}], lo[{k}]);");
    }

    // Baked challenge constants
    for (id, val) in challenges.iter().enumerate() {
        let literal = field_to_msl_literal(val);
        let _ = writeln!(s, "        Fr chal_{id} = {literal};");
    }
    s.push('\n');

    // Evaluate at each grid point
    for t in 0..num_evals {
        let _ = writeln!(s, "        {{ // t = {t}");

        // Opening interpolations
        match t {
            0 => {
                for k in 0..num_inputs {
                    let _ = writeln!(s, "            Fr open_{k} = lo[{k}];");
                }
            }
            1 => {
                for k in 0..num_inputs {
                    let _ = writeln!(s, "            Fr open_{k} = hi[{k}];");
                }
            }
            _ => {
                let t_field = F::from_u64(t as u64);
                let literal = field_to_msl_literal(&t_field);
                let _ = writeln!(s, "            Fr tv = {literal};");
                for k in 0..num_inputs {
                    let _ = writeln!(
                        s,
                        "            Fr open_{k} = fr_add(lo[{k}], fr_mul(tv, diff_{k}));"
                    );
                }
            }
        }

        // Walk expression tree via visitor
        let mut visitor = MslCodeGen {
            code: String::new(),
            next_id: 0,
            _marker: PhantomData::<F>,
        };
        let root_name = expr.visit(&mut visitor);
        let _ = write!(s, "{}", visitor.code);
        let _ = writeln!(s, "            evals[{t}] = {root_name};");
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
            Var::Opening(id) => format!("open_{id}"),
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
