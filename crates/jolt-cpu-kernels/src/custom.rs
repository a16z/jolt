//! Custom kernel compilation from symbolic expressions.
//!
//! Compiles an [`Expr`] into a [`CpuKernel`] closure by converting the
//! expression tree into a vector of stack-machine operations at compile
//! time. At evaluation time, the kernel executes these ops once per grid
//! point — no allocations, no tree traversal.

use jolt_compute::CpuKernel;
use jolt_field::Field;
use jolt_ir::{Expr, ExprNode, Var};

/// Stack-machine operation with field constants baked in.
///
/// Produced at compile time by walking the `Expr` tree. At evaluation time,
/// the kernel iterates this flat sequence once per grid point — no recursion,
/// no allocation, predictable branches.
#[derive(Clone, Copy)]
enum CompiledOp<F> {
    /// Push a baked field constant (from `ExprNode::Constant(i128)`).
    Constant(F),
    /// Push the interpolated value of opening `idx`:
    /// `lo[idx] + t * (hi[idx] - lo[idx])`
    Opening(usize),
    /// Push a baked challenge value.
    Challenge(F),
    /// Pop, negate, push.
    Neg,
    /// Pop two, add, push.
    Add,
    /// Pop two, subtract (second popped from first popped), push.
    Sub,
    /// Pop two, multiply, push.
    Mul,
}

/// Walk the expression tree and produce a post-order sequence of ops.
///
/// Challenge variables are resolved from `challenges` by index. If a
/// challenge index is out of bounds, it is baked as `F::zero()`.
fn compile_ops<F: Field>(expr: &Expr, challenges: &[F]) -> Vec<CompiledOp<F>> {
    let mut ops = Vec::with_capacity(expr.len());
    compile_node(expr, expr.root(), challenges, &mut ops);
    ops
}

fn compile_node<F: Field>(
    expr: &Expr,
    id: jolt_ir::ExprId,
    challenges: &[F],
    ops: &mut Vec<CompiledOp<F>>,
) {
    match expr.get(id) {
        ExprNode::Constant(val) => {
            ops.push(CompiledOp::Constant(F::from_i128(val)));
        }
        ExprNode::Var(Var::Opening(idx)) => {
            ops.push(CompiledOp::Opening(idx as usize));
        }
        ExprNode::Var(Var::Challenge(idx)) => {
            let val = challenges
                .get(idx as usize)
                .copied()
                .unwrap_or_else(F::zero);
            ops.push(CompiledOp::Challenge(val));
        }
        ExprNode::Neg(inner) => {
            compile_node(expr, inner, challenges, ops);
            ops.push(CompiledOp::Neg);
        }
        ExprNode::Add(lhs, rhs) => {
            compile_node(expr, lhs, challenges, ops);
            compile_node(expr, rhs, challenges, ops);
            ops.push(CompiledOp::Add);
        }
        ExprNode::Sub(lhs, rhs) => {
            compile_node(expr, lhs, challenges, ops);
            compile_node(expr, rhs, challenges, ops);
            ops.push(CompiledOp::Sub);
        }
        ExprNode::Mul(lhs, rhs) => {
            compile_node(expr, lhs, challenges, ops);
            compile_node(expr, rhs, challenges, ops);
            ops.push(CompiledOp::Mul);
        }
    }
}

/// Build a `CpuKernel<F>` from a compiled op sequence.
fn kernel_from_ops<F: Field>(ops: Vec<CompiledOp<F>>) -> CpuKernel<F> {
    CpuKernel::new(move |lo: &[F], hi: &[F], out: &mut [F]| {
        let mut stack: Vec<F> = Vec::with_capacity(ops.len());

        for (t, slot) in out.iter_mut().enumerate() {
            let t_f = F::from_u64(t as u64);
            stack.clear();

            for op in &ops {
                match *op {
                    CompiledOp::Constant(c) => stack.push(c),
                    CompiledOp::Opening(idx) => {
                        let val = lo[idx] + t_f * (hi[idx] - lo[idx]);
                        stack.push(val);
                    }
                    CompiledOp::Challenge(c) => stack.push(c),
                    CompiledOp::Neg => {
                        let v = stack.pop().unwrap();
                        stack.push(-v);
                    }
                    CompiledOp::Add => {
                        let b = stack.pop().unwrap();
                        let a = stack.pop().unwrap();
                        stack.push(a + b);
                    }
                    CompiledOp::Sub => {
                        let b = stack.pop().unwrap();
                        let a = stack.pop().unwrap();
                        stack.push(a - b);
                    }
                    CompiledOp::Mul => {
                        let b = stack.pop().unwrap();
                        let a = stack.pop().unwrap();
                        stack.push(a * b);
                    }
                }
            }

            debug_assert_eq!(
                stack.len(),
                1,
                "expression should leave exactly one value on stack"
            );
            *slot = stack[0];
        }
    })
}

/// Compile an `Expr` into a `CpuKernel<F>` with challenge values baked in.
///
/// The expression is walked once at compile time to produce a flat sequence
/// of stack-machine operations. At evaluation time, the kernel runs this
/// sequence once per grid point `t ∈ {0, ..., degree}`.
///
/// - **Opening variables** are interpolated per invocation:
///   `val = lo[idx] + t * (hi[idx] - lo[idx])`
/// - **Challenge variables** are baked as constants from `challenges[idx]`.
///   Out-of-bounds challenge indices are baked as `F::zero()`.
/// - **Constants** are baked as `F::from_i128(val)`.
pub fn compile_with_challenges<F: Field>(
    expr: &Expr,
    _num_inputs: usize,
    _degree: usize,
    challenges: &[F],
) -> CpuKernel<F> {
    let ops = compile_ops::<F>(expr, challenges);
    kernel_from_ops(ops)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_ir::ExprBuilder;
    use num_traits::Zero;

    fn eval_kernel(kernel: &CpuKernel<Fr>, lo: &[Fr], hi: &[Fr], n: usize) -> Vec<Fr> {
        let mut out = vec![Fr::zero(); n];
        kernel.evaluate(lo, hi, &mut out);
        out
    }

    #[test]
    fn constant_expr() {
        let b = ExprBuilder::new();
        let c = b.constant(42);
        let expr = b.build(c);

        let ops = compile_ops::<Fr>(&expr, &[]);
        assert_eq!(ops.len(), 1);

        let kernel: CpuKernel<Fr> = compile_with_challenges(&expr, 0, 1, &[]);
        assert_eq!(eval_kernel(&kernel, &[], &[], 3), vec![Fr::from_u64(42); 3]);
    }

    #[test]
    fn single_opening() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(a);

        let kernel: CpuKernel<Fr> = compile_with_challenges(&expr, 1, 1, &[]);
        let result = eval_kernel(&kernel, &[Fr::from_u64(3)], &[Fr::from_u64(7)], 2);

        assert_eq!(result[0], Fr::from_u64(3)); // t=0
        assert_eq!(result[1], Fr::from_u64(7)); // t=1
    }

    #[test]
    fn add_two_openings() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(a + bv);

        let kernel: CpuKernel<Fr> = compile_with_challenges(&expr, 2, 1, &[]);
        let lo = vec![Fr::from_u64(3), Fr::from_u64(10)];
        let hi = vec![Fr::from_u64(7), Fr::from_u64(20)];
        let result = eval_kernel(&kernel, &lo, &hi, 2);

        assert_eq!(result[0], Fr::from_u64(13));
        assert_eq!(result[1], Fr::from_u64(27));
    }

    #[test]
    fn negation() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(-a);

        let kernel: CpuKernel<Fr> = compile_with_challenges(&expr, 1, 1, &[]);
        let result = eval_kernel(&kernel, &[Fr::from_u64(5)], &[Fr::from_u64(5)], 1);
        assert_eq!(result[0], -Fr::from_u64(5));
    }

    #[test]
    fn subtraction() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(a - bv);

        let kernel: CpuKernel<Fr> = compile_with_challenges(&expr, 2, 1, &[]);
        let lo = vec![Fr::from_u64(10), Fr::from_u64(3)];
        let hi = vec![Fr::from_u64(20), Fr::from_u64(7)];
        let result = eval_kernel(&kernel, &lo, &hi, 2);

        assert_eq!(result[0], Fr::from_u64(7));
        assert_eq!(result[1], Fr::from_u64(13));
    }

    #[test]
    fn complex_expression() {
        // (o0 + 1) * (o1 - o0)
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let one = b.constant(1);
        let expr = b.build((a + one) * (bv - a));

        let kernel: CpuKernel<Fr> = compile_with_challenges(&expr, 2, 2, &[]);
        let lo = vec![Fr::from_u64(2), Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(4), Fr::from_u64(9)];
        let result = eval_kernel(&kernel, &lo, &hi, 3);

        assert_eq!(result[0], Fr::from_u64(9));
        assert_eq!(result[1], Fr::from_u64(25));
        assert_eq!(result[2], Fr::from_u64(49));
    }

    #[test]
    fn challenge_binding_single() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * a);

        let kernel = compile_with_challenges::<Fr>(&expr, 1, 1, &[Fr::from_u64(7)]);
        let result = eval_kernel(&kernel, &[Fr::from_u64(5)], &[Fr::from_u64(10)], 2);

        assert_eq!(result[0], Fr::from_u64(35));
        assert_eq!(result[1], Fr::from_u64(70));
    }

    #[test]
    fn challenge_binding_multiple() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c0 = b.challenge(0);
        let c1 = b.challenge(1);
        let expr = b.build(c0 * a + c1 * bv);

        let challenges = vec![Fr::from_u64(3), Fr::from_u64(5)];
        let kernel = compile_with_challenges::<Fr>(&expr, 2, 1, &challenges);
        let lo = vec![Fr::from_u64(10), Fr::from_u64(20)];
        let hi = vec![Fr::from_u64(10), Fr::from_u64(20)];
        let result = eval_kernel(&kernel, &lo, &hi, 2);

        assert_eq!(result[0], Fr::from_u64(130));
        assert_eq!(result[1], Fr::from_u64(130));
    }

    #[test]
    fn challenge_binding_booleanity_weighted() {
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));

        let kernel = compile_with_challenges::<Fr>(&expr, 1, 2, &[Fr::from_u64(11)]);
        let result = eval_kernel(&kernel, &[Fr::from_u64(3)], &[Fr::from_u64(7)], 3);

        assert_eq!(result[0], Fr::from_u64(66));
        assert_eq!(result[1], Fr::from_u64(462));
    }

    #[test]
    fn missing_challenge_defaults_to_zero() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c0 = b.challenge(0);
        let c1 = b.challenge(1);
        let expr = b.build(c0 * a + c1 * bv);

        let kernel = compile_with_challenges::<Fr>(&expr, 2, 1, &[Fr::from_u64(3)]);
        let lo = vec![Fr::from_u64(10), Fr::from_u64(20)];
        let hi = vec![Fr::from_u64(10), Fr::from_u64(20)];
        let result = eval_kernel(&kernel, &lo, &hi, 2);

        assert_eq!(result[0], Fr::from_u64(30));
    }

    #[test]
    fn no_challenges_same_as_empty_slice() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(a * a);

        let k1: CpuKernel<Fr> = compile_with_challenges(&expr, 1, 2, &[]);
        let k2: CpuKernel<Fr> = compile_with_challenges(&expr, 1, 2, &[]);

        let lo = vec![Fr::from_u64(4)];
        let hi = vec![Fr::from_u64(6)];
        assert_eq!(eval_kernel(&k1, &lo, &hi, 3), eval_kernel(&k2, &lo, &hi, 3));
    }
}
