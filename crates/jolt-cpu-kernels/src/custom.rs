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
    CpuKernel::new(move |lo: &[F], hi: &[F], degree: usize| {
        let num_outputs = degree + 1;
        let mut evals = Vec::with_capacity(num_outputs);
        let mut stack: Vec<F> = Vec::with_capacity(ops.len());

        for t in 0..num_outputs {
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
            evals.push(stack[0]);
        }

        evals
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

    #[test]
    fn constant_expr() {
        let b = ExprBuilder::new();
        let c = b.constant(42);
        let expr = b.build(c);

        let ops = compile_ops::<Fr>(&expr, &[]);
        assert_eq!(ops.len(), 1);

        let kernel: CpuKernel<Fr> = compile_with_challenges(&expr, 0, 1, &[]);
        let result = kernel.evaluate(&[], &[], 2);
        assert_eq!(result, vec![Fr::from_u64(42); 3]);
    }

    #[test]
    fn single_opening() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(a);

        let kernel: CpuKernel<Fr> = compile_with_challenges(&expr, 1, 1, &[]);
        let lo = vec![Fr::from_u64(3)];
        let hi = vec![Fr::from_u64(7)];
        let result = kernel.evaluate(&lo, &hi, 1);

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
        let result = kernel.evaluate(&lo, &hi, 1);

        // t=0: 3+10=13, t=1: 7+20=27
        assert_eq!(result[0], Fr::from_u64(13));
        assert_eq!(result[1], Fr::from_u64(27));
    }

    #[test]
    fn negation() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(-a);

        let kernel: CpuKernel<Fr> = compile_with_challenges(&expr, 1, 1, &[]);
        let lo = vec![Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(5)];
        let result = kernel.evaluate(&lo, &hi, 0);

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
        let result = kernel.evaluate(&lo, &hi, 1);

        // t=0: 10-3=7, t=1: 20-7=13
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
        let result = kernel.evaluate(&lo, &hi, 2);

        // t=0: (2+1)*(5-2) = 3*3 = 9
        assert_eq!(result[0], Fr::from_u64(9));
        // t=1: (4+1)*(9-4) = 5*5 = 25
        assert_eq!(result[1], Fr::from_u64(25));
        // t=2: a=2+2*(4-2)=6, bv=5+2*(9-5)=13 -> (6+1)*(13-6) = 7*7 = 49
        assert_eq!(result[2], Fr::from_u64(49));
    }

    #[test]
    fn challenge_binding_single() {
        // expr = c0 * o0  (gamma * opening)
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * a);

        let gamma_val = Fr::from_u64(7);
        let kernel = compile_with_challenges::<Fr>(&expr, 1, 1, &[gamma_val]);

        let lo = vec![Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(10)];
        let result = kernel.evaluate(&lo, &hi, 1);

        // t=0: 7*5 = 35
        assert_eq!(result[0], Fr::from_u64(35));
        // t=1: 7*10 = 70
        assert_eq!(result[1], Fr::from_u64(70));
    }

    #[test]
    fn challenge_binding_multiple() {
        // expr = c0 * o0 + c1 * o1
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
        let result = kernel.evaluate(&lo, &hi, 1);

        // Both t=0 and t=1: 3*10 + 5*20 = 30 + 100 = 130
        assert_eq!(result[0], Fr::from_u64(130));
        assert_eq!(result[1], Fr::from_u64(130));
    }

    #[test]
    fn challenge_binding_booleanity_weighted() {
        // Weighted booleanity: c0 * (o0^2 - o0)
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));

        let gamma_val = Fr::from_u64(11);
        let kernel = compile_with_challenges::<Fr>(&expr, 1, 2, &[gamma_val]);

        let lo = vec![Fr::from_u64(3)];
        let hi = vec![Fr::from_u64(7)];
        let result = kernel.evaluate(&lo, &hi, 2);

        // t=0: 11 * (9 - 3) = 66
        assert_eq!(result[0], Fr::from_u64(66));
        // t=1: 11 * (49 - 7) = 462
        assert_eq!(result[1], Fr::from_u64(462));
    }

    #[test]
    fn missing_challenge_defaults_to_zero() {
        // expr = c0 * o0 + c1 * o1, but only supply c0
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c0 = b.challenge(0);
        let c1 = b.challenge(1);
        let expr = b.build(c0 * a + c1 * bv);

        // Only bind c0; c1 defaults to zero
        let kernel = compile_with_challenges::<Fr>(&expr, 2, 1, &[Fr::from_u64(3)]);

        let lo = vec![Fr::from_u64(10), Fr::from_u64(20)];
        let hi = vec![Fr::from_u64(10), Fr::from_u64(20)];
        let result = kernel.evaluate(&lo, &hi, 1);

        // 3*10 + 0*20 = 30
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
        assert_eq!(k1.evaluate(&lo, &hi, 2), k2.evaluate(&lo, &hi, 2),);
    }
}
