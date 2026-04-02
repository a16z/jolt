//! Generic eval closure compiler.
//!
//! Compiles any [`KernelSpec`]'s sum-of-products composition into an eval
//! closure on the contiguous grid `{0, 1, 2, …, num_evals-1}`.
//!
//! This is the fallback for compositions that don't match the ProductSum
//! fast-path (see [`super::product_sum`]).

use jolt_compiler::{Factor, Formula};
use jolt_field::Field;

/// Compile a [`KernelSpec`]'s composition into a boxable eval closure.
///
/// Returns the closure only — the caller wraps it in [`CpuKernel`] with
/// the appropriate metadata from the spec.
pub fn compile_fn<F: Field>(
    formula: &Formula,
) -> impl Fn(&[F], &[F], &[F], &mut [F]) + Send + Sync + 'static {
    let terms: Vec<_> = formula
        .terms
        .iter()
        .map(|t| {
            let coeff = F::from_i128(t.coefficient);
            let factors: Vec<_> = t
                .factors
                .iter()
                .map(|f| match f {
                    Factor::Input(i) => CompiledFactor::Input(*i as usize),
                    Factor::Challenge(i) => CompiledFactor::Challenge(*i as usize),
                })
                .collect();
            (coeff, factors)
        })
        .collect();

    move |lo: &[F], hi: &[F], challenges: &[F], out: &mut [F]| {
        for (slot_idx, slot) in out.iter_mut().enumerate() {
            let t_f = F::from_u64(slot_idx as u64);

            let mut sum = F::zero();
            for (coeff, factors) in &terms {
                let mut val = *coeff;
                for factor in factors {
                    val *= match *factor {
                        CompiledFactor::Input(i) => lo[i] + t_f * (hi[i] - lo[i]),
                        CompiledFactor::Challenge(i) => {
                            challenges.get(i).copied().unwrap_or_else(F::zero)
                        }
                    };
                }
                sum += val;
            }
            *slot = sum;
        }
    }
}

#[cfg(test)]
pub fn compile<F: Field>(formula: &Formula) -> crate::CpuKernel<F> {
    use jolt_compiler::{BindingOrder, Iteration};
    crate::CpuKernel::new(
        compile_fn(formula),
        formula.degree() + 1,
        Iteration::Dense,
        BindingOrder::LowToHigh,
    )
}

/// Factor resolved at kernel eval time.
#[derive(Clone, Copy)]
enum CompiledFactor {
    Input(usize),
    Challenge(usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuKernel;
    use jolt_compiler::{Factor, Formula, ProductTerm};
    use jolt_field::Fr;
    use num_traits::Zero;

    fn eval_kernel(kernel: &CpuKernel<Fr>, lo: &[Fr], hi: &[Fr], n: usize) -> Vec<Fr> {
        let mut out = vec![Fr::zero(); n];
        kernel.evaluate(lo, hi, &[], &mut out);
        out
    }

    fn eval_kernel_with(
        kernel: &CpuKernel<Fr>,
        lo: &[Fr],
        hi: &[Fr],
        challenges: &[Fr],
        n: usize,
    ) -> Vec<Fr> {
        let mut out = vec![Fr::zero(); n];
        kernel.evaluate(lo, hi, challenges, &mut out);
        out
    }

    #[test]
    fn simple_product() {
        // a * b, degree 2 → 3 evals on grid {0, 1, 2}
        let formula = Formula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(1)],
        }]);
        let kernel = compile::<Fr>(&formula);

        let lo = vec![Fr::from_u64(3), Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(7), Fr::from_u64(11)];
        let result = eval_kernel(&kernel, &lo, &hi, 3);

        // t=0: 3*5 = 15
        assert_eq!(result[0], Fr::from_u64(15));
        // t=1: 7*11 = 77
        assert_eq!(result[1], Fr::from_u64(77));
        // t=2: (3+2*4)*(5+2*6) = 11*17 = 187
        assert_eq!(result[2], Fr::from_u64(187));
    }

    #[test]
    fn booleanity_with_challenge() {
        // gamma * (h^2 - h), degree 2 → 3 evals on grid {0, 1, 2}
        let formula = Formula::from_terms(vec![
            ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Challenge(0), Factor::Input(0), Factor::Input(0)],
            },
            ProductTerm {
                coefficient: -1,
                factors: vec![Factor::Challenge(0), Factor::Input(0)],
            },
        ]);
        let kernel = compile::<Fr>(&formula);

        let lo = vec![Fr::from_u64(3)];
        let hi = vec![Fr::from_u64(7)];
        let challenges = [Fr::from_u64(11)];
        let result = eval_kernel_with(&kernel, &lo, &hi, &challenges, 3);

        // t=0: h=3, 11*(9-3) = 66
        assert_eq!(result[0], Fr::from_u64(66));
        // t=1: h=7, 11*(49-7) = 462
        assert_eq!(result[1], Fr::from_u64(462));
        // t=2: h=11, 11*(121-11) = 1210
        assert_eq!(result[2], Fr::from_u64(1210));
    }

    #[test]
    fn linear_combination() {
        // c0*a + c1*b
        let formula = Formula::from_terms(vec![
            ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Challenge(0), Factor::Input(0)],
            },
            ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Challenge(1), Factor::Input(1)],
            },
        ]);
        let kernel = compile::<Fr>(&formula);
        let challenges = vec![Fr::from_u64(3), Fr::from_u64(5)];

        let lo = vec![Fr::from_u64(10), Fr::from_u64(20)];
        let hi = vec![Fr::from_u64(10), Fr::from_u64(20)];
        let result = eval_kernel_with(&kernel, &lo, &hi, &challenges, 2);

        // Both are constant: 3*10 + 5*20 = 130
        assert_eq!(result[0], Fr::from_u64(130));
        assert_eq!(result[1], Fr::from_u64(130));
    }

    #[test]
    fn missing_challenge_defaults_to_zero() {
        let formula = Formula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Challenge(0), Factor::Input(0)],
        }]);
        let kernel = compile::<Fr>(&formula);

        let lo = vec![Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(10)];
        // No challenges provided — defaults to zero
        let result = eval_kernel(&kernel, &lo, &hi, 1);
        assert_eq!(result[0], Fr::zero());
    }

    #[test]
    fn constant_only() {
        let formula = Formula::from_terms(vec![ProductTerm {
            coefficient: 42,
            factors: vec![],
        }]);
        let kernel = compile::<Fr>(&formula);
        let result = eval_kernel(&kernel, &[], &[], 3);
        assert_eq!(result, vec![Fr::from_u64(42); 3]);
    }
}
