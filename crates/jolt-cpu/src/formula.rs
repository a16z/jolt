//! Generic kernel compilation from [`Formula`].
//!
//! Compiles the normalized sum-of-products representation into a [`CpuKernel`]
//! that evaluates on the standard grid `{0, 2, 3, …, degree}` (skipping `t=1`,
//! which is derived from the sumcheck claim externally).
//!
//! This handles all formulas that don't match a specialized fast-path
//! (ProductSum, eq-product, Hamming booleanity).

use crate::CpuKernel;
use jolt_compiler::{Factor, Formula};
use jolt_field::Field;

/// Compile a [`Formula`] into a [`CpuKernel`].
///
/// Challenge factors are resolved at eval time from the `challenges` slice
/// passed to [`CpuKernel::evaluate`]. Out-of-bounds challenge indices
/// evaluate to `F::zero()`.
pub fn compile<F: Field>(formula: &Formula) -> CpuKernel<F> {
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

    CpuKernel::new(move |lo: &[F], hi: &[F], challenges: &[F], out: &mut [F]| {
        for (slot_idx, slot) in out.iter_mut().enumerate() {
            // Standard grid: {0, 2, 3, …} — slot 0 → t=0, slot k≥1 → t=k+1
            let t = if slot_idx == 0 { 0 } else { slot_idx + 1 };
            let t_f = F::from_u64(t as u64);

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
    })
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
        // a * b
        let formula = Formula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(1)],
        }]);
        let kernel = compile::<Fr>(&formula);

        let lo = vec![Fr::from_u64(3), Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(7), Fr::from_u64(11)];
        // Grid: {0, 2} — 2 evals (degree 2, skipping t=1)
        let result = eval_kernel(&kernel, &lo, &hi, 2);

        // t=0: 3*5 = 15
        assert_eq!(result[0], Fr::from_u64(15));
        // t=2: (3+2*4)*(5+2*6) = 11*17 = 187
        assert_eq!(result[1], Fr::from_u64(187));
    }

    #[test]
    fn booleanity_with_challenge() {
        // gamma * (h^2 - h) = gamma*h*h - gamma*h
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
        // Grid: {0, 2} — 2 evals (degree 2)
        let result = eval_kernel_with(&kernel, &lo, &hi, &challenges, 2);

        // t=0: 11*(3*3 - 3) = 11*6 = 66
        assert_eq!(result[0], Fr::from_u64(66));
        // t=2: h=3+2*4=11, 11*(11*11 - 11) = 11*110 = 1210
        assert_eq!(result[1], Fr::from_u64(1210));
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
