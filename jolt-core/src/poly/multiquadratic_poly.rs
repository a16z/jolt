use allocative::Allocative;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::{BindingOrder, PolynomialBinding};

/// Multiquadratic polynomial represented by its evaluations on the grid
/// {0, 1, ∞}^num_vars in base-3 layout (z_0 least-significant / fastest-varying).
#[derive(Allocative)]
pub struct MultiquadraticPolynomial<F: JoltField> {
    num_vars: usize,
    evals: Vec<F>,
}

impl<F: JoltField> MultiquadraticPolynomial<F> {
    /// Construct a multiquadratic polynomial from its full grid of evaluations.
    /// The caller is responsible for ensuring that `evals` is laid out in base-3
    /// order with z_0 as the least-significant digit.
    pub fn new(num_vars: usize, evals: Vec<F>) -> Self {
        let expected_len = 3usize.pow(num_vars as u32);
        debug_assert!(
            evals.len() == expected_len,
            "MultiquadraticPolynomial: expected {} evals, got {}",
            expected_len,
            evals.len()
        );
        Self { num_vars, evals }
    }

    /// Number of variables in the polynomial.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Underlying evaluations on {0, 1, ∞}^num_vars.
    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    /// Given evaluations of a degree-1 multivariate polynomial over {0,1}^dim,
    /// expand them to the corresponding multiquadratic grid over {0,1,∞}^dim.
    ///
    /// The input is a length-2^dim slice `input` containing evaluations on the
    /// Boolean hypercube. The caller must provide two length-3^dim buffers:
    /// - `output` will contain the final {0,1,∞}^dim values on return
    /// - `tmp` is a scratch buffer which this routine may use internally
    ///
    /// Layout is product-order with the last variable as the fastest-varying
    /// coordinate. For each 1D slice (f0, f1) along a new dimension we write
    /// (f(0), f(1), f(∞)) = (f0, f1, f1 - f0), so ∞ stores the slope.
    #[inline(always)]
    pub fn expand_linear_grid_to_multiquadratic(
        input: &[F],      // initial buffer (size 2^dim)
        output: &mut [F], // final buffer (size 3^dim)
        tmp: &mut [F],    // scratch buffer, also (size 3^dim)
        dim: usize,
    ) {
        let in_size = 1usize << dim;
        let out_size = 3usize.pow(dim as u32);

        debug_assert_eq!(input.len(), in_size);
        debug_assert_eq!(output.len(), out_size);
        debug_assert_eq!(tmp.len(), out_size);

        match dim {
            0 => {
                output[0] = input[0];
                return;
            }
            1 => {
                Self::expand_linear_dim1(input, output);
                return;
            }
            2 => {
                Self::expand_linear_dim2(input, output);
                return;
            }
            3 => {
                Self::expand_linear_dim3(input, output);
                return;
            }
            _ => {}
        }

        // Fill output by expanding one dimension at a time.
        // We treat slices of increasing "arity"

        // Copy the initial evaluations into the start of either
        // tmp or output, depending on parity of dim.
        // We'll alternate between tmp and output as we expand dimensions.
        let (mut cur, mut next) = if dim % 2 == 1 {
            tmp[..input.len()].copy_from_slice(input);
            (tmp, output)
        } else {
            output[..input.len()].copy_from_slice(input);
            (output, tmp)
        };

        let mut in_stride = 1usize;
        let mut out_stride = 1usize;
        let mut blocks = 1 << (dim - 1);

        // sanity checks
        assert_eq!(cur.len(), out_size);
        assert_eq!(next.len(), out_size);
        assert_eq!(input.len(), in_size);

        // start from the smallest subcubes and expand dimension by dimension
        for _ in 0..dim {
            for b in 0..blocks {
                let in_off = b * 2 * in_stride;
                let out_off = b * 3 * out_stride;

                for j in 0..in_stride {
                    // 1d extrapolate
                    let f0 = cur[in_off + j];
                    let f1 = cur[in_off + in_stride + j];
                    next[out_off + j] = f0;
                    next[out_off + out_stride + j] = f1;
                    next[out_off + 2 * out_stride + j] = f1 - f0;
                }
            }
            // swap buffers
            std::mem::swap(&mut cur, &mut next);
            in_stride *= 3;
            out_stride *= 3;
            blocks /= 2;
        }
    }

    #[inline(always)]
    fn expand_linear_dim1(input: &[F], output: &mut [F]) {
        debug_assert_eq!(input.len(), 2);
        debug_assert_eq!(output.len(), 3);

        let f0 = input[0];
        let f1 = input[1];

        output[0] = f0;
        output[1] = f1;
        output[2] = f1 - f0;
    }

    #[inline(always)]
    fn expand_linear_dim2(input: &[F], output: &mut [F]) {
        debug_assert_eq!(input.len(), 4);
        debug_assert_eq!(output.len(), 9);

        let f00 = input[0]; // f(0,0)
        let f01 = input[1]; // f(0,1)
        let f10 = input[2]; // f(1,0)
        let f11 = input[3]; // f(1,1)

        // First extrapolate along the fastest-varying variable (second coordinate).
        let a00 = f00;
        let a01 = f01;
        let a0_inf = f01 - f00;

        let a10 = f10;
        let a11 = f11;
        let a1_inf = f11 - f10;

        // Then extrapolate along the remaining variable.
        let inf0 = a10 - a00;
        let inf1 = a11 - a01;
        let inf_inf = a1_inf - a0_inf;

        // Layout: index = 3 * enc(x0) + enc(x1), x1 fastest, enc: {0,1,∞} -> {0,1,2}.
        output[0] = a00; // (0,0)
        output[1] = a01; // (0,1)
        output[2] = a0_inf; // (0,∞)

        output[3] = a10; // (1,0)
        output[4] = a11; // (1,1)
        output[5] = a1_inf; // (1,∞)

        output[6] = inf0; // (∞,0)
        output[7] = inf1; // (∞,1)
        output[8] = inf_inf; // (∞,∞)
    }

    #[inline(always)]
    fn expand_linear_dim3(input: &[F], output: &mut [F]) {
        debug_assert_eq!(input.len(), 8);
        debug_assert_eq!(output.len(), 27);

        // Corner values f(x0, x1, x2) with x2 fastest.
        let f000 = input[0];
        let f001 = input[1];
        let f010 = input[2];
        let f011 = input[3];
        let f100 = input[4];
        let f101 = input[5];
        let f110 = input[6];
        let f111 = input[7];

        // Stage 1: extrapolate along x2 (fastest variable) for each (x0, x1).
        let g000 = f000;
        let g001 = f001;
        let g00_inf = f001 - f000;

        let g010 = f010;
        let g011 = f011;
        let g01_inf = f011 - f010;

        let g100 = f100;
        let g101 = f101;
        let g10_inf = f101 - f100;

        let g110 = f110;
        let g111 = f111;
        let g11_inf = f111 - f110;

        // Stage 2: extrapolate along x1 for each (x0, x2).
        // x0 = 0
        let h0_0_0 = g000;
        let h0_1_0 = g010;
        let h0_inf_0 = g010 - g000;

        let h0_0_1 = g001;
        let h0_1_1 = g011;
        let h0_inf_1 = g011 - g001;

        let h0_0_inf = g00_inf;
        let h0_1_inf = g01_inf;
        let h0_inf_inf = g01_inf - g00_inf;

        // x0 = 1
        let h1_0_0 = g100;
        let h1_1_0 = g110;
        let h1_inf_0 = g110 - g100;

        let h1_0_1 = g101;
        let h1_1_1 = g111;
        let h1_inf_1 = g111 - g101;

        let h1_0_inf = g10_inf;
        let h1_1_inf = g11_inf;
        let h1_inf_inf = g11_inf - g10_inf;

        // Stage 3: extrapolate along x0 for each (x1, x2).
        // Index: idx(x0, x1, x2) = 9 * enc(x0) + 3 * enc(x1) + enc(x2),
        // enc: {0,1,∞} -> {0,1,2}, x2 fastest.

        // (x1, x2) = (0, 0)
        output[0] = h0_0_0; // (0,0,0)
        output[9] = h1_0_0; // (1,0,0)
        output[18] = h1_0_0 - h0_0_0; // (∞,0,0)

        // (0, 1)
        output[1] = h0_0_1; // (0,0,1)
        output[10] = h1_0_1; // (1,0,1)
        output[19] = h1_0_1 - h0_0_1; // (∞,0,1)

        // (0, ∞)
        output[2] = h0_0_inf; // (0,0,∞)
        output[11] = h1_0_inf; // (1,0,∞)
        output[20] = h1_0_inf - h0_0_inf; // (∞,0,∞)

        // (1, 0)
        output[3] = h0_1_0; // (0,1,0)
        output[12] = h1_1_0; // (1,1,0)
        output[21] = h1_1_0 - h0_1_0; // (∞,1,0)

        // (1, 1)
        output[4] = h0_1_1; // (0,1,1)
        output[13] = h1_1_1; // (1,1,1)
        output[22] = h1_1_1 - h0_1_1; // (∞,1,1)

        // (1, ∞)
        output[5] = h0_1_inf; // (0,1,∞)
        output[14] = h1_1_inf; // (1,1,∞)
        output[23] = h1_1_inf - h0_1_inf; // (∞,1,∞)

        // (∞, 0)
        output[6] = h0_inf_0; // (0,∞,0)
        output[15] = h1_inf_0; // (1,∞,0)
        output[24] = h1_inf_0 - h0_inf_0; // (∞,∞,0)

        // (∞, 1)
        output[7] = h0_inf_1; // (0,∞,1)
        output[16] = h1_inf_1; // (1,∞,1)
        output[25] = h1_inf_1 - h0_inf_1; // (∞,∞,1)

        // (∞, ∞)
        output[8] = h0_inf_inf; // (0,∞,∞)
        output[17] = h1_inf_inf; // (1,∞,∞)
        output[26] = h1_inf_inf - h0_inf_inf; // (∞,∞,∞)
    }

    /// Bind the first (least-significant) variable z_0 := r, reducing the
    /// dimension from w to w-1 and keeping the base-3 layout invariant.
    ///
    /// For each assignment to (z_1, ..., z_{w-1}), we have three stored values
    ///   f(0, ..), f(1, ..), f(∞, ..)
    /// and interpolate the unique quadratic in z_0 that matches them, then
    /// evaluate it at z_0 = r.
    pub fn bind_first_variable(&mut self, r: F::Challenge) {
        let w = self.num_vars;
        debug_assert!(w > 0);

        let new_size = 3_usize.pow((w - 1) as u32);
        let one = F::one();

        let r_term = r * (r - one);
        for new_idx in 0..new_size {
            let old_base_idx = new_idx * 3;
            let eval_at_0 = self.evals[old_base_idx]; // z_0 = 0
            let eval_at_1 = self.evals[old_base_idx + 1]; // z_0 = 1
            let eval_at_inf = self.evals[old_base_idx + 2]; // z_0 = ∞

            self.evals[new_idx] = eval_at_0 * (one - r) + eval_at_1 * r + eval_at_inf * r_term;
        }

        self.num_vars -= 1;
        self.evals.truncate(new_size);
    }

    /// Project t'(z_0, z_1, ..., z_{w-1}) to a univariate in z_0 by summing
    /// against `E_active` over the remaining coordinates.
    ///
    /// The `E_active` table is interpreted identically to the existing outer
    /// Spartan streaming implementation: each index encodes, in binary, which
    /// of z_1..z_{w-1} take the "active" value (mapped to base-3 offset 1).
    /// `first_coord_val` is the z_0 coordinate in {0, 1, 2}, where 2 encodes ∞.
    pub fn project_to_first_variable(&self, E_active: &[F], first_coord_val: usize) -> F {
        let w = self.num_vars;
        debug_assert!(w >= 1);

        let offset = first_coord_val; // z_0 lives at the units place in base-3

        E_active
            .par_iter()
            .enumerate()
            .map(|(eq_active_idx, eq_active_val)| {
                let mut index = offset;
                let mut temp = eq_active_idx;
                let mut power = 3; // start at 3^1 for z_1

                for _ in 0..(w - 1) {
                    if temp & 1 == 1 {
                        index += power;
                    }
                    power *= 3;
                    temp >>= 1;
                }

                self.evals[index] * *eq_active_val
            })
            .sum()
    }
}

impl<F: JoltField> PolynomialBinding<F> for MultiquadraticPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.num_vars == 0 || self.evals.len() == 1
    }

    #[tracing::instrument(skip_all, name = "MultiquadraticPolynomial::bind")]
    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        match order {
            BindingOrder::LowToHigh => self.bind_first_variable(r),
            BindingOrder::HighToLow => {
                // Not currently needed by the outer Spartan streaming code.
                unimplemented!(
                    "HighToLow binding order is not implemented for MultiquadraticPolynomial"
                )
            }
        }
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        // Window sizes are small; fall back to the sequential implementation.
        self.bind(r, order);
    }

    fn final_sumcheck_claim(&self) -> F {
        debug_assert!(self.is_bound());
        debug_assert_eq!(self.evals.len(), 1);
        self.evals[0]
    }
}
