//! (multilinear) polynomial utlities
use crate::arithmetic::Field;

/// Compute the evaluation of a multilinear polynomial at a given point
/// Uses the lagrange evaluation basis
/// Ref: Section 2.5 of Dory paper.
pub fn compute_polynomial_evaluation<F: Field + Clone>(coeffs: &[F], point: &[F]) -> F {
    let mut eval_vec: Vec<F> = vec![F::zero(); coeffs.len()];

    let expected_size = 1 << point.len();
    assert!(
        coeffs.len() <= expected_size,
        "Too many coefficients: got {}, max for {} variables is {}",
        coeffs.len(),
        point.len(),
        expected_size
    );

    multilinear_lagrange_vec(&mut eval_vec, point); // mutates eval_vec

    // Compute inner product <coeffs, eval_vec> = sum(coeffs[i] * eval_vec[i])
    coeffs
        .iter()
        .zip(eval_vec.iter())
        .map(|(a, b)| a.mul(b))
        .fold(F::zero(), |acc, x| acc.add(&x))
}

/// Computes the evaluation vector for a multilinear polynomial at a given point.
///
/// The evaluation vector contains the values of all 2^n multilinear Lagrange basis functions
/// evaluated at the given point. These basis functions are products of the form:
/// (1-x₁)^b₁ * x₁^(1-b₁) * (1-x₂)^b₂ * x₂^(1-b₂) * ... where each bᵢ ∈ {0,1}
///
/// To evaluate a multilinear polynomial with coefficients `coeffs` at `point`:
/// result = coeffs · evaluation_vector
pub fn multilinear_lagrange_vec<F>(v: &mut [F], point: &[F])
where
    F: Field + Clone,
{
    assert!(
        v.len() <= (1 << point.len()),
        "Vector length must be at most 2^point.len()"
    );

    // empty point means constant polynomial (all basis functions = 1)
    if point.is_empty() || v.is_empty() {
        v.fill(F::one());
        return;
    }

    // Initialize for first variable: basis functions [1-x₀, x₀]
    let one_minus_p0 = F::one().sub(&point[0]);
    v[0] = one_minus_p0;
    if v.len() > 1 {
        v[1] = point[0].clone();
    }

    // For each subsequent variable, double the active portion of the evaluation vector
    // by splitting each existing value into (value * (1-pᵢ)) and (value * pᵢ)
    for (level, p) in point[1..].iter().enumerate() {
        let mid = 1 << (level + 1); // Size of active portion after previous variables

        // Apply the transformation: right[i] = left[i] * p, left[i] = left[i] * (1-p)
        let one_minus_p = F::one().sub(p);

        if mid >= v.len() {
            // No right portion if we've filled the vector, just multiply all by (1-p)
            for li in v.iter_mut() {
                *li = li.clone().mul(&one_minus_p);
            }
        } else {
            // We can split the vector:
            let (left, right) = v.split_at_mut(mid);
            let k = left.len().min(right.len());

            // Transform paired elements
            for (li, ri) in left[..k].iter_mut().zip(right[..k].iter_mut()) {
                *ri = li.clone().mul(p); // New basis function with current variable
                *li = li.clone().mul(&one_minus_p); // Existing basis function without current variable
            }

            // Handle remaining left elements (when left is longer than right)
            for li in left[k..].iter_mut() {
                *li = li.clone().mul(&one_minus_p);
            }
        }
    }
}

/// Compute vectors L and R for matrix-based polynomial evaluation
/// Given a polynomial arranged as a matrix M, computes L and R such that:
/// polynomial_evaluation(b_point) = L^T × M × R
pub fn compute_left_right_vec<F: Field + Clone>(
    b_point: &[F],
    sigma: usize, // log₂(max_columns) - matrix width
    nu: usize,    // log₂(vector_length) - matrix length
) -> (Vec<F>, Vec<F>) {
    let mut right_vec = vec![F::zero(); 1 << nu]; // Column evaluation vector
    let mut left_vec = vec![F::zero(); 1 << nu]; // Row evaluation vector
    let point_dim = b_point.len();

    match point_dim {
        // Case 1: Constant polynomial (0 variables)
        0 => {
            right_vec[0] = F::one();
            left_vec[0] = F::one();
            // Matrix is 1×1, so L^T × M × R = 1 × M[0,0] × 1
        }

        // Case 2: All variables fit in columns (single row needed)
        n if n <= sigma => {
            // All variables determine column position
            multilinear_lagrange_vec(&mut right_vec[..1 << point_dim], b_point);
            left_vec[0] = F::one(); // Only need first row
                                    // L^T × M × R = [1, 0, ...] × M × R
        }

        // Case 3: Variables split between rows and columns (no padding)
        n if n <= sigma * 2 => {
            // Split variables: first `nu` for columns, rest for rows
            multilinear_lagrange_vec(&mut right_vec, &b_point[..nu]); // Column vars
            multilinear_lagrange_vec(&mut left_vec[..1 << (point_dim - nu)], &b_point[nu..]);
            // Row vars
            // L^T × M × R where both L and R have meaningful entries
        }

        // Case 4: Too many variables - need column padding
        _ => {
            // Use max column capacity, put remaining variables in rows
            multilinear_lagrange_vec(&mut right_vec[..(1 << sigma)], &b_point[..sigma]); // First σ vars → columns
            multilinear_lagrange_vec(&mut left_vec, &b_point[sigma..]); // Remaining vars → rows
                                                                        // Matrix has padded columns but we only use the first 2^σ columns
        }
    }

    (left_vec, right_vec)
}

/// Splits evaluation point coordinates into left/right tensors for matrix operations.
/// Outputs can be fed to `multilinear_lagrange_vec` to get the same result as `compute_left_right_vec`.
pub fn compute_l_r_tensors<F: Field + Copy>(
    b_point: &[F],
    sigma: usize,
    nu: usize,
) -> (Vec<F>, Vec<F>) {
    let mut r_coords = vec![F::zero(); nu]; // Column coordinates
    let mut l_coords = vec![F::zero(); nu]; // Row coordinates
    let num_vars = b_point.len();

    match num_vars {
        0 => {}

        n if n <= sigma => {
            // All variables → columns
            r_coords[..n].copy_from_slice(b_point);
        }

        n if n <= sigma * 2 => {
            // Split variables between rows and columns
            r_coords.copy_from_slice(&b_point[..nu]);
            l_coords[..(n - nu)].copy_from_slice(&b_point[nu..]);
        }

        _ => {
            // Too many variables: max columns, rest → rows
            r_coords[..sigma].copy_from_slice(&b_point[..sigma]);
            l_coords.copy_from_slice(&b_point[sigma..]);
        }
    }

    (l_coords, r_coords)
}

/// Computes v = L^T × M in Dory's VMV protocol
/// First step of Vector-Matrix-Vector: L^T * M
pub fn compute_v_vec<F: Field + Clone>(
    a: &[F],        // Polynomial coefficients (flattened matrix M)
    left_vec: &[F], // L vector (row evaluation weights)
    sigma: usize,   // log₂(columns) - matrix width
    nu: usize,      // log₂(rows) - matrix height
) -> Vec<F> {
    let mut v = vec![F::zero(); 1 << nu]; // Result: v = L^T × M

    // Process each row of matrix M
    for (row_idx, row_coeffs) in a.chunks(1 << sigma).enumerate() {
        if row_idx >= left_vec.len() {
            break;
        }

        let l_weight = &left_vec[row_idx]; // Weight for this row

        // Add weighted row to result: v += l_weight * row
        for (col_idx, a_coeff) in row_coeffs.iter().enumerate() {
            if col_idx >= v.len() {
                break;
            }

            let product = l_weight.mul(a_coeff);
            v[col_idx] = v[col_idx].add(&product);
        }
    }

    v
}
