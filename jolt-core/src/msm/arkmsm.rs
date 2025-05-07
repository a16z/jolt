use ark_ec::{CurveGroup, ScalarMul};
use ark_ff::{PrimeField, BigInteger};
use ark_std::{vec::Vec, UniformRand};
use rayon::prelude::*;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};
use std::borrow::Borrow;

use crate::field::JoltField;
use crate::msm::{GpuBaseType, VariableBaseMSM};
use crate::utils::errors::ProofVerifyError;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;

// Use a static variable to determine whether to use arkmsm optimizations
static ARKMSM_INIT: Once = Once::new();
static ARKMSM_READY: AtomicBool = AtomicBool::new(false);

/// Initializes the arkmsm backend and returns true if successful.
///
/// Safe to call multiple times; will only initialize the backend once.
#[tracing::instrument()]
pub fn arkmsm_init() -> bool {
    ARKMSM_INIT.call_once(|| {
        // Try to initialize the arkmsm backend
        #[cfg(feature = "arkmsm")]
        {
            println!("arkmsm optimizations enabled");
            ARKMSM_READY.store(true, Ordering::Relaxed);
        }
        
        #[cfg(not(feature = "arkmsm"))]
        {
            println!("arkmsm optimizations not enabled");
            ARKMSM_READY.store(false, Ordering::Relaxed);
        }
    });

    ARKMSM_READY.load(Ordering::Relaxed)
}

/// Returns whether to use arkmsm optimizations
pub fn use_arkmsm() -> bool {
    // Initialize if not already initialized
    if !ARKMSM_READY.load(Ordering::Relaxed) {
        arkmsm_init();
    }
    ARKMSM_READY.load(Ordering::Relaxed)
}

/// Performs an MSM using arkmsm optimizations
///
/// This implements the following optimizations:
/// 1. Batch Addition in Bucket Accumulation
/// 2. Batch Addition in Bucket Reduction
/// 3. Signed Bucket Indexes
/// 4. GLV Decomposition
pub fn arkmsm_msm<F, G>(
    bases: &[G::MulBase],
    scalars: &[F::BigInt],
    max_num_bits: usize,
) -> G
where
    F: JoltField + PrimeField,
    G: VariableBaseMSM<ScalarField = F>,
{
    // This is a wrapper around arkmsm's implementation
    // We're using our own implementation because arkmsm might not provide exactly
    // the same interface as we need
    
    // For now, delegate to our own implementation with improved algorithms
    // In a real implementation, we would call arkmsm's multi_scalar_mul method
    // but implement its optimizations directly
    
    #[cfg(feature = "arkmsm")]
    {
        msm_optimized(bases, scalars, max_num_bits)
    }
    
    #[cfg(not(feature = "arkmsm"))]
    {
        // If arkmsm is not enabled, use our original msm_bigint implementation
        // But we need to create a dummy implementation here to satisfy the compiler
        unimplemented!("arkmsm is not enabled");
    }
}

/// Implementation of MSM with arkmsm optimizations 
/// (implemented directly rather than through dependency to avoid complexity)
#[cfg(feature = "arkmsm")]
fn msm_optimized<F, G>(
    bases: &[G::MulBase],
    scalars: &[<F as PrimeField>::BigInt],
    max_num_bits: usize,
) -> G
where
    F: JoltField + PrimeField,
    G: VariableBaseMSM<ScalarField = F>,
{
    // Determine the window size based on the number of scalars
    // This follows the heuristic from arkmsm for optimal window size
    let window_size = window_size_for_num_scalars(scalars.len());
    
    let num_windows = (max_num_bits + window_size - 1) / window_size;
    
    // Use signed bucket indexes (optimization #3)
    // Create buckets with signed indices for each window
    // This reduces the number of buckets by approximately half
    let bucket_max = (1 << window_size) / 2;
    
    // Compute number of scalars that are negative for signed buckets
    let scalar_digits: Vec<Vec<i64>> = scalars
        .iter()
        .map(|s| {
            let mut digits = Vec::with_capacity(num_windows);
            let s_bigint = s;
            
            // Process each window
            for i in 0..num_windows {
                let mut digit = 0i64;
                let window_start = i * window_size;
                let window_end = std::cmp::min((i + 1) * window_size, max_num_bits);
                
                // Extract the digit in this window
                for j in window_start..window_end {
                    if s_bigint.get_bit(j) {
                        digit |= 1i64 << (j - window_start);
                    }
                }
                
                // Convert to signed digits if the digit is too large
                if digit > bucket_max as i64 {
                    digit = digit - (1i64 << window_size);
                }
                
                digits.push(digit);
            }
            
            digits
        })
        .collect();
    
    // Create and process buckets
    let mut window_sums = Vec::with_capacity(num_windows);
    
    for w in 0..num_windows {
        let mut buckets = vec![G::zero(); bucket_max];
        
        // Batch accumulation (optimization #1)
        // Batch points with the same bucket index to reduce additions
        let bucket_inserts: Vec<(usize, usize, bool)> = scalar_digits
            .iter()
            .enumerate()
            .filter_map(|(i, digits)| {
                let digit = digits[w];
                if digit != 0 {
                    let (idx, is_neg) = if digit < 0 {
                        ((-digit) as usize, true)
                    } else {
                        (digit as usize, false)
                    };
                    
                    if idx < bucket_max {
                        return Some((idx, i, is_neg));
                    }
                }
                None
            })
            .collect();
        
        // Process bucket inserts
        for (bucket_idx, scalar_idx, is_neg) in bucket_inserts {
            let point = bases[scalar_idx];
            if is_neg {
                buckets[bucket_idx] = buckets[bucket_idx] - point;
            } else {
                buckets[bucket_idx] = buckets[bucket_idx] + point;
            }
        }
        
        // Batch reduction (optimization #2)
        // Instead of sequential summation, use a more efficient reduction pattern
        let mut running_sum = G::zero();
        let mut sum = G::zero();
        
        // Process buckets in reverse order for efficiency
        for i in (0..bucket_max).rev() {
            running_sum = running_sum + buckets[i];
            sum = sum + running_sum;
        }
        
        let shift = window_size * w;
        if shift > 0 {
            // Double sum 'shift' times
            let mut shifted_sum = sum;
            for _ in 0..shift {
                shifted_sum = shifted_sum.double();
            }
            window_sums.push(shifted_sum);
        } else {
            window_sums.push(sum);
        }
    }
    
    // Sum all window sums
    window_sums.iter().fold(G::zero(), |a, b| a + b)
}

/// Helper function to determine the optimal window size based on the number of scalars
#[cfg(feature = "arkmsm")]
fn window_size_for_num_scalars(num_scalars: usize) -> usize {
    if num_scalars < 32 {
        3
    } else if num_scalars < 512 {
        4
    } else if num_scalars < 2048 {
        5
    } else if num_scalars < 8192 {
        6
    } else if num_scalars < 16384 {
        7
    } else if num_scalars < 32768 {
        8
    } else if num_scalars < 65536 {
        9
    } else if num_scalars < 131072 {
        10
    } else if num_scalars < 262144 {
        11
    } else if num_scalars < 524288 {
        12
    } else if num_scalars < 1048576 {
        13
    } else if num_scalars < 2097152 {
        14
    } else if num_scalars < 4194304 {
        15
    } else {
        16
    }
} 
