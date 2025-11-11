#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::poly::multilinear_polynomial::MultilinearPolynomial;
    use ark_bn254::Fr;
    use ark_ff::Zero;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_streaming_commitment_equivalence() {
        // Reset globals first
        DoryGlobals::reset();

        // Setup parameters - use smaller values for testing
        let num_vars = 4; // 2^4 = 16 elements
        let num_coeffs = 1 << num_vars;

        // For this test, we'll use row_len=4, num_rows=4 to get 16 total
        let row_len = 4;
        let num_rows = num_coeffs / row_len;

        // Initialize Dory with row_len as K and total size as T
        // This creates a total_size = K * T = 4 * 16 = 64 = 2^6
        let _guard = DoryGlobals::initialize(row_len, num_coeffs);

        // The prover setup needs to match the total size expected by DoryGlobals
        // DoryGlobals with K=4, T=16 creates total_size=64=2^6, so we need 6 vars
        let setup_num_vars = 6;
        let prover_setup = DoryCommitmentScheme::setup_prover(setup_num_vars);

        println!("Test setup: num_vars={}, num_coeffs={}, row_len={}, num_rows={}",
            num_vars, num_coeffs, row_len, num_rows);
        println!("Prover setup g1_vec length: {}", prover_setup.g1_vec.len());
        println!("DoryGlobals num_columns: {}", DoryGlobals::get_num_columns());
        println!("DoryGlobals max_num_rows: {}", DoryGlobals::get_max_num_rows());

        // Create a simple test polynomial with known coefficients
        let coeffs: Vec<Fr> = (0..num_coeffs).map(|i| Fr::from(i as u64)).collect();
        let poly = MultilinearPolynomial::from(coeffs.clone());

        // Method 1: Regular commit
        let (regular_commit, regular_hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        // Method 2: Simulate streaming commit by processing rows
        let mut tier1_commitments = Vec::new();

        // Get G1 bases - use DoryGlobals num_columns for the actual row length
        let actual_row_len = DoryGlobals::get_num_columns();
        let g1_slice = unsafe {
            std::slice::from_raw_parts(
                prover_setup.g1_vec.as_ptr() as *const ArkG1,
                prover_setup.g1_vec.len(),
            )
        };
        let bases: Vec<G1Affine> = g1_slice
            .iter()
            .take(actual_row_len)
            .map(|g| g.0.into_affine())
            .collect();

        // Process each row using actual_row_len chunks
        let actual_num_rows = num_coeffs / actual_row_len;
        for row_idx in 0..actual_num_rows {
            let row_start = row_idx * actual_row_len;
            let row_end = row_start + actual_row_len.min(num_coeffs - row_start);
            let row_coeffs = &coeffs[row_start..row_end];

            // Convert to ark field elements
            let ark_coeffs: Vec<ark_bn254::Fr> = row_coeffs.to_vec();

            // Compute MSM for this row
            let row_commitment =
                ArkG1(VariableBaseMSM::msm_field_elements(&bases[..row_coeffs.len()], &ark_coeffs).unwrap());
            tier1_commitments.push(row_commitment);
        }

        // Compute tier 2 commitment
        let (streaming_commit, streaming_hint) =
            DoryCommitmentScheme::compute_tier_2_commit(&tier1_commitments, &prover_setup);

        // Compare results
        assert_eq!(regular_commit, streaming_commit, "Commitments should match");
        assert_eq!(
            regular_hint.len(),
            streaming_hint.len(),
            "Hints should have same length"
        );

        for (i, (reg, stream)) in regular_hint.iter().zip(streaming_hint.iter()).enumerate() {
            assert_eq!(reg, stream, "Row {} commitment should match", i);
        }
    }

    #[test]
    #[serial]
    fn test_streaming_with_i128_coefficients() {
        // Reset globals first
        DoryGlobals::reset();

        // Test with i128 coefficients (like RdInc polynomial) - use smaller values
        let num_vars = 4;
        let num_coeffs = 1 << num_vars;
        let row_len = 4;
        let num_rows = num_coeffs / row_len;

        let _guard = DoryGlobals::initialize(row_len, num_coeffs);

        // DoryGlobals with K=4, T=16 creates total_size=64=2^6, so we need 6 vars
        let setup_num_vars = 6;
        let prover_setup = DoryCommitmentScheme::setup_prover(setup_num_vars);

        // Create i128 coefficients
        let coeffs_i128: Vec<i128> = (0..num_coeffs)
            .map(|i| (i as i128) - 500) // Some negative and positive values
            .collect();

        // Convert to field elements for regular commit
        let coeffs_field: Vec<Fr> = coeffs_i128
            .iter()
            .map(|&c| {
                if c >= 0 {
                    Fr::from(c as u128)
                } else {
                    -Fr::from((-c) as u128)
                }
            })
            .collect();
        let poly = MultilinearPolynomial::from(coeffs_field);

        let (regular_commit, regular_hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        // Streaming commit using i128 MSM directly
        let mut tier1_commitments = Vec::new();

        let actual_row_len = DoryGlobals::get_num_columns();
        let g1_slice = unsafe {
            std::slice::from_raw_parts(
                prover_setup.g1_vec.as_ptr() as *const ArkG1,
                prover_setup.g1_vec.len(),
            )
        };
        let bases: Vec<G1Affine> = g1_slice
            .iter()
            .take(actual_row_len)
            .map(|g| g.0.into_affine())
            .collect();

        let actual_num_rows = num_coeffs / actual_row_len;
        for row_idx in 0..actual_num_rows {
            let row_start = row_idx * actual_row_len;
            let row_end = row_start + actual_row_len.min(num_coeffs - row_start);
            let row_coeffs = &coeffs_i128[row_start..row_end];

            // Use i128 MSM directly
            let row_commitment = ArkG1(VariableBaseMSM::msm_i128(&bases[..row_coeffs.len()], row_coeffs).unwrap());
            tier1_commitments.push(row_commitment);
        }

        let (streaming_commit, streaming_hint) =
            DoryCommitmentScheme::compute_tier_2_commit(&tier1_commitments, &prover_setup);

        assert_eq!(
            regular_commit, streaming_commit,
            "i128 commitments should match"
        );
        assert_eq!(
            regular_hint.len(),
            streaming_hint.len(),
            "i128 hints should have same length"
        );
    }

    #[test]
    #[serial]
    fn test_one_hot_polynomial_commitment() {
        // Reset globals first
        DoryGlobals::reset();

        // Test with one-hot polynomial (like InstructionRa) - use smaller values
        let num_vars = 4;
        let num_coeffs = 1 << num_vars;
        let k_chunk = 4; // Number of possible bases for one-hot
        let row_len = 4;
        let num_rows = num_coeffs / row_len;

        let _guard = DoryGlobals::initialize(row_len, num_coeffs);

        // DoryGlobals with K=4, T=16 creates total_size=64=2^6, so we need 6 vars
        let setup_num_vars = 6;
        let prover_setup = DoryCommitmentScheme::setup_prover(setup_num_vars);

        // Create one-hot indices (each element selects a base index)
        let indices: Vec<usize> = (0..num_coeffs).map(|i| i % k_chunk).collect();

        // Create a OneHot polynomial using the correct approach
        use crate::poly::one_hot_polynomial::OneHotPolynomial;
        let indices_opt: Vec<Option<u8>> = indices.iter().map(|&idx| Some(idx as u8)).collect();
        let one_hot_poly = OneHotPolynomial::from_indices(indices_opt, k_chunk);
        let poly = MultilinearPolynomial::OneHot(one_hot_poly);

        let (regular_commit, _regular_hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        // Streaming commit using base addition
        let actual_row_len = DoryGlobals::get_num_columns();
        let g1_slice = unsafe {
            std::slice::from_raw_parts(
                prover_setup.g1_vec.as_ptr() as *const ArkG1,
                prover_setup.g1_vec.len(),
            )
        };
        let bases: Vec<G1Affine> = g1_slice
            .iter()
            .take(actual_row_len)
            .map(|g| g.0.into_affine())
            .collect();

        // Process like one_hot_polynomial::commit_rows does
        let actual_num_rows = (num_coeffs as u128 * k_chunk as u128 / actual_row_len as u128) as usize;
        let rows_per_k = num_coeffs / actual_row_len;

        let mut tier1_commitments = vec![ArkG1(G1Projective::zero()); actual_num_rows];

        // Process chunks of row_len
        for (chunk_index, chunk) in indices.chunks(actual_row_len).enumerate() {
            // Collect indices for each k
            let mut indices_per_k: Vec<Vec<usize>> = vec![Vec::new(); k_chunk];

            for (col_index, &k) in chunk.iter().enumerate() {
                indices_per_k[k].push(col_index);
            }

            // Use batch addition for all k values at once
            let results = jolt_optimizations::batch_g1_additions_multi(
                &bases,
                &indices_per_k
            );

            // Place results in the correct positions
            for (k, result) in results.into_iter().enumerate() {
                if !indices_per_k[k].is_empty() {
                    let row_idx = chunk_index + k * rows_per_k;
                    if row_idx < actual_num_rows {
                        tier1_commitments[row_idx] = ArkG1(G1Projective::from(result));
                    }
                }
            }
        }

        let (streaming_commit, _streaming_hint) =
            DoryCommitmentScheme::compute_tier_2_commit(&tier1_commitments, &prover_setup);

        // For one-hot, the commitments should match exactly
        assert_eq!(
            regular_commit, streaming_commit,
            "One-hot commitments should match"
        );
    }
}
