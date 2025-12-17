//! Comprehensive tests for the generic jagged bijection transform

use crate::poly::dense_mlpoly::DensePolynomial;
use crate::zkvm::recursion::bijection::*;
use crate::zkvm::recursion::constraints_sys::{
    ConstraintSystem, ConstraintType, DoryMatrixBuilder, MatrixConstraint, PolyType,
};
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        commitment::{
            dory::{DoryCommitmentScheme, DoryGlobals},
            commitment_scheme::CommitmentScheme,
        },
    },
    zkvm::recursion::RecursionProver,
    transcripts::{Blake2bTranscript, Transcript},
};
use ark_ff::UniformRand;
use ark_std::test_rng;
use ark_bn254::Fr;
use serial_test::serial;
use dory::backends::arkworks::ArkGT;

/// Helper to create a test constraint system with mixed constraint types
fn create_mixed_constraint_system() -> ConstraintSystem {
    // For testing, we'll create a simple constraint system without real witnesses
    let constraints = vec![
        // GT exp constraints (4-var)
        MatrixConstraint {
            constraint_index: 0,
            constraint_type: ConstraintType::GtExp { bit: true },
        },
        MatrixConstraint {
            constraint_index: 1,
            constraint_type: ConstraintType::GtExp { bit: false },
        },
        MatrixConstraint {
            constraint_index: 2,
            constraint_type: ConstraintType::GtExp { bit: true },
        },
        // GT mul constraints (4-var)
        MatrixConstraint {
            constraint_index: 3,
            constraint_type: ConstraintType::GtMul,
        },
        MatrixConstraint {
            constraint_index: 4,
            constraint_type: ConstraintType::GtMul,
        },
        // G1 scalar mul constraint (8-var)
        MatrixConstraint {
            constraint_index: 5,
            constraint_type: ConstraintType::G1ScalarMul {
                base_point: (Fq::one(), Fq::one()),
            },
        },
    ];

    // Create a dummy g polynomial
    let g_poly = DensePolynomial::new(vec![Fq::one(); 256]);

    // Build constraint system
    ConstraintSystem::from_witness(
        constraints.into_iter().map(|c| c.constraint_type).collect(),
        g_poly,
    ).unwrap()
}

#[test]
fn test_bijection_with_constraint_system() {
    let cs = create_mixed_constraint_system();
    let builder = ConstraintSystemJaggedBuilder::from_constraints(&cs.constraints);
    let (bijection, mapping) = builder.build();

    // We should have:
    // - 3 GT exp constraints × 4 poly types = 12 polynomials (4-var each)
    // - 2 GT mul constraints × 4 poly types = 8 polynomials (4-var each)
    // - 1 G1 scalar mul × 7 poly types = 7 polynomials (8-var each)
    // Total: 27 polynomials
    assert_eq!(bijection.num_polynomials(), 27);
    assert_eq!(mapping.num_polynomials(), 27);

    // Check dense size:
    // 20 polynomials × 16 (4-var) + 7 polynomials × 256 (8-var) = 320 + 1792 = 2112
    assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection), 20 * 16 + 7 * 256);

    // Test some specific mappings
    let (c_idx, p_type) = mapping.decode(0);
    assert_eq!(c_idx, 0);
    assert_eq!(p_type as usize, PolyType::Base as usize);

    let (c_idx, p_type) = mapping.decode(12); // First GT mul poly
    assert_eq!(c_idx, 3); // 4th constraint (0-indexed)
    assert_eq!(p_type as usize, PolyType::MulLhs as usize);

    let (c_idx, p_type) = mapping.decode(20); // First G1 scalar mul poly
    assert_eq!(c_idx, 5); // 6th constraint
    assert_eq!(p_type as usize, PolyType::G1ScalarMulXA as usize);
}

#[test]
fn test_compression_ratio() {
    let cs = create_mixed_constraint_system();

    // Calculate padded sparse size (everything at 8 vars)
    let num_rows = cs.matrix.num_rows;
    let evaluations_per_row = 1 << cs.matrix.num_constraint_vars; // 256
    let sparse_padded_size = num_rows * evaluations_per_row;

    // Build dense polynomial
    let (dense_poly, bijection) = cs.build_dense_polynomial();
    let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection);

    // Calculate compression ratio
    let compression_ratio = 1.0 - (dense_size as f64 / sparse_padded_size as f64);

    println!("Sparse padded size: {}", sparse_padded_size);
    println!("Dense size: {}", dense_size);
    println!("Compression ratio: {:.2}%", compression_ratio * 100.0);

    // With mixed 4-var and 8-var constraints, we should achieve significant compression
    // GT operations are padded 16x, so we save ~93% on those
    assert!(
        compression_ratio > 0.5,
        "Expected >50% compression, got {:.2}%",
        compression_ratio * 100.0
    );
}

#[test]
fn test_dense_polynomial_extraction() {
    let cs = create_mixed_constraint_system();
    let (dense_poly, bijection) = cs.build_dense_polynomial();

    // Dense polynomial should be padded to power of 2
    assert_eq!(dense_poly.len(), dense_poly.len().next_power_of_two());

    // Verify some extracted values match original matrix
    let builder = ConstraintSystemJaggedBuilder::from_constraints(&cs.constraints);
    let (_, mapping) = builder.build();

    // Check first few values
    for dense_idx in 0..10 {
        let poly_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection,dense_idx);
        let eval_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection,dense_idx);

        let (constraint_idx, poly_type) = mapping.decode(poly_idx);
        let matrix_row = cs.matrix.row_index(poly_type, constraint_idx);
        let offset = cs.matrix.storage_offset(matrix_row);

        let expected_val = cs.matrix.evaluations[offset + eval_idx];
        let actual_val = dense_poly.Z[dense_idx];

        assert_eq!(
            actual_val, expected_val,
            "Mismatch at dense_idx {}, poly_idx {}, eval_idx {}",
            dense_idx, poly_idx, eval_idx
        );
    }
}

#[test]
fn test_bijection_boundary_cases() {
    // Empty case
    let empty_bijection = VarCountJaggedBijection::new(vec![]);
    assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&empty_bijection), 0);
    assert_eq!(empty_bijection.num_polynomials(), 0);
    assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::sparse_to_dense(&empty_bijection, 0, 0), None);

    // Single polynomial
    let single = VarCountJaggedBijection::new(vec![JaggedPolynomial::new(3)]);
    assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&single), 8);
    assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::row(&single, 0), 0);
    assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::row(&single, 7), 0);
    assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::sparse_to_dense(&single, 0, 7), Some(7));
    assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::sparse_to_dense(&single, 0, 8), None);
    assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::sparse_to_dense(&single, 1, 0), None);

    // Large polynomial
    let large = VarCountJaggedBijection::new(vec![JaggedPolynomial::new(10)]);
    assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&large), 1024);
    assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::row(&large, 1023), 0);
    assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::col(&large, 1023), 1023);
}

#[test]
fn test_mixed_variable_counts() {
    // Test with various variable counts
    let polys = vec![
        JaggedPolynomial::new(2), // 4 evaluations
        JaggedPolynomial::new(3), // 8 evaluations
        JaggedPolynomial::new(4), // 16 evaluations
        JaggedPolynomial::new(5), // 32 evaluations
        JaggedPolynomial::new(6), // 64 evaluations
        JaggedPolynomial::new(7), // 128 evaluations
        JaggedPolynomial::new(8), // 256 evaluations
    ];

    let bijection = VarCountJaggedBijection::new(polys);

    let expected_size = 4 + 8 + 16 + 32 + 64 + 128 + 256;
    assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection), expected_size);

    // Test cumulative indexing
    let mut cumulative = 0;
    for (i, size) in [4, 8, 16, 32, 64, 128, 256].iter().enumerate() {
        // First index of this polynomial
        assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection,cumulative), i);
        assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection,cumulative), 0);

        // Last index of this polynomial
        assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection,cumulative + size - 1), i);
        assert_eq!(<VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection,cumulative + size - 1), size - 1);

        cumulative += size;
    }
}

#[test]
fn test_constraint_mapping_consistency() {
    let cs = create_mixed_constraint_system();
    let builder = ConstraintSystemJaggedBuilder::from_constraints(&cs.constraints);
    let (bijection, mapping) = builder.build();

    // Verify every polynomial maps to a valid constraint
    for poly_idx in 0..mapping.num_polynomials() {
        let (constraint_idx, poly_type) = mapping.decode(poly_idx);

        // Constraint index should be valid
        assert!(
            constraint_idx < cs.constraints.len(),
            "Invalid constraint index {} for poly {}",
            constraint_idx,
            poly_idx
        );

        // Polynomial type should match constraint type
        let constraint = &cs.constraints[constraint_idx];
        match constraint.constraint_type {
            ConstraintType::GtExp { .. } => {
                assert!(
                    matches!(
                        poly_type,
                        PolyType::Base | PolyType::RhoPrev | PolyType::RhoCurr | PolyType::Quotient
                    ),
                    "Invalid poly type {:?} for GT exp constraint",
                    poly_type
                );
            }
            ConstraintType::GtMul => {
                assert!(
                    matches!(
                        poly_type,
                        PolyType::MulLhs
                            | PolyType::MulRhs
                            | PolyType::MulResult
                            | PolyType::MulQuotient
                    ),
                    "Invalid poly type {:?} for GT mul constraint",
                    poly_type
                );
            }
            ConstraintType::G1ScalarMul { .. } => {
                assert!(
                    matches!(
                        poly_type,
                        PolyType::G1ScalarMulXA
                            | PolyType::G1ScalarMulYA
                            | PolyType::G1ScalarMulXT
                            | PolyType::G1ScalarMulYT
                            | PolyType::G1ScalarMulXANext
                            | PolyType::G1ScalarMulYANext
                            | PolyType::G1ScalarMulIndicator
                    ),
                    "Invalid poly type {:?} for G1 scalar mul constraint",
                    poly_type
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "Dense index 100 out of bounds")]
fn test_out_of_bounds_access() {
    let bijection = VarCountJaggedBijection::new(vec![
        JaggedPolynomial::new(3), // 8 evaluations
        JaggedPolynomial::new(4), // 16 evaluations
    ]);

    // Total size is 24, so 100 should panic
    <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection,100);
}

#[test]
fn test_jagged_bijection_with_real_dory_proof() {
    use crate::poly::{
        commitment::{
            commitment_scheme::{CommitmentScheme, RecursionExt},
            dory::{DoryCommitmentScheme, DoryGlobals},
        },
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    };
    use crate::zkvm::recursion::ConstraintSystem;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;

    // Initialize Dory globals
    DoryGlobals::reset();
    DoryGlobals::initialize(1 << 2, 1 << 2);

    let num_vars = 4;
    let mut rng = StdRng::seed_from_u64(42);

    // Create a test polynomial and commitment
    let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
    let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

    let coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));
    let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

    // Create evaluation point and proof
    let point: Vec<<Fr as crate::field::JoltField>::Challenge> = (0..num_vars)
        .map(|_| <Fr as crate::field::JoltField>::Challenge::random(&mut rng))
        .collect();

    let mut prover_transcript = crate::transcripts::Blake2bTranscript::new(b"test_jagged");
    let proof = DoryCommitmentScheme::prove(
        &prover_setup,
        &poly,
        &point,
        Some(hint),
        &mut prover_transcript,
    );

    let evaluation = PolynomialEvaluation::evaluate(&poly, &point);
    let mut extract_transcript = crate::transcripts::Blake2bTranscript::new(b"test_jagged");

    // Create constraint system from Dory proof
    let (system, _hints) = ConstraintSystem::new(
        &proof,
        &verifier_setup,
        &mut extract_transcript,
        &point,
        &evaluation,
        &commitment,
    )
    .expect("System creation should succeed");

    // Now apply the jagged bijection to create dense polynomial
    let (dense_poly, bijection) = system.build_dense_polynomial();

    // Verify the bijection properties
    println!("Number of constraints: {}", system.constraints.len());
    println!("Number of polynomials: {}", bijection.num_polynomials());
    println!("Dense size: {}", <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection));
    println!("Dense poly padded size: {}", dense_poly.len());

    // Count constraint types
    let mut gt_exp_count = 0;
    let mut gt_mul_count = 0;
    let mut g1_scalar_mul_count = 0;

    for constraint in &system.constraints {
        match &constraint.constraint_type {
            ConstraintType::GtExp { .. } => gt_exp_count += 1,
            ConstraintType::GtMul => gt_mul_count += 1,
            ConstraintType::G1ScalarMul { .. } => g1_scalar_mul_count += 1,
        }
    }

    println!("  GT exp: {}", gt_exp_count);
    println!("  GT mul: {}", gt_mul_count);
    println!("  G1 scalar mul: {}", g1_scalar_mul_count);

    // Calculate compression ratio
    let sparse_padded_size = system.matrix.num_rows * (1 << system.matrix.num_constraint_vars);
    let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection);
    let compression_ratio = 1.0 - (dense_size as f64 / sparse_padded_size as f64);

    println!(
        "Compression ratio: {:.2}% (sparse: {}, dense: {})",
        compression_ratio * 100.0,
        sparse_padded_size,
        dense_size
    );

    // Verify compression is significant when we have GT operations
    if gt_exp_count > 0 || gt_mul_count > 0 {
        assert!(
            compression_ratio > 0.3,
            "Expected significant compression with GT operations, got {:.2}%",
            compression_ratio * 100.0
        );
    }

    // Test bijection correctness
    for dense_idx in 0..<VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection).min(100) {
        let row = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection,dense_idx);
        let col = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection,dense_idx);

        // Verify round trip
        let reconstructed = <VarCountJaggedBijection as JaggedTransform<Fq>>::sparse_to_dense(&bijection, row, col)
            .expect("Should map back");
        assert_eq!(reconstructed, dense_idx);

        // Verify polynomial has expected variable count
        let num_vars = <VarCountJaggedBijection as JaggedTransform<Fq>>::poly_num_vars(&bijection, row);
        assert!(
            num_vars == 4 || num_vars == 8,
            "Expected 4 or 8 variables, got {}",
            num_vars
        );
    }

    // Verify the dense polynomial values match the sparse matrix
    let builder = ConstraintSystemJaggedBuilder::from_constraints(&system.constraints);
    let (_, mapping) = builder.build();

    for i in 0..10.min(<VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection)) {
        let poly_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection,i);
        let eval_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection,i);

        let (constraint_idx, poly_type) = mapping.decode(poly_idx);
        let matrix_row = system.matrix.row_index(poly_type, constraint_idx);
        let offset = system.matrix.storage_offset(matrix_row);

        // Get the number of variables for this polynomial to handle padding correctly
        let num_vars = <VarCountJaggedBijection as JaggedTransform<Fq>>::poly_num_vars(&bijection, poly_idx);

        // For 4-var polynomials, each value is repeated 16 times consecutively in the sparse matrix
        let sparse_idx = if num_vars == 4 {
            eval_idx * 16  // Skip to the start of this evaluation's repeated block
        } else {
            eval_idx       // 8-var polynomials: direct indexing
        };

        let expected_val = system.matrix.evaluations[offset + sparse_idx];
        let actual_val = dense_poly.Z[i];

        assert_eq!(
            actual_val, expected_val,
            "Value mismatch at dense index {}",
            i
        );
    }
}

/// Create a test constraint system with known polynomial values
fn create_test_constraint_system_with_values() -> ConstraintSystem {
    let constraints = vec![
        // GT exp constraint (4-var)
        MatrixConstraint {
            constraint_index: 0,
            constraint_type: ConstraintType::GtExp { bit: true },
        },
        // GT mul constraint (4-var)
        MatrixConstraint {
            constraint_index: 1,
            constraint_type: ConstraintType::GtMul,
        },
        // G1 scalar mul constraint (8-var)
        MatrixConstraint {
            constraint_index: 2,
            constraint_type: ConstraintType::G1ScalarMul {
                base_point: (Fq::one(), Fq::one()),
            },
        },
    ];

    // Create a g polynomial with known values
    let mut g_values = vec![];
    for i in 0..256 {
        g_values.push(Fq::from((i + 1) as u64));
    }
    let g_poly = DensePolynomial::new(g_values);

    ConstraintSystem::from_witness(
        constraints.into_iter().map(|c| c.constraint_type).collect(),
        g_poly,
    )
    .unwrap()
}

#[test]
fn test_jagged_relation_with_constraint_system() {
    let constraint_system = create_test_constraint_system_with_values();

    // Build dense polynomial
    let (dense_poly, jagged_bijection) = constraint_system.build_dense_polynomial();

    // Get dimensions
    let num_s_vars = constraint_system.num_s_vars();
    let num_x_vars = constraint_system.matrix.num_constraint_vars;

    println!("Test setup:");
    println!("  - num_s_vars: {}", num_s_vars);
    println!("  - num_x_vars: {}", num_x_vars);
    println!("  - matrix evaluations len: {}", constraint_system.matrix.evaluations.len());
    println!("  - dense size: {}", <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&jagged_bijection));

    // Test with random evaluation points
    let mut rng = test_rng();
    let zr: Vec<Fq> = (0..num_s_vars).map(|_| Fq::rand(&mut rng)).collect();
    let zc: Vec<Fq> = (0..num_x_vars).map(|_| Fq::rand(&mut rng)).collect();

    // Method 1: Evaluate sparse polynomial directly
    let sparse_mlpoly = MultilinearPolynomial::from(constraint_system.matrix.evaluations.clone());
    let mut eval_point = Vec::new();
    eval_point.extend_from_slice(&zc);
    eval_point.extend_from_slice(&zr);
    let eval_challenges: Vec<<Fq as JoltField>::Challenge> = eval_point
        .iter()
        .rev()
        .map(|&x| x.into())
        .collect();
    let sparse_eval = PolynomialEvaluation::evaluate(&sparse_mlpoly, &eval_challenges);

    // Method 2: Compute using dense polynomial and bijection
    let eq_row_evals = EqPolynomial::<Fq>::evals(&zr);
    let eq_col_evals = EqPolynomial::<Fq>::evals(&zc);

    let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&jagged_bijection);
    let mut dense_eval = Fq::zero();

    for i in 0..dense_size {
        let row = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&jagged_bijection, i);
        let col = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&jagged_bijection, i);
        let eq_row = eq_row_evals[row];
        let eq_col = eq_col_evals[col];
        let q_val = dense_poly.Z[i];
        dense_eval += q_val * eq_row * eq_col;
    }

    println!("\nEvaluations:");
    println!("  - Sparse eval: {}", sparse_eval);
    println!("  - Dense eval:  {}", dense_eval);

    // Debug: Print matrix layout info
    println!("\nMatrix layout check:");
    println!("  - Polynomial ordering in bijection:");

    let builder = ConstraintSystemJaggedBuilder::from_constraints(&constraint_system.constraints);
    for (i, (c_idx, p_type, num_vars)) in builder.polynomials.iter().enumerate().take(10) {
        println!("    [{}] constraint_idx={}, poly_type={:?}, num_vars={}",
            i, c_idx, p_type, num_vars);

        // Also check what row this maps to in the matrix
        let matrix_row = constraint_system.matrix.row_index(*p_type, *c_idx);
        println!("         -> matrix row: {}", matrix_row);
    }

    assert_eq!(
        sparse_eval, dense_eval,
        "Sparse and dense evaluations should match!"
    );
}

/// Create a constraint system with real diverse values to properly test isomorphism
fn create_constraint_system_with_real_values() -> ConstraintSystem {
    let constraints = vec![
        // GT exp constraint (4-var)
        MatrixConstraint {
            constraint_index: 0,
            constraint_type: ConstraintType::GtExp { bit: true },
        },
        // GT mul constraint (4-var)
        MatrixConstraint {
            constraint_index: 1,
            constraint_type: ConstraintType::GtMul,
        },
        // G1 scalar mul constraint (8-var)
        MatrixConstraint {
            constraint_index: 2,
            constraint_type: ConstraintType::G1ScalarMul {
                base_point: (Fq::one(), Fq::one()),
            },
        },
    ];

    // Create a g polynomial with diverse values that will expose padding issues
    let mut g_values = vec![];
    let mut rng = test_rng();
    for i in 0..256 {
        // Use random values so we can detect if padding/extraction is wrong
        g_values.push(Fq::rand(&mut rng));
    }
    let g_poly = DensePolynomial::new(g_values);

    ConstraintSystem::from_witness(
        constraints.into_iter().map(|c| c.constraint_type).collect(),
        g_poly,
    )
    .unwrap()
}

#[test]
fn test_sparse_dense_isomorphism_value_by_value() {
    let constraint_system = create_constraint_system_with_real_values();

    // Build dense polynomial
    let (dense_poly, jagged_bijection) = constraint_system.build_dense_polynomial();

    println!("Testing sparse/dense isomorphism value by value...");

    // Build the mapping to understand which constraint/poly type each polynomial corresponds to
    let builder = ConstraintSystemJaggedBuilder::from_constraints(&constraint_system.constraints);
    let (_, mapping) = builder.build();

    // Check EVERY value in the dense polynomial matches the correct value from sparse
    let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&jagged_bijection);
    let mut values_checked = 0;

    for dense_idx in 0..dense_size {
        let poly_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&jagged_bijection, dense_idx);
        let eval_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&jagged_bijection, dense_idx);

        let (constraint_idx, poly_type) = mapping.decode(poly_idx);
        let num_vars = <VarCountJaggedBijection as JaggedTransform<Fq>>::poly_num_vars(&jagged_bijection, poly_idx);

        // Get the expected value from the sparse matrix
        let matrix_row = constraint_system.matrix.row_index(poly_type, constraint_idx);
        let storage_offset = constraint_system.matrix.storage_offset(matrix_row);

        let sparse_idx = if num_vars == 4 {
            // For 4-var: value at eval_idx is repeated starting at position eval_idx * 16
            eval_idx * 16
        } else {
            // For 8-var: direct indexing
            eval_idx
        };

        let expected_val = constraint_system.matrix.evaluations[storage_offset + sparse_idx];
        let actual_val = dense_poly.Z[dense_idx];

        assert_eq!(
            actual_val, expected_val,
            "Value mismatch at dense[{}]: poly[{}] eval[{}] (constraint {} type {:?} {}var)\nExpected: {}\nActual: {}",
            dense_idx, poly_idx, eval_idx, constraint_idx, poly_type, num_vars, expected_val, actual_val
        );

        values_checked += 1;

        // Also verify the padding pattern for 4-var polynomials
        if num_vars == 4 && eval_idx == 0 {
            // Check that the next 15 positions in sparse have the same value (repetition padding)
            for offset in 1..16 {
                let padded_val = constraint_system.matrix.evaluations[storage_offset + sparse_idx + offset];
                assert_eq!(
                    padded_val, expected_val,
                    "Padding check failed: position {} should equal position {}",
                    sparse_idx + offset, sparse_idx
                );
            }
        }
    }

    println!("✓ Successfully verified {} dense values match their sparse counterparts", values_checked);

    // Now verify the jagged relation holds with these real values
    let num_s_vars = constraint_system.num_s_vars();
    let num_x_vars = constraint_system.matrix.num_constraint_vars;

    let mut rng = test_rng();
    let zr: Vec<Fq> = (0..num_s_vars).map(|_| Fq::rand(&mut rng)).collect();
    let zc: Vec<Fq> = (0..num_x_vars).map(|_| Fq::rand(&mut rng)).collect();

    // Evaluate sparse
    let sparse_mlpoly = MultilinearPolynomial::from(constraint_system.matrix.evaluations.clone());
    let mut eval_point = Vec::new();
    eval_point.extend_from_slice(&zc);
    eval_point.extend_from_slice(&zr);
    let eval_challenges: Vec<<Fq as JoltField>::Challenge> = eval_point
        .iter()
        .rev()
        .map(|&x| x.into())
        .collect();
    let sparse_eval = PolynomialEvaluation::evaluate(&sparse_mlpoly, &eval_challenges);

    // Evaluate dense
    let eq_row_evals = EqPolynomial::<Fq>::evals(&zr);
    let eq_col_evals = EqPolynomial::<Fq>::evals(&zc);

    let mut dense_eval = Fq::zero();
    for i in 0..dense_size {
        let row = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&jagged_bijection, i);
        let col = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&jagged_bijection, i);
        let eq_row = eq_row_evals[row];
        let eq_col = eq_col_evals[col];
        let q_val = dense_poly.Z[i];
        dense_eval += q_val * eq_row * eq_col;
    }

    println!("\nJagged relation test with real values:");
    println!("  Sparse eval: {}", sparse_eval);
    println!("  Dense eval:  {}", dense_eval);

    assert_eq!(
        sparse_eval, dense_eval,
        "Jagged relation failed with real values!"
    );
}

#[test]
#[serial]
fn test_sparse_dense_bijection_with_real_dory_witness() {
    use crate::poly::commitment::dory::wrappers::ArkDoryProof;

    // Initialize Dory
    DoryGlobals::reset();
    DoryGlobals::initialize(1 << 2, 1 << 2);

    let mut rng = test_rng();

    // Create a real Dory proof
    let num_vars = 4;
    let poly_coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly_coefficients));

    let prover_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover(num_vars);
    let verifier_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_verifier(&prover_setup);

    let (commitment, hint) =
        <DoryCommitmentScheme as CommitmentScheme>::commit(&poly, &prover_setup);

    let mut point_transcript: Blake2bTranscript = Transcript::new(b"test_point");
    let point_challenges: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
        .map(|_| point_transcript.challenge_scalar_optimized::<Fr>())
        .collect();

    let evaluation = PolynomialEvaluation::evaluate(&poly, &point_challenges);

    let mut prover_transcript: Blake2bTranscript = Transcript::new(b"dory_test_proof");
    let opening_proof = <DoryCommitmentScheme as CommitmentScheme>::prove(
        &prover_setup,
        &poly,
        &point_challenges,
        Some(hint),
        &mut prover_transcript,
    );

    // Create RecursionProver from Dory proof
    let gamma = Fq::rand(&mut rng);
    let delta = Fq::rand(&mut rng);

    let mut witness_transcript: Blake2bTranscript = Transcript::new(b"dory_test_proof");

    let ark_proof = ArkDoryProof::from(opening_proof);
    let ark_commitment = ArkGT::from(commitment);

    let prover = RecursionProver::<Fq>::new_from_dory_proof(
        &ark_proof,
        &verifier_setup,
        &mut witness_transcript,
        &point_challenges,
        &evaluation,
        &ark_commitment,
        gamma,
        delta,
    )
    .expect("Failed to create recursion prover");

    // Get the constraint system and build dense polynomial
    let constraint_system = &prover.constraint_system;
    let (dense_poly, jagged_bijection) = constraint_system.build_dense_polynomial();

    let num_s_vars = constraint_system.num_s_vars();
    let num_x_vars = constraint_system.matrix.num_constraint_vars;
    let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&jagged_bijection);

    println!("Testing bijection with real Dory witness:");
    println!("  - Sparse matrix size: {}", constraint_system.matrix.evaluations.len());
    println!("  - Dense size: {}", dense_size);
    println!("  - num_s_vars: {}", num_s_vars);
    println!("  - num_x_vars: {}", num_x_vars);

    // Build the mapping to understand constraint types
    let builder = ConstraintSystemJaggedBuilder::from_constraints(&constraint_system.constraints);
    let polynomials_info = builder.polynomials.clone();
    let (_, mapping) = builder.build();

    // Test 1: For every dense index, verify we can recover the value from sparse
    println!("\nTest 1: Dense → Sparse mapping");
    let sample_size = 100.min(dense_size);
    for dense_idx in (0..dense_size).step_by(dense_size / sample_size) {
        let row = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&jagged_bijection, dense_idx);
        let col = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&jagged_bijection, dense_idx);

        let (constraint_idx, poly_type) = mapping.decode(row);
        let num_vars = <VarCountJaggedBijection as JaggedTransform<Fq>>::poly_num_vars(&jagged_bijection, row);

        // Get value from sparse matrix
        let matrix_row = constraint_system.matrix.row_index(poly_type, constraint_idx);
        let storage_offset = constraint_system.matrix.storage_offset(matrix_row);

        let sparse_idx = if num_vars == 4 {
            col * 16 // For 4-var padded to 8-var
        } else {
            col // For 8-var
        };

        let sparse_value = constraint_system.matrix.evaluations[storage_offset + sparse_idx];
        let dense_value = dense_poly.Z[dense_idx];

        assert_eq!(
            sparse_value, dense_value,
            "Bijection failed at dense[{}] → sparse[{}] (poly {} type {:?})",
            dense_idx, storage_offset + sparse_idx, row, poly_type
        );
    }
    println!("✓ Successfully verified {} dense → sparse mappings", sample_size);

    // Test 2: For sparse positions with unique values, verify they exist in dense
    println!("\nTest 2: Sparse → Dense mapping (unique values only)");
    let mut verified_sparse_to_dense = 0;

    // Sample some polynomial types and constraints
    for constraint_idx in 0..10.min(constraint_system.constraints.len()) {
        let constraint_type = &constraint_system.constraints[constraint_idx].constraint_type;

        let poly_types = match constraint_type {
            ConstraintType::GtExp { .. } => vec![
                PolyType::Base,
                PolyType::RhoPrev,
                PolyType::RhoCurr,
                PolyType::Quotient,
            ],
            ConstraintType::GtMul => vec![
                PolyType::MulLhs,
                PolyType::MulRhs,
                PolyType::MulResult,
                PolyType::MulQuotient,
            ],
            ConstraintType::G1ScalarMul { .. } => vec![
                PolyType::G1ScalarMulXA,
                PolyType::G1ScalarMulYA,
            ],
        };

        for poly_type in poly_types {
            let matrix_row = constraint_system.matrix.row_index(poly_type, constraint_idx);
            let storage_offset = constraint_system.matrix.storage_offset(matrix_row);

            // Determine if this is a 4-var or 8-var polynomial
            let num_vars = match constraint_type {
                ConstraintType::GtExp { .. } | ConstraintType::GtMul => 4,
                ConstraintType::G1ScalarMul { .. } => 8,
            };

            // For 4-var: check positions 0, 16, 32, ... (unique values)
            // For 8-var: check all positions
            let step = if num_vars == 4 { 16 } else { 1 };
            let num_unique = if num_vars == 4 { 16 } else { 256 };

            for unique_idx in 0..5.min(num_unique) {
                let sparse_pos = storage_offset + unique_idx * step;
                let sparse_value = constraint_system.matrix.evaluations[sparse_pos];

                // Find this in the dense polynomial
                // We need to find which polynomial this belongs to
                let poly_idx = polynomials_info
                    .iter()
                    .position(|(c_idx, p_type, _)| {
                        *c_idx == constraint_idx && *p_type == poly_type
                    })
                    .expect("Should find polynomial");

                // Map to dense index
                let dense_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::sparse_to_dense(
                    &jagged_bijection,
                    poly_idx,
                    unique_idx,
                )
                .expect("Should map to dense");

                let dense_value = dense_poly.Z[dense_idx];

                assert_eq!(
                    sparse_value, dense_value,
                    "Reverse bijection failed: sparse[{}] → dense[{}]",
                    sparse_pos, dense_idx
                );

                verified_sparse_to_dense += 1;
            }
        }
    }
    println!("✓ Successfully verified {} sparse → dense mappings", verified_sparse_to_dense);

    // Test 3: Verify padding pattern for 4-var polynomials
    println!("\nTest 3: Verifying 4-var padding pattern");
    for constraint_idx in 0..3.min(constraint_system.constraints.len()) {
        if let ConstraintType::GtExp { .. } | ConstraintType::GtMul =
            &constraint_system.constraints[constraint_idx].constraint_type
        {
            let matrix_row = constraint_system.matrix.row_index(PolyType::Base, constraint_idx);
            let storage_offset = constraint_system.matrix.storage_offset(matrix_row);

            // Check that each unique value is repeated 16 times
            for unique_idx in 0..3 {
                let base_value = constraint_system.matrix.evaluations[storage_offset + unique_idx * 16];

                for repeat in 1..16 {
                    let padded_value = constraint_system.matrix.evaluations[storage_offset + unique_idx * 16 + repeat];
                    assert_eq!(
                        base_value, padded_value,
                        "Padding pattern broken at unique[{}] repeat[{}]",
                        unique_idx, repeat
                    );
                }
            }
        }
    }
    println!("✓ Padding pattern verified for 4-var polynomials");

    println!("\n✓ All bijection tests passed with real Dory witness!");
}