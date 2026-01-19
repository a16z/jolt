use crate::poly::dense_mlpoly::DensePolynomial;
use crate::zkvm::recursion::bijection::*;
use crate::zkvm::recursion::constraints_sys::{
    ConstraintSystem, ConstraintType, MatrixConstraint, PolyType,
};
use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{DoryCommitmentScheme, DoryGlobals},
        },
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::{Blake2bTranscript, Transcript},
    zkvm::recursion::RecursionProver,
};
use ark_bn254::Fq;
use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_ff::{One, Zero};
use ark_std::test_rng;
use rand::{rngs::StdRng, SeedableRng};
use serial_test::serial;

fn create_mixed_constraint_system() -> ConstraintSystem {
    let constraints = vec![
        MatrixConstraint {
            constraint_index: 0,
            constraint_type: ConstraintType::PackedGtExp,
        },
        MatrixConstraint {
            constraint_index: 1,
            constraint_type: ConstraintType::PackedGtExp,
        },
        MatrixConstraint {
            constraint_index: 2,
            constraint_type: ConstraintType::PackedGtExp,
        },
        MatrixConstraint {
            constraint_index: 3,
            constraint_type: ConstraintType::GtMul,
        },
        MatrixConstraint {
            constraint_index: 4,
            constraint_type: ConstraintType::GtMul,
        },
        MatrixConstraint {
            constraint_index: 5,
            constraint_type: ConstraintType::G1ScalarMul {
                base_point: (Fq::one(), Fq::one()),
            },
        },
    ];

    let g_poly = DensePolynomial::new(vec![Fq::one(); 2048]);

    ConstraintSystem::from_witness(
        constraints.into_iter().map(|c| c.constraint_type).collect(),
        g_poly,
    )
    .unwrap()
}

#[test]
fn test_bijection_with_constraint_system() {
    let cs = create_mixed_constraint_system();
    let builder = ConstraintSystemJaggedBuilder::from_constraints(&cs.constraints);
    let (bijection, mapping) = builder.build();

    // PackedGtExp has 2 poly types (RhoPrev, Quotient) - Base/Bit/RhoNext are public inputs: 3 × 2 = 6
    // GtMul has 4 poly types: 2 × 4 = 8
    // G1ScalarMul has 7 poly types: 1 × 7 = 7
    // Total: 21 polynomials
    assert_eq!(bijection.num_polynomials(), 21);
    assert_eq!(mapping.num_polynomials(), 21);

    // Dense sizes:
    // - PackedGtExp polynomials are 11-var (2048 each): 6 × 2048 = 12288
    // - GtMul polynomials are 4-var (16 each): 8 × 16 = 128
    // - G1ScalarMul polynomials are 8-var (256 each): 7 × 256 = 1792
    // Total: 14208
    assert_eq!(
        <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection),
        6 * 2048 + 8 * 16 + 7 * 256
    );

    let (c_idx, p_type) = mapping.decode(0);
    assert_eq!(c_idx, 0);
    // First poly type is now RhoPrev (Base was removed - it's a public input)
    assert_eq!(p_type as usize, PolyType::RhoPrev as usize);

    // With 6 PackedGtExp polys (indices 0-5), index 6 is the first GtMul poly
    let (c_idx, p_type) = mapping.decode(6);
    assert_eq!(c_idx, 3);
    assert_eq!(p_type as usize, PolyType::MulLhs as usize);

    // Index 14 (6 GT exp + 8 GT mul = 14) is the first G1ScalarMul poly
    let (c_idx, p_type) = mapping.decode(14);
    assert_eq!(c_idx, 5);
    assert_eq!(p_type as usize, PolyType::G1ScalarMulXA as usize);
}

#[test]
fn test_compression_ratio() {
    let cs = create_mixed_constraint_system();

    let num_rows = cs.matrix.num_rows;
    let evaluations_per_row = 1 << cs.matrix.num_constraint_vars;
    let sparse_padded_size = num_rows * evaluations_per_row;

    let (_, bijection, _mapping) = cs.build_dense_polynomial();
    let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection);

    let compression_ratio = 1.0 - (dense_size as f64 / sparse_padded_size as f64);

    assert!(
        compression_ratio > 0.5,
        "Expected >50% compression, got {:.2}%",
        compression_ratio * 100.0
    );
}

#[test]
fn test_dense_polynomial_extraction() {
    let cs = create_mixed_constraint_system();
    let (dense_poly, bijection, _mapping) = cs.build_dense_polynomial();

    assert_eq!(dense_poly.len(), dense_poly.len().next_power_of_two());

    let builder = ConstraintSystemJaggedBuilder::from_constraints(&cs.constraints);
    let (_, mapping) = builder.build();

    for dense_idx in 0..10 {
        let poly_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection, dense_idx);
        let eval_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection, dense_idx);

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
    let empty_bijection = VarCountJaggedBijection::new(vec![]);
    assert_eq!(
        <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&empty_bijection),
        0
    );
    assert_eq!(empty_bijection.num_polynomials(), 0);
    assert_eq!(
        <VarCountJaggedBijection as JaggedTransform<Fq>>::sparse_to_dense(&empty_bijection, 0, 0),
        None
    );

    let single = VarCountJaggedBijection::new(vec![JaggedPolynomial::new(3)]);
    assert_eq!(
        <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&single),
        8
    );
    assert_eq!(
        <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&single, 0),
        0
    );
    assert_eq!(
        <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&single, 7),
        0
    );
    assert_eq!(
        <VarCountJaggedBijection as JaggedTransform<Fq>>::sparse_to_dense(&single, 0, 7),
        Some(7)
    );
    assert_eq!(
        <VarCountJaggedBijection as JaggedTransform<Fq>>::sparse_to_dense(&single, 0, 8),
        None
    );
    assert_eq!(
        <VarCountJaggedBijection as JaggedTransform<Fq>>::sparse_to_dense(&single, 1, 0),
        None
    );

    let large = VarCountJaggedBijection::new(vec![JaggedPolynomial::new(10)]);
    assert_eq!(
        <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&large),
        1024
    );
    assert_eq!(
        <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&large, 1023),
        0
    );
    assert_eq!(
        <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&large, 1023),
        1023
    );
}

#[test]
fn test_mixed_variable_counts() {
    let polys = vec![
        JaggedPolynomial::new(2),
        JaggedPolynomial::new(3),
        JaggedPolynomial::new(4),
        JaggedPolynomial::new(5),
        JaggedPolynomial::new(6),
        JaggedPolynomial::new(7),
        JaggedPolynomial::new(8),
    ];

    let bijection = VarCountJaggedBijection::new(polys);

    let expected_size = 4 + 8 + 16 + 32 + 64 + 128 + 256;
    assert_eq!(
        <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection),
        expected_size
    );

    let mut cumulative = 0;
    for (i, size) in [4, 8, 16, 32, 64, 128, 256].iter().enumerate() {
        assert_eq!(
            <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection, cumulative),
            i
        );
        assert_eq!(
            <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection, cumulative),
            0
        );

        assert_eq!(
            <VarCountJaggedBijection as JaggedTransform<Fq>>::row(
                &bijection,
                cumulative + size - 1
            ),
            i
        );
        assert_eq!(
            <VarCountJaggedBijection as JaggedTransform<Fq>>::col(
                &bijection,
                cumulative + size - 1
            ),
            size - 1
        );

        cumulative += size;
    }
}

#[test]
fn test_constraint_mapping_consistency() {
    let cs = create_mixed_constraint_system();
    let builder = ConstraintSystemJaggedBuilder::from_constraints(&cs.constraints);
    let (_, mapping) = builder.build();

    for poly_idx in 0..mapping.num_polynomials() {
        let (constraint_idx, poly_type) = mapping.decode(poly_idx);

        assert!(
            constraint_idx < cs.constraints.len(),
            "Invalid constraint index {} for poly {}",
            constraint_idx,
            poly_idx
        );

        let constraint = &cs.constraints[constraint_idx];
        match constraint.constraint_type {
            ConstraintType::PackedGtExp => {
                // Base and Bit are public inputs, not committed polynomials
                assert!(
                    matches!(
                        poly_type,
                        PolyType::RhoPrev | PolyType::Quotient
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
    let bijection =
        VarCountJaggedBijection::new(vec![JaggedPolynomial::new(3), JaggedPolynomial::new(4)]);

    <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection, 100);
}

#[test]
fn test_jagged_bijection_with_real_dory_proof() {
    use crate::poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{DoryCommitmentScheme, DoryGlobals},
        },
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    };
    use crate::zkvm::recursion::ConstraintSystem;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;

    DoryGlobals::reset();
    DoryGlobals::initialize(1 << 2, 1 << 2);

    let num_vars = 4;
    let mut rng = StdRng::seed_from_u64(42);

    let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
    let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

    let coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));
    let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

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

    let (system, _hints) = ConstraintSystem::new(
        &proof,
        &verifier_setup,
        &mut extract_transcript,
        &point,
        &evaluation,
        &commitment,
    )
    .expect("System creation should succeed");

    let (dense_poly, bijection, _mapping) = system.build_dense_polynomial();

    let mut gt_exp_count = 0;
    let mut gt_mul_count = 0;
    let mut _g1_scalar_mul_count = 0;

    for constraint in &system.constraints {
        match &constraint.constraint_type {
            ConstraintType::PackedGtExp => gt_exp_count += 1,
            ConstraintType::GtMul => gt_mul_count += 1,
            ConstraintType::G1ScalarMul { .. } => _g1_scalar_mul_count += 1,
        }
    }

    let sparse_padded_size = system.matrix.num_rows * (1 << system.matrix.num_constraint_vars);
    let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection);
    let compression_ratio = 1.0 - (dense_size as f64 / sparse_padded_size as f64);

    if gt_exp_count > 0 || gt_mul_count > 0 {
        assert!(
            compression_ratio > 0.3,
            "Expected significant compression with GT operations, got {:.2}%",
            compression_ratio * 100.0
        );
    }

    for dense_idx in
        0..<VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection).min(100)
    {
        let row = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection, dense_idx);
        let col = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection, dense_idx);

        let reconstructed =
            <VarCountJaggedBijection as JaggedTransform<Fq>>::sparse_to_dense(&bijection, row, col)
                .expect("Should map back");
        assert_eq!(reconstructed, dense_idx);

        let num_vars =
            <VarCountJaggedBijection as JaggedTransform<Fq>>::poly_num_vars(&bijection, row);
        assert!(
            num_vars == 4 || num_vars == 8 || num_vars == 11,
            "Expected 4, 8, or 11 variables, got {}",
            num_vars
        );
    }

    let builder = ConstraintSystemJaggedBuilder::from_constraints(&system.constraints);
    let (_, mapping) = builder.build();

    for i in 0..10.min(<VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection)) {
        let poly_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection, i);
        let eval_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection, i);

        let (constraint_idx, poly_type) = mapping.decode(poly_idx);
        let matrix_row = system.matrix.row_index(poly_type, constraint_idx);
        let offset = system.matrix.storage_offset(matrix_row);

        let sparse_idx = eval_idx;

        let expected_val = system.matrix.evaluations[offset + sparse_idx];
        let actual_val = dense_poly.Z[i];

        assert_eq!(
            actual_val, expected_val,
            "Value mismatch at dense index {}",
            i
        );
    }
}

fn create_test_constraint_system_with_values() -> ConstraintSystem {
    let constraints = vec![
        MatrixConstraint {
            constraint_index: 0,
            constraint_type: ConstraintType::PackedGtExp,
        },
        MatrixConstraint {
            constraint_index: 1,
            constraint_type: ConstraintType::GtMul,
        },
        MatrixConstraint {
            constraint_index: 2,
            constraint_type: ConstraintType::G1ScalarMul {
                base_point: (Fq::one(), Fq::one()),
            },
        },
    ];

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
    let (dense_poly, jagged_bijection, _mapping) = constraint_system.build_dense_polynomial();

    let num_s_vars = constraint_system.num_s_vars();
    let num_x_vars = constraint_system.matrix.num_constraint_vars;

    let mut rng = test_rng();
    let zr: Vec<Fq> = (0..num_s_vars).map(|_| Fq::rand(&mut rng)).collect();
    let zc: Vec<Fq> = (0..num_x_vars).map(|_| Fq::rand(&mut rng)).collect();

    let sparse_mlpoly = MultilinearPolynomial::from(constraint_system.matrix.evaluations.clone());
    let mut eval_point = Vec::new();
    eval_point.extend_from_slice(&zc);
    eval_point.extend_from_slice(&zr);
    let eval_challenges: Vec<<Fq as JoltField>::Challenge> =
        eval_point.iter().rev().map(|&x| x.into()).collect();
    let sparse_eval = PolynomialEvaluation::evaluate(&sparse_mlpoly, &eval_challenges);

    let eq_row_evals = EqPolynomial::<Fq>::evals(&zr);
    let eq_col_evals = EqPolynomial::<Fq>::evals(&zc);

    let dense_size =
        <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&jagged_bijection);
    let mut dense_eval = Fq::zero();

    for i in 0..dense_size {
        let row = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&jagged_bijection, i);
        let col = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&jagged_bijection, i);
        let eq_row = eq_row_evals[row];
        let eq_col = eq_col_evals[col];
        let q_val = dense_poly.Z[i];
        dense_eval += q_val * eq_row * eq_col;
    }

    assert_eq!(
        sparse_eval, dense_eval,
        "Sparse and dense evaluations should match!"
    );
}

fn create_constraint_system_with_real_values() -> ConstraintSystem {
    let constraints = vec![
        MatrixConstraint {
            constraint_index: 0,
            constraint_type: ConstraintType::PackedGtExp,
        },
        MatrixConstraint {
            constraint_index: 1,
            constraint_type: ConstraintType::GtMul,
        },
        MatrixConstraint {
            constraint_index: 2,
            constraint_type: ConstraintType::G1ScalarMul {
                base_point: (Fq::one(), Fq::one()),
            },
        },
    ];

    let mut g_values = vec![];
    let mut rng = test_rng();
    for _ in 0..256 {
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
    let (dense_poly, jagged_bijection, _mapping) = constraint_system.build_dense_polynomial();

    let builder = ConstraintSystemJaggedBuilder::from_constraints(&constraint_system.constraints);
    let (_, mapping) = builder.build();

    let dense_size =
        <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&jagged_bijection);

    for dense_idx in 0..dense_size {
        let poly_idx =
            <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&jagged_bijection, dense_idx);
        let eval_idx =
            <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&jagged_bijection, dense_idx);

        let (constraint_idx, poly_type) = mapping.decode(poly_idx);

        let matrix_row = constraint_system
            .matrix
            .row_index(poly_type, constraint_idx);
        let storage_offset = constraint_system.matrix.storage_offset(matrix_row);

        let sparse_idx = eval_idx;

        let expected_val = constraint_system.matrix.evaluations[storage_offset + sparse_idx];
        let actual_val = dense_poly.Z[dense_idx];

        assert_eq!(
            actual_val, expected_val,
            "Value mismatch at dense[{}]: poly[{}] eval[{}] (constraint {} type {:?})",
            dense_idx, poly_idx, eval_idx, constraint_idx, poly_type
        );
    }

    let num_s_vars = constraint_system.num_s_vars();
    let num_x_vars = constraint_system.matrix.num_constraint_vars;

    let mut rng = test_rng();
    let zr: Vec<Fq> = (0..num_s_vars).map(|_| Fq::rand(&mut rng)).collect();
    let zc: Vec<Fq> = (0..num_x_vars).map(|_| Fq::rand(&mut rng)).collect();

    let sparse_mlpoly = MultilinearPolynomial::from(constraint_system.matrix.evaluations.clone());
    let mut eval_point = Vec::new();
    eval_point.extend_from_slice(&zc);
    eval_point.extend_from_slice(&zr);
    let eval_challenges: Vec<<Fq as JoltField>::Challenge> =
        eval_point.iter().rev().map(|&x| x.into()).collect();
    let sparse_eval = PolynomialEvaluation::evaluate(&sparse_mlpoly, &eval_challenges);

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

    assert_eq!(
        sparse_eval, dense_eval,
        "Jagged relation failed with real values!"
    );
}

#[test]
#[serial]
fn test_sparse_dense_bijection_with_real_dory_witness() {
    use crate::poly::commitment::dory::wrappers::ArkDoryProof;

    DoryGlobals::reset();
    DoryGlobals::initialize(1 << 2, 1 << 2);

    let mut rng = test_rng();

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

    let gamma = Fq::rand(&mut rng);
    let delta = Fq::rand(&mut rng);

    let mut witness_transcript: Blake2bTranscript = Transcript::new(b"dory_test_proof");

    use dory::backends::arkworks::ArkGT;
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

    let constraint_system = &prover.constraint_system;
    let (dense_poly, jagged_bijection, _mapping) = constraint_system.build_dense_polynomial();

    let dense_size =
        <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&jagged_bijection);

    let builder = ConstraintSystemJaggedBuilder::from_constraints(&constraint_system.constraints);
    let polynomials_info = builder.polynomials.clone();
    let (_, mapping) = builder.build();

    let sample_size = 100.min(dense_size);
    for dense_idx in (0..dense_size).step_by(dense_size / sample_size) {
        let row =
            <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&jagged_bijection, dense_idx);
        let col =
            <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&jagged_bijection, dense_idx);

        let (constraint_idx, poly_type) = mapping.decode(row);

        let matrix_row = constraint_system
            .matrix
            .row_index(poly_type, constraint_idx);
        let storage_offset = constraint_system.matrix.storage_offset(matrix_row);

        let sparse_idx = col;

        let sparse_value = constraint_system.matrix.evaluations[storage_offset + sparse_idx];
        let dense_value = dense_poly.Z[dense_idx];

        assert_eq!(
            sparse_value,
            dense_value,
            "Bijection failed at dense[{}] → sparse[{}] (poly {} type {:?})",
            dense_idx,
            storage_offset + sparse_idx,
            row,
            poly_type
        );
    }

    let mut _verified_sparse_to_dense = 0;

    for constraint_idx in 0..10.min(constraint_system.constraints.len()) {
        let constraint_type = &constraint_system.constraints[constraint_idx].constraint_type;

        let poly_types = match constraint_type {
            // Base and Bit are public inputs, not committed polynomials
            ConstraintType::PackedGtExp => vec![
                PolyType::RhoPrev,
                PolyType::Quotient,
            ],
            ConstraintType::GtMul => vec![
                PolyType::MulLhs,
                PolyType::MulRhs,
                PolyType::MulResult,
                PolyType::MulQuotient,
            ],
            ConstraintType::G1ScalarMul { .. } => {
                vec![PolyType::G1ScalarMulXA, PolyType::G1ScalarMulYA]
            }
        };

        for poly_type in poly_types {
            let matrix_row = constraint_system
                .matrix
                .row_index(poly_type, constraint_idx);
            let storage_offset = constraint_system.matrix.storage_offset(matrix_row);

            // PackedGtExp uses 11-var, GtMul uses 4-var, G1ScalarMul uses 8-var
            let num_vars = match constraint_type {
                ConstraintType::PackedGtExp => 11,
                ConstraintType::GtMul => 4,
                ConstraintType::G1ScalarMul { .. } => 8,
            };

            let step = 1;
            let num_unique = 1 << num_vars;

            for unique_idx in 0..5.min(num_unique) {
                let sparse_pos = storage_offset + unique_idx * step;
                let sparse_value = constraint_system.matrix.evaluations[sparse_pos];

                let poly_idx = polynomials_info
                    .iter()
                    .position(|(c_idx, p_type, _)| *c_idx == constraint_idx && *p_type == poly_type)
                    .expect("Should find polynomial");

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

                _verified_sparse_to_dense += 1;
            }
        }
    }

    // Only test GtMul constraints for zero padding (PackedGtExp uses full 11-var, no padding)
    for constraint_idx in 0..3.min(constraint_system.constraints.len()) {
        if let ConstraintType::GtMul = &constraint_system.constraints[constraint_idx].constraint_type
        {
            let matrix_row = constraint_system
                .matrix
                .row_index(PolyType::MulLhs, constraint_idx);
            let storage_offset = constraint_system.matrix.storage_offset(matrix_row);

            let mut has_nonzero = false;
            for i in 0..16 {
                if !constraint_system.matrix.evaluations[storage_offset + i].is_zero() {
                    has_nonzero = true;
                    break;
                }
            }
            assert!(
                has_nonzero,
                "Should have at least one non-zero value in first 16 positions"
            );

            // GtMul is 4-var (16 elements) padded to 11-var (2048), so positions 16..2048 should be zero
            for i in 16..2048 {
                assert!(
                    constraint_system.matrix.evaluations[storage_offset + i].is_zero(),
                    "Position {} should be zero for 4-var polynomial with zero padding",
                    i
                );
            }
        }
    }
}
