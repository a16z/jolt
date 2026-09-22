//! Trusted v2 preprocessing and conditional clear SPARK proving.
//! This prototype does not establish joint extraction, ROM composition or ZK.
mod protocol;
pub mod sparse;

use jolt_field::{CanonicalBytes, Fr, Ring, Zero};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme};
use jolt_openings::CommitmentScheme;
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_r1cs::{LinearCombination, Variable};
use jolt_spartan_verifier::preprocessed::{
    ComputationKey, MatrixApplicationIds, MatrixError, MatrixShape, PublicColumnProof,
};
use jolt_spartan_verifier::SpartanKey;
use jolt_transcript::Bn254WideBlake2bTranscript;

/// Committed tables from authenticated preprocessing, retained for future sparse
/// evaluation. Immutable getters support independent encoding checks.
pub struct PreprocessedMatrices {
    key: ComputationKey,
    direct: SpartanKey<Fr>,
    public: Vec<Fr>,
    operations: Vec<Fr>,
    memory: Vec<Fr>,
    addresses: [Vec<(usize, usize)>; 3],
}
impl PreprocessedMatrices {
    pub fn new(
        key: &SpartanKey<Fr>,
        application: MatrixApplicationIds,
        setup: &HyperKZGProverSetup,
    ) -> Result<Self, MatrixError> {
        let direct = key.clone();
        let matrices = key.matrices();
        let normalized: Vec<_> = [&matrices.a, &matrices.b, &matrices.c]
            .into_iter()
            .map(|matrix| {
                matrix
                    .iter()
                    .map(|row| {
                        LinearCombination {
                            terms: row.iter().map(|(c, v)| (Variable::new(*c), *v)).collect(),
                        }
                        .into_sparse_row()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let max_private = normalized
            .iter()
            .map(|matrix| {
                matrix
                    .iter()
                    .flatten()
                    .filter(|(column, _)| *column >= key.public_columns())
                    .count()
            })
            .max()
            .ok_or(MatrixError::Shape)?;
        let shape = MatrixShape::new(
            matrices.num_constraints,
            matrices.num_vars,
            key.public_columns(),
            max_private,
        )?;
        let public_len = shape
            .public_slabs
            .checked_mul(shape.padded_rows)
            .ok_or(MatrixError::Shape)?;
        let ops_len = shape.operations.checked_mul(16).ok_or(MatrixError::Shape)?;
        let memory_len = shape.memory.checked_mul(2).ok_or(MatrixError::Shape)?;
        let vk = HyperKZGScheme::verifier_setup(setup);
        let capacity = vk.binding()?.num_powers;
        if [public_len, ops_len, memory_len]
            .into_iter()
            .any(|n| n as u64 > capacity)
        {
            return Err(MatrixError::Shape);
        }
        let mut addresses: [Vec<(usize, usize)>; 3] = std::array::from_fn(|_| Vec::new());
        let mut public = vec![Fr::zero(); public_len];
        let mut operations = vec![Fr::zero(); ops_len];
        let mut row_audit = vec![0u64; shape.memory];
        let mut col_audit = vec![0u64; shape.memory];
        let mut encoding = b"JOLT-SPARK-MAT\0\0".to_vec();
        encoding.extend((shape.rows as u64).to_le_bytes());
        encoding.extend((shape.columns as u64).to_le_bytes());
        for (matrix_index, matrix) in normalized.iter().enumerate() {
            let mut entries = Vec::new();
            for (row_index, row) in matrix.iter().enumerate() {
                encoding.extend((row.len() as u64).to_le_bytes());
                for &(column, value) in row {
                    encoding.extend((column as u64).to_le_bytes());
                    encoding.extend(value.to_bytes_le_vec());
                    if column < shape.public_columns {
                        let index = (matrix_index * shape.public_columns + column)
                            * shape.padded_rows
                            + row_index;
                        *public.get_mut(index).ok_or(MatrixError::Shape)? = value;
                    } else {
                        entries.push((row_index, column - shape.public_columns, value));
                    }
                }
            }
            entries.resize(shape.operations, (0, 0, Fr::zero()));
            for (operation, (row, column, value)) in entries.into_iter().enumerate() {
                addresses
                    .get_mut(matrix_index)
                    .ok_or(MatrixError::Shape)?
                    .push((row, column));
                let rt = row_audit.get_mut(row).ok_or(MatrixError::Shape)?;
                let ct = col_audit.get_mut(column).ok_or(MatrixError::Shape)?;
                for (slab, v) in [
                    (matrix_index, Fr::from_u64(row as u64)),
                    (3 + matrix_index, Fr::from_u64(*rt)),
                    (6 + matrix_index, Fr::from_u64(column as u64)),
                    (9 + matrix_index, Fr::from_u64(*ct)),
                    (12 + matrix_index, value),
                ] {
                    *operations
                        .get_mut(slab * shape.operations + operation)
                        .ok_or(MatrixError::Shape)? = v;
                }
                *rt = rt.checked_add(1).ok_or(MatrixError::Shape)?;
                *ct = ct.checked_add(1).ok_or(MatrixError::Shape)?;
            }
        }
        let memory = row_audit
            .into_iter()
            .chain(col_audit)
            .map(Fr::from_u64)
            .collect::<Vec<_>>();
        let mut commitments = Vec::with_capacity(3);
        for table in [&public, &operations, &memory] {
            commitments.push(HyperKZGScheme::commit(&Polynomial::new(table.clone()), setup)?.0);
        }
        let commitments = commitments.try_into().map_err(|_| MatrixError::Shape)?;
        let key = ComputationKey::new(
            shape,
            application,
            ComputationKey::digest(&encoding),
            commitments,
            &vk,
        )?;
        Ok(Self {
            key,
            direct,
            public,
            operations,
            memory,
            addresses,
        })
    }
    pub fn key(&self) -> &ComputationKey {
        &self.key
    }
    pub fn public_table(&self) -> &[Fr] {
        &self.public
    }
    pub fn operations_table(&self) -> &[Fr] {
        &self.operations
    }
    pub fn memory_table(&self) -> &[Fr] {
        &self.memory
    }

    /// Opens every public column at rx under one actual merged PCS commitment.
    pub fn prove_public(
        &self,
        rx: &[Fr],
        setup: &HyperKZGProverSetup,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<PublicColumnProof, MatrixError> {
        let shape = self.key.shape();
        if rx.len() != shape.padded_rows.trailing_zeros() as usize {
            return Err(MatrixError::Query);
        }
        let weights = EqPolynomial::new(rx.to_vec()).evaluations();
        let evaluations = self
            .public
            .chunks_exact(shape.padded_rows)
            .take(shape.public_columns * 3)
            .map(|column| column.iter().zip(&weights).map(|(a, b)| *a * b).sum())
            .collect::<Vec<_>>();
        let (point, value) = self
            .key
            .public_opening_query(rx, &evaluations, transcript)?;
        let opening = HyperKZGScheme::open(
            &Polynomial::new(self.public.clone()),
            &point,
            value,
            setup,
            None,
            transcript,
        )?;
        Ok(PublicColumnProof {
            evaluations,
            opening,
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "fixed independent fixture assertions"
)]
mod tests {
    use super::*;
    use crate::{prove, prove_rounds, rounds::OuterRounds};
    use jolt_crypto::{Bn254, JoltGroup};
    use jolt_field::One;
    use jolt_hyperkzg::{HyperKZGSetupParams, HyperKZGVerifierSetup};
    use jolt_poly::CompressedPoly;
    use jolt_r1cs::ConstraintMatrices;
    use jolt_spartan_verifier::{
        preprocessed::{PreprocessedProof, PublicColumnQuery},
        OUTER_DEGREE,
    };
    use jolt_sumcheck::{BooleanHypercube, SumcheckClaim, SUMCHECK_ROUND_TRANSCRIPT_LABEL};
    use jolt_transcript::Transcript;

    pub(super) fn setup() -> (HyperKZGProverSetup, HyperKZGVerifierSetup) {
        let beta = Fr::from_u64(7);
        HyperKZGScheme::setup(HyperKZGSetupParams {
            g1_powers: std::iter::successors(Some(Fr::one()), |x| Some(*x * beta))
                .take(64)
                .map(|x| Bn254::g1_generator().scalar_mul(&x))
                .collect(),
            g2: Bn254::g2_generator(),
            beta_g2: Bn254::g2_generator().scalar_mul(&beta),
            setup_id: [9; 32],
            max_public_degree: 63,
        })
        .unwrap()
    }
    pub(super) fn relation() -> SpartanKey<Fr> {
        let o = Fr::one();
        SpartanKey::new(
            ConstraintMatrices::new(
                4,
                8,
                vec![
                    vec![(3, o), (0, o), (3, o + o), (3, -(o + o)), (2, Fr::zero())],
                    vec![(5, o)],
                    vec![(2, o + o)],
                    vec![(7, o)],
                ],
                vec![vec![(4, o)], vec![(0, o)], vec![(6, o)], vec![(1, o)]],
                vec![vec![], vec![], vec![], vec![]],
            ),
            2,
            [19; 32],
        )
        .unwrap()
    }
    pub(super) fn ids() -> MatrixApplicationIds {
        MatrixApplicationIds {
            circuit: [1; 32],
            profile: [2; 32],
            public_schema: [3; 32],
            table: [4; 32],
        }
    }
    fn transcript() -> Bn254WideBlake2bTranscript {
        Bn254WideBlake2bTranscript::new(b"spartan-preprocessed-clear-v2")
    }

    #[test]
    fn preprocessing_normalizes_and_carries_audits_across_dummy_operations() {
        let (pk, _) = setup();
        let tables = PreprocessedMatrices::new(&relation(), ids(), &pk).unwrap();
        assert_eq!(tables.key().shape().operations, 4);
        assert_eq!(tables.public_table().len(), 64);
        assert_eq!(tables.operations_table()[20], Fr::from_u64(5));
        assert_eq!(tables.operations_table()[44], Fr::from_u64(4));
        assert_eq!(tables.memory_table()[0], Fr::from_u64(9));
        assert_eq!(tables.memory_table()[8], Fr::from_u64(8));
        assert!(tables.operations_table()[60..].iter().all(Zero::is_zero));
        assert_eq!(tables.public_table()[0], Fr::one());
    }

    #[test]
    fn real_outer_sumcheck_and_public_pcs_v2_prefix() {
        let (pk, vk) = setup();
        let direct = relation();
        let tables = PreprocessedMatrices::new(&direct, ids(), &pk).unwrap();
        let key = tables.key();
        let id = key.id();
        let inputs = [5, 0].map(Fr::from_u64);
        let witness = [3, 0, 0, 11, 0].map(Fr::from_u64);
        let proof =
            prove::<HyperKZGScheme>(&direct, &inputs, &witness, &pk, &mut transcript()).unwrap();
        direct
            .verify::<HyperKZGScheme>(&inputs, &proof, &vk, &mut transcript())
            .unwrap();
        let mut pt = transcript();
        let mut vt = transcript();
        let tau = key
            .begin(&id, &vk, &inputs, &proof.witness_commitment, &mut pt)
            .unwrap();
        assert_eq!(
            tau,
            key.begin(&id, &vk, &inputs, &proof.witness_commitment, &mut vt)
                .unwrap()
        );
        let assignment = std::iter::once(Fr::one())
            .chain(inputs)
            .chain(witness)
            .collect::<Vec<_>>();
        let mut rounds = OuterRounds::new(&direct, &assignment, &tau).unwrap();
        let (outer, rx, claim) =
            prove_rounds(&mut rounds, OUTER_DEGREE, Fr::zero(), &mut pt).unwrap();
        let evals = rounds.evaluations().unwrap();
        let checked = outer
            .verify(
                &SumcheckClaim {
                    num_vars: 2,
                    degree: OUTER_DEGREE,
                    claimed_sum: Fr::zero(),
                },
                BooleanHypercube,
                SUMCHECK_ROUND_TRANSCRIPT_LABEL,
                &mut vt,
            )
            .unwrap();
        direct
            .check_outer(&tau, checked.point.as_slice(), checked.value, evals)
            .unwrap();
        assert_eq!(rx, checked.point.as_slice());
        assert_eq!(claim, checked.value);
        pt.append_values(b"outer-evaluations", &evals);
        vt.append_values(b"outer-evaluations", &evals);
        let rho = [pt.challenge(), pt.challenge(), pt.challenge()];
        assert_eq!(rho, [vt.challenge(), vt.challenge(), vt.challenge()]);
        let opening = tables.prove_public(&rx, &pk, &mut pt).unwrap();
        let contribution = key
            .verify_public(
                &id,
                &vk,
                PublicColumnQuery {
                    inputs: &inputs,
                    point: &rx,
                    matrix_weights: rho,
                },
                &opening,
                &mut vt,
            )
            .unwrap();
        let row_weights = EqPolynomial::new(rx).evaluations();
        let public = [Fr::one(), inputs[0], inputs[1]];
        assert_eq!(
            contribution,
            direct
                .matrices()
                .linear_form_bilinear_eval(&row_weights, &public, 0, 3, rho)
                .unwrap()
        );
        let inner = evals.iter().zip(rho).map(|(a, b)| *a * b).sum::<Fr>() - contribution;
        pt.append_labeled(b"spartan-inner", &inner);
        vt.append_labeled(b"spartan-inner", &inner);
        assert_eq!(pt.state(), vt.state());
        assert_eq!(
            pt.state(),
            [
                141, 43, 136, 230, 234, 237, 101, 126, 216, 184, 247, 28, 160, 166, 131, 129, 13,
                76, 194, 133, 60, 152, 43, 147, 8, 86, 96, 226, 104, 217, 120, 26
            ]
        );
        assert_eq!(
            pt.challenge().to_bytes_le_vec(),
            [
                88, 254, 116, 183, 154, 16, 100, 186, 123, 20, 5, 61, 135, 185, 11, 184, 172, 74,
                253, 156, 158, 251, 116, 156, 129, 49, 130, 210, 67, 188, 4, 1
            ]
        );
    }

    #[test]
    fn public_pcs_rejects_tampered_values_points_key_and_shapes() {
        let (pk, vk) = setup();
        let tables = PreprocessedMatrices::new(&relation(), ids(), &pk).unwrap();
        let key = tables.key();
        let id = key.id();
        let rx = [2, 3].map(Fr::from_u64);
        let proof = tables.prove_public(&rx, &pk, &mut transcript()).unwrap();
        let inputs = [5, 0].map(Fr::from_u64);
        let verify = |proof: &PublicColumnProof, point: &[Fr], id: &[u8; 32]| {
            key.verify_public(
                id,
                &vk,
                PublicColumnQuery {
                    inputs: &inputs,
                    point,
                    matrix_weights: [Fr::one(); 3],
                },
                proof,
                &mut transcript(),
            )
        };
        assert!(verify(&proof, &rx, &id).is_ok());
        let mut altered = proof.clone();
        altered.evaluations[0] += Fr::one();
        assert!(verify(&altered, &rx, &id).is_err());
        altered = proof.clone();
        let _ = altered.evaluations.pop();
        assert!(matches!(
            verify(&altered, &rx, &id),
            Err(MatrixError::Query)
        ));
        assert!(verify(&proof, &[Fr::one(), Fr::one()], &id).is_err());
        assert!(matches!(
            verify(&proof, &rx, &[0; 32]),
            Err(MatrixError::Identity)
        ));
        assert!(matches!(
            verify(&proof, &rx[..1], &id),
            Err(MatrixError::Query)
        ));
        altered = proof.clone();
        altered.opening.w[0] = Bn254::g1_generator();
        assert!(verify(&altered, &rx, &id).is_err());
        assert!(MatrixShape::new(0, 8, 3, 4).is_err());
        assert!(MatrixShape::new(4, 8, 8, 4).is_err());
    }
    #[test]
    fn complete_v2_relation_and_each_authenticated_component_reject_tampering() {
        let (pk, vk) = setup();
        let tables = PreprocessedMatrices::new(&relation(), ids(), &pk).unwrap();
        let key = tables.key();
        let id = key.id();
        let inputs = [5, 0].map(Fr::from_u64);
        let proof = tables
            .prove(&inputs, &[3, 0, 0, 11, 0].map(Fr::from_u64), &pk)
            .unwrap();
        key.verify(&id, &inputs, &proof, &vk).unwrap();
        let reject =
            |proof: &PreprocessedProof| assert!(key.verify(&id, &inputs, proof, &vk).is_err());
        for i in 0..16 {
            let mut bad = proof.clone();
            bad.sparse.roots[i] += Fr::one();
            reject(&bad);
        }
        for i in 0..6 {
            let mut bad = proof.clone();
            bad.sparse.dot_halves[i] += Fr::one();
            reject(&bad);
        }
        for i in 0..6 {
            let mut bad = proof.clone();
            bad.sparse.dereferences.evaluations[i] += Fr::one();
            reject(&bad);
        }
        for i in 0..16 {
            let mut bad = proof.clone();
            bad.sparse.operation_values.evaluations[i] += Fr::one();
            reject(&bad);
        }
        for i in 0..2 {
            let mut bad = proof.clone();
            bad.sparse.audit_values.evaluations[i] += Fr::one();
            reject(&bad);
        }
        for i in 0..3 {
            let mut bad = proof.clone();
            bad.private_values[i] += Fr::one();
            reject(&bad);
        }
        let mut bad = proof.clone();
        bad.sparse.dereference_commitment = Bn254::g1_generator();
        reject(&bad);
        bad = proof.clone();
        bad.sparse.dereferences.opening.w[0] = Bn254::g1_generator();
        reject(&bad);
        bad = proof.clone();
        bad.sparse.operation_values.opening.w[0] = Bn254::g1_generator();
        reject(&bad);
        bad = proof.clone();
        bad.sparse.audit_values.opening.w[0] = Bn254::g1_generator();
        reject(&bad);
        bad = proof.clone();
        bad.witness_opening.w[0] = Bn254::g1_generator();
        reject(&bad);
        bad = proof.clone();
        bad.witness_evaluation += Fr::one();
        reject(&bad);
        bad = proof.clone();
        bad.sparse.operations.layers[0].ends[0][0] += Fr::one();
        reject(&bad);
        bad = proof.clone();
        bad.sparse.memory.layers[0].ends[0][1] += Fr::one();
        reject(&bad);
        bad = proof.clone();
        bad.sparse.operations.layers[1].dot_ends[0][2] += Fr::one();
        reject(&bad);
        bad = proof.clone();
        let _ = bad.sparse.operations.layers.pop();
        reject(&bad);
        bad = proof.clone();
        bad.sparse.memory.layers[0].ends.clear();
        reject(&bad);
        bad = proof.clone();
        bad.sparse.operations.layers[0]
            .dot_ends
            .push([Fr::zero(); 3]);
        reject(&bad);
        bad = proof.clone();
        bad.sparse.audit_values.evaluations.push(Fr::zero());
        reject(&bad);
        assert!(key
            .verify(&id, &[6, 0].map(Fr::from_u64), &proof, &vk)
            .is_err());
        assert!(key.verify(&id, &inputs[..1], &proof, &vk).is_err());
        assert!(key.verify(&[0; 32], &inputs, &proof, &vk).is_err());
    }

    #[test]
    fn sparse_non_boolean_queries_and_empty_n2_relation() {
        use jolt_spartan_verifier::preprocessed::sparse::SparseQuery;
        let (pk, vk) = setup();
        let direct = relation();
        let tables = PreprocessedMatrices::new(&direct, ids(), &pk).unwrap();
        for (x, y) in [([2, 3], [4, 5, 6]), ([7, 11], [13, 17, 19])] {
            let x = x.map(Fr::from_u64);
            let y = y.map(Fr::from_u64);
            let rw = EqPolynomial::new(x.to_vec()).evaluations();
            let cw = EqPolynomial::new(y.to_vec()).evaluations();
            let mut values = [Fr::zero(); 3];
            for (i, value) in values.iter_mut().enumerate() {
                let mut rho = [Fr::zero(); 3];
                rho[i] = Fr::one();
                *value = direct
                    .matrices()
                    .linear_form_bilinear_eval(&rw, &cw[..5], 3, 5, rho)
                    .unwrap();
            }
            let query = || SparseQuery {
                rows: &x,
                columns: &y,
                values,
            };
            let proof = tables
                .prove_sparse(query(), &pk, &mut transcript())
                .unwrap();
            tables
                .key()
                .verify_sparse(query(), &proof, &vk, &mut transcript())
                .unwrap();
            let changed = [Fr::one(), Fr::one()];
            assert!(tables
                .key()
                .verify_sparse(
                    SparseQuery {
                        rows: &changed,
                        columns: &y,
                        values
                    },
                    &proof,
                    &vk,
                    &mut transcript()
                )
                .is_err());
            assert!(tables
                .key()
                .verify_sparse(
                    SparseQuery {
                        rows: &x[..1],
                        columns: &y,
                        values
                    },
                    &proof,
                    &vk,
                    &mut transcript()
                )
                .is_err());
        }
        let empty = SpartanKey::new(
            ConstraintMatrices::new(4, 8, vec![vec![]; 4], vec![vec![]; 4], vec![vec![]; 4]),
            2,
            [19; 32],
        )
        .unwrap();
        let tables = PreprocessedMatrices::new(&empty, ids(), &pk).unwrap();
        assert_eq!(tables.key().shape().operations, 2);
        let inputs = [5, 0].map(Fr::from_u64);
        let proof = tables
            .prove(&inputs, &[3, 4, 5, 6, 7].map(Fr::from_u64), &pk)
            .unwrap();
        assert!(proof.sparse.operations.layers[0]
            .sumcheck
            .round_polynomials
            .is_empty());
        assert_eq!(proof.sparse.operations.layers[0].dot_ends.len(), 6);
        tables
            .key()
            .verify(&tables.key().id(), &inputs, &proof, &vk)
            .unwrap();
    }

    #[test]
    fn actual_setup_vector_and_changed_setup_are_bound_at_full_verifier_entry() {
        let setup_with = |beta: Fr, id: [u8; 32], degree: u64| {
            HyperKZGScheme::setup(HyperKZGSetupParams {
                g1_powers: std::iter::successors(Some(Fr::one()), |x| Some(*x * beta))
                    .take(64)
                    .map(|x| Bn254::g1_generator().scalar_mul(&x))
                    .collect(),
                g2: Bn254::g2_generator(),
                beta_g2: Bn254::g2_generator().scalar_mul(&beta),
                setup_id: id,
                max_public_degree: degree,
            })
            .unwrap()
        };
        let (_, one_vk) = setup_with(Fr::one(), [9; 32], 63);
        assert_eq!(
            one_vk.binding().unwrap().canonical_bytes,
            include_bytes!("../tests/fixtures/beta-one-setup.bin").as_slice()
        );
        let (pk, vk) = setup();
        let tables = PreprocessedMatrices::new(&relation(), ids(), &pk).unwrap();
        let inputs = [5, 0].map(Fr::from_u64);
        let proof = tables
            .prove(&inputs, &[3, 0, 0, 11, 0].map(Fr::from_u64), &pk)
            .unwrap();
        tables
            .key()
            .verify(&tables.key().id(), &inputs, &proof, &vk)
            .unwrap();
        for (_, changed) in [
            setup_with(Fr::from_u64(7), [10; 32], 63),
            setup_with(Fr::from_u64(7), [9; 32], 64),
            setup_with(Fr::from_u64(11), [9; 32], 63),
        ] {
            assert!(matches!(
                tables
                    .key()
                    .verify(&tables.key().id(), &inputs, &proof, &changed),
                Err(MatrixError::Identity)
            ));
        }
    }
    #[test]
    fn wire_v1_accepts_honest_proof_and_rejects_untrusted_bytes() {
        use jolt_field::CanonicalBytes;
        use jolt_spartan_verifier::preprocessed::wire::{verify_bytes, WireError};
        let (pk, vk) = setup();
        let tables = PreprocessedMatrices::new(&relation(), ids(), &pk).unwrap();
        let key = tables.key();
        let inputs = [5, 0].map(Fr::from_u64);
        let proof = tables
            .prove(&inputs, &[3, 0, 0, 11, 0].map(Fr::from_u64), &pk)
            .unwrap();
        let input_bytes = inputs
            .iter()
            .flat_map(CanonicalBytes::to_bytes_le_vec)
            .collect::<Vec<_>>();
        let key_bytes = key.canonical_bytes();
        let encoded = key.encode_proof(&proof).unwrap();
        assert_eq!(
            encoded,
            include_bytes!("../tests/fixtures/preprocessed-wire-v1-toy.bin").as_slice()
        );
        assert_eq!(encoded.len(), 9149);
        verify_bytes(&key.id(), &vk, &key_bytes, &input_bytes, &encoded).unwrap();
        let decoded = key.decode_proof(&encoded).unwrap();
        assert_eq!(key.encode_proof(&decoded).unwrap(), encoded);
        // Fixed geometry has no attacker-selected count or recursive container depth.
        for len in [0, 1, 51, 52, encoded.len() - 1] {
            assert!(matches!(
                verify_bytes(&key.id(), &vk, &key_bytes, &input_bytes, &encoded[..len]),
                Err(WireError::Length)
            ));
        }
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(matches!(
            key.decode_proof(&trailing),
            Err(WireError::Length)
        ));
        for offset in [0, 16] {
            let mut bad = encoded.clone();
            bad[offset] ^= 1;
            assert!(matches!(key.decode_proof(&bad), Err(WireError::Version)));
        }
        let mut bad = encoded.clone();
        bad[20] ^= 1;
        assert!(matches!(
            key.decode_proof(&bad),
            Err(WireError::Protocol(MatrixError::Identity))
        ));
        let mut bad = encoded.clone();
        bad[52..84].fill(255);
        assert!(matches!(key.decode_proof(&bad), Err(WireError::Group)));
        let mut bad = encoded.clone();
        bad[85..117].fill(255);
        assert!(matches!(key.decode_proof(&bad), Err(WireError::Field)));
        let mut modulus_alias = encoded.clone();
        modulus_alias[85..117].copy_from_slice(&key_bytes[44..76]);
        assert!(matches!(
            key.decode_proof(&modulus_alias),
            Err(WireError::Field)
        ));
        let mut bad_inputs = input_bytes.clone();
        bad_inputs[..32].fill(255);
        assert!(matches!(
            verify_bytes(&key.id(), &vk, &key_bytes, &bad_inputs, &encoded),
            Err(WireError::Field)
        ));
        assert!(matches!(
            verify_bytes(&key.id(), &vk, &key_bytes, &input_bytes[..63], &encoded),
            Err(WireError::Length)
        ));
        let mut bad_inputs = input_bytes.clone();
        bad_inputs[0] += 1;
        assert!(verify_bytes(&key.id(), &vk, &key_bytes, &bad_inputs, &encoded).is_err());
        let mut bad = encoded.clone();
        bad[85] ^= 1;
        assert!(verify_bytes(&key.id(), &vk, &key_bytes, &input_bytes, &bad).is_err());
        for width in [0, 4, 255] {
            let mut invalid_width = encoded.clone();
            invalid_width[84] = width;
            assert!(matches!(
                key.decode_proof(&invalid_width),
                Err(WireError::RoundEncoding)
            ));
        }
        let mut padded = encoded.clone();
        padded[84] = 1;
        padded[117..181].fill(0);
        padded[117] = 1;
        assert!(matches!(key.decode_proof(&padded), Err(WireError::Padding)));
        let mut shortened = proof.clone();
        shortened.outer.round_polynomials[0] = CompressedPoly::new(vec![]);
        assert!(matches!(
            key.encode_proof(&shortened),
            Err(WireError::RoundEncoding)
        ));
        let mut bad_zero_round = proof;
        bad_zero_round.sparse.operations.layers[0]
            .sumcheck
            .round_polynomials
            .push(CompressedPoly::new(vec![Fr::zero(); 3]));
        assert!(matches!(
            key.encode_proof(&bad_zero_round),
            Err(WireError::Shape)
        ));
    }

    #[test]
    fn wire_v1_key_identity_geometry_and_setup_gates() {
        use jolt_spartan_verifier::preprocessed::wire::WireError;
        let (pk, vk) = setup();
        let tables = PreprocessedMatrices::new(&relation(), ids(), &pk).unwrap();
        let key = tables.key();
        let bytes = key.canonical_bytes();
        assert!(matches!(
            ComputationKey::decode_wire(&bytes, &[0; 32], &vk),
            Err(WireError::Protocol(MatrixError::Identity))
        ));
        for len in [0, 475] {
            assert!(matches!(
                ComputationKey::decode_wire(&bytes[..len], &key.id(), &vk),
                Err(WireError::Length)
            ));
        }
        let mut bad = bytes.clone();
        bad.push(0);
        assert!(matches!(
            ComputationKey::decode_wire(&bad, &key.id(), &vk),
            Err(WireError::Length)
        ));
        let mut bad = bytes.clone();
        bad[0] ^= 1;
        assert!(matches!(
            ComputationKey::decode_wire(&bad, &ComputationKey::digest(&bad), &vk),
            Err(WireError::Version)
        ));
        let mut bad = bytes.clone();
        bad[76..84].fill(255);
        assert!(ComputationKey::decode_wire(&bad, &ComputationKey::digest(&bad), &vk).is_err());
        let mut too_deep = bytes.clone();
        for offset in [76, 100, 124] {
            too_deep[offset..offset + 8].copy_from_slice(&(1u64 << 33).to_le_bytes());
        }
        assert!(matches!(
            ComputationKey::decode_wire(&too_deep, &ComputationKey::digest(&too_deep), &vk),
            Err(WireError::Limit)
        ));
        let mut too_public = bytes.clone();
        for (offset, value) in [(84, 1030u64), (92, 1025), (108, 8), (124, 8), (132, 4096)] {
            too_public[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }
        assert!(matches!(
            ComputationKey::decode_wire(&too_public, &ComputationKey::digest(&too_public), &vk),
            Err(WireError::Limit)
        ));
        let mut bad = bytes.clone();
        bad[380..412].fill(255);
        assert!(matches!(
            ComputationKey::decode_wire(&bad, &ComputationKey::digest(&bad), &vk),
            Err(WireError::Group)
        ));
        let (_, changed_setup) = HyperKZGScheme::setup(HyperKZGSetupParams {
            g1_powers: vec![Bn254::g1_generator(); 64],
            g2: Bn254::g2_generator(),
            beta_g2: Bn254::g2_generator(),
            setup_id: [10; 32],
            max_public_degree: 63,
        })
        .unwrap();
        assert!(matches!(
            ComputationKey::decode_wire(&bytes, &key.id(), &changed_setup),
            Err(WireError::Protocol(MatrixError::Identity))
        ));
    }

    #[test]
    fn wire_v1_empty_matrix_and_zero_product_networks_are_complete() {
        use jolt_field::CanonicalBytes;
        use jolt_spartan_verifier::preprocessed::wire::verify_bytes;
        let (pk, vk) = setup();
        let empty = SpartanKey::new(
            ConstraintMatrices::new(4, 8, vec![vec![]; 4], vec![vec![]; 4], vec![vec![]; 4]),
            2,
            [19; 32],
        )
        .unwrap();
        let tables = PreprocessedMatrices::new(&empty, ids(), &pk).unwrap();
        let inputs = [Fr::zero(); 2];
        let proof = tables.prove(&inputs, &[Fr::zero(); 5], &pk).unwrap();
        let key = tables.key();
        assert_eq!(key.shape().operations, 2);
        assert!(proof.sparse.operations.layers[0]
            .sumcheck
            .round_polynomials
            .is_empty());
        let bytes = key.encode_proof(&proof).unwrap();
        verify_bytes(
            &key.id(),
            &vk,
            &key.canonical_bytes(),
            &inputs
                .iter()
                .flat_map(CanonicalBytes::to_bytes_le_vec)
                .collect::<Vec<_>>(),
            &bytes,
        )
        .unwrap();
        assert_eq!(
            key.encode_proof(&key.decode_proof(&bytes).unwrap())
                .unwrap(),
            bytes
        );
    }

    #[test]
    fn full_v2_transcript_and_actual_payload_regression() {
        let (pk, vk) = setup();
        let tables = PreprocessedMatrices::new(&relation(), ids(), &pk).unwrap();
        let inputs = [5, 0].map(Fr::from_u64);
        let (proof, mut transcript) = tables
            .prove_session(&inputs, &[3, 0, 0, 11, 0].map(Fr::from_u64), &pk)
            .unwrap();
        tables
            .key()
            .verify(&tables.key().id(), &inputs, &proof, &vk)
            .unwrap();
        let pcs = [
            &proof.public.opening,
            &proof.sparse.dereferences.opening,
            &proof.sparse.operation_values.opening,
            &proof.sparse.audit_values.opening,
            &proof.witness_opening,
        ];
        let groups = 2 + pcs.iter().map(|p| p.com.len() + p.w.len()).sum::<usize>();
        let mut scalars = 3 + proof.public.evaluations.len() + 3 + 16 + 6 + 6 + 16 + 2 + 1;
        scalars += pcs
            .iter()
            .flat_map(|p| p.v.iter())
            .map(Vec::len)
            .sum::<usize>();
        for sc in [&proof.outer, &proof.inner] {
            scalars += sc
                .round_polynomials
                .iter()
                .map(|p| p.coeffs_except_linear_term().len())
                .sum::<usize>();
        }
        for layer in proof
            .sparse
            .operations
            .layers
            .iter()
            .chain(&proof.sparse.memory.layers)
        {
            scalars += layer
                .sumcheck
                .round_polynomials
                .iter()
                .map(|p| p.coeffs_except_linear_term().len())
                .sum::<usize>()
                + 2 * layer.ends.len()
                + 3 * layer.dot_ends.len();
        }
        let bytes = bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        assert_eq!(
            transcript.state(),
            [
                39, 228, 17, 152, 138, 110, 135, 28, 57, 155, 111, 51, 150, 113, 78, 142, 225, 11,
                64, 1, 223, 137, 255, 39, 65, 61, 149, 117, 234, 53, 240, 249
            ]
        );
        assert_eq!(
            transcript.challenge().to_bytes_le_vec(),
            [
                196, 45, 185, 169, 164, 73, 125, 50, 196, 184, 92, 121, 114, 144, 122, 255, 127,
                184, 39, 111, 80, 229, 98, 165, 220, 159, 75, 101, 34, 200, 201, 9
            ]
        );
        assert_eq!((scalars, groups, bytes.len()), (248, 36, 9176));
    }
}
