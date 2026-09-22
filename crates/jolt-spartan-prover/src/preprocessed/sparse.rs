//! Reference-oriented honest prover for the pinned private SPARK reductions.
use jolt_crypto::Bn254G1;
use jolt_field::{Fr, Ring, Zero};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme};
use jolt_openings::CommitmentScheme;
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_spartan_verifier::preprocessed::sparse::{
    Network, NetworkResult, ProductLayerProof, ProductNetworkProof, ProductReduction, Slab,
    SlabOpening, SparseMatrixProof, SparseQuery,
};
use jolt_spartan_verifier::preprocessed::{ComputationKey, MatrixError};
use jolt_sumcheck::{CompressedSumcheckProof, ProveRounds, SumcheckError};
use jolt_transcript::Bn254WideBlake2bTranscript;

use super::PreprocessedMatrices;
use crate::prove_rounds;

struct ProductRounds {
    terms: Vec<[Polynomial<Fr>; 3]>,
    weights: Vec<Fr>,
    rounds: usize,
}
impl ProductRounds {
    fn bind(&mut self, r: Fr) {
        for term in &mut self.terms {
            for poly in term {
                poly.bind_with_order(r, BindingOrder::HighToLow);
            }
        }
    }
    fn ends(&self) -> Result<Vec<[Fr; 3]>, MatrixError> {
        self.terms
            .iter()
            .map(|[a, b, c]| {
                Ok([
                    *a.evals().first().ok_or(MatrixError::Shape)?,
                    *b.evals().first().ok_or(MatrixError::Shape)?,
                    *c.evals().first().ok_or(MatrixError::Shape)?,
                ])
            })
            .collect()
    }
}
impl ProveRounds<Fr> for ProductRounds {
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        _round: usize,
        _claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        let mut evaluations = [Fr::zero(); 4];
        for (weight, [a, b, c]) in self.weights.iter().zip(&self.terms) {
            for i in 0..a.len() / 2 {
                for (x, value) in evaluations.iter_mut().enumerate() {
                    let x = Fr::from_u64(x as u64);
                    *value += *weight
                        * a.sumcheck_round_eval(i, x)
                        * b.sumcheck_round_eval(i, x)
                        * c.sumcheck_round_eval(i, x);
                }
            }
        }
        Ok(UnivariatePoly::from_evals(&evaluations))
    }
    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}

struct ProductNetwork {
    trees: Vec<Vec<Vec<Fr>>>,
    dots: Option<Vec<[Vec<Fr>; 3]>>,
}
impl ProductNetwork {
    fn new(leaves: Vec<Vec<Fr>>, dots: Option<Vec<[Vec<Fr>; 3]>>) -> Result<Self, MatrixError> {
        let length = leaves.first().ok_or(MatrixError::Shape)?.len();
        if length < 2 || !length.is_power_of_two() || leaves.iter().any(|x| x.len() != length) {
            return Err(MatrixError::Shape);
        }
        if let Some(dots) = &dots {
            if dots.len() != 6 || dots.iter().flatten().any(|x| x.len() != length / 2) {
                return Err(MatrixError::Shape);
            }
        }
        let trees = leaves
            .into_iter()
            .map(|leaf| {
                let mut layers = vec![leaf];
                while let Some(previous) = layers.last() {
                    if previous.len() == 1 {
                        break;
                    }
                    let (a, b) = previous.split_at(previous.len() / 2);
                    let next = a.iter().zip(b).map(|(a, b)| *a * b).collect();
                    layers.push(next);
                }
                layers.reverse();
                layers
            })
            .collect();
        Ok(Self { trees, dots })
    }
    fn roots(&self) -> Result<Vec<Fr>, MatrixError> {
        self.trees
            .iter()
            .map(|tree| {
                tree.first()
                    .and_then(|x| x.first())
                    .copied()
                    .ok_or(MatrixError::Shape)
            })
            .collect()
    }
    fn prove(
        &self,
        network: Network,
        halves: Option<[Fr; 6]>,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<(ProductNetworkProof, NetworkResult), MatrixError> {
        let count = self.trees.first().ok_or(MatrixError::Shape)?.len() - 1;
        let mut reduction =
            ProductReduction::new(network, count, self.roots()?, halves, transcript)?;
        let mut layers = Vec::with_capacity(count);
        for j in 0..count {
            let (weights, claim) = reduction.begin_layer(transcript)?;
            let eq = EqPolynomial::new(reduction.point().to_vec()).evaluations();
            let mut terms = Vec::with_capacity(weights.len());
            for tree in &self.trees {
                let current = tree.get(j + 1).ok_or(MatrixError::Shape)?;
                let (a, b) = current.split_at(current.len() / 2);
                terms.push([
                    Polynomial::new(eq.clone()),
                    Polynomial::new(a.to_vec()),
                    Polynomial::new(b.to_vec()),
                ]);
            }
            if j + 1 == count {
                if let Some(dots) = &self.dots {
                    for dot in dots {
                        terms.push(dot.clone().map(Polynomial::new));
                    }
                }
            }
            if terms.len() != weights.len() {
                return Err(MatrixError::Shape);
            }
            let mut rounds = ProductRounds {
                terms,
                weights: weights.clone(),
                rounds: j,
            };
            let (sumcheck, point, final_claim) = if j == 0 {
                (CompressedSumcheckProof::default(), Vec::new(), claim)
            } else {
                prove_rounds(&mut rounds, 3, claim, transcript)?
            };
            let terminal = rounds.ends()?;
            let ends = terminal
                .iter()
                .take(network.width())
                .map(|[_, a, b]| [*a, *b])
                .collect::<Vec<_>>();
            let dot_ends = terminal
                .into_iter()
                .skip(network.width())
                .collect::<Vec<_>>();
            reduction.finish_layer(&weights, &point, final_claim, &ends, &dot_ends, transcript)?;
            layers.push(ProductLayerProof {
                sumcheck,
                ends,
                dot_ends,
            });
        }
        Ok((ProductNetworkProof { layers }, reduction.finish()?))
    }
}

struct SparseWitness {
    dereference_commitment: Bn254G1,
    roots: [Fr; 16],
    halves: [Fr; 6],
    operations: ProductNetwork,
    memory: ProductNetwork,
    dereferences: Vec<Fr>,
}

impl PreprocessedMatrices {
    fn open_slab(
        table: &[Fr],
        length: usize,
        count: usize,
        point: &[Fr],
        phase: Slab,
        setup: &HyperKZGProverSetup,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<SlabOpening, MatrixError> {
        if length < 2
            || !length.is_power_of_two()
            || point.len() != length.trailing_zeros() as usize
            || !table.len().is_multiple_of(length)
            || table.len() / length < count
        {
            return Err(MatrixError::Shape);
        }
        let eq = EqPolynomial::new(point.to_vec()).evaluations();
        let evaluations = table
            .chunks_exact(length)
            .take(count)
            .map(|slab| slab.iter().zip(&eq).map(|(a, b)| *a * b).sum())
            .collect::<Vec<_>>();
        let (query, value) = ComputationKey::slab_query(&evaluations, point, phase, transcript)?;
        let opening = HyperKZGScheme::open(
            &Polynomial::new(table.to_vec()),
            &query,
            value,
            setup,
            None,
            transcript,
        )?;
        Ok(SlabOpening {
            evaluations,
            opening,
        })
    }
    #[expect(
        clippy::indexing_slicing,
        reason = "private tables and address lists originate from checked preprocessing; fixed-array and slab offsets follow key dimensions"
    )]
    fn prepare_sparse(
        &self,
        query: SparseQuery<'_>,
        setup: &HyperKZGProverSetup,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<SparseWitness, MatrixError> {
        let shape = self.key.shape();
        let n = shape.operations;
        let l = shape.memory;
        let (x, y) = self.key.sparse_points(&query)?;
        let memories = [
            EqPolynomial::new(x).evaluations(),
            EqPolynomial::new(y).evaluations(),
        ];
        let mut dereferences = vec![Fr::zero(); 8 * n];
        for matrix in 0..3 {
            for (k, &(row, col)) in self.addresses[matrix].iter().enumerate() {
                dereferences[matrix * n + k] = *memories[0].get(row).ok_or(MatrixError::Shape)?;
                dereferences[(3 + matrix) * n + k] =
                    *memories[1].get(col).ok_or(MatrixError::Shape)?;
            }
        }
        let dereference_commitment =
            HyperKZGScheme::commit(&Polynomial::new(dereferences.clone()), setup)?.0;
        let (alpha, beta) = self
            .key
            .begin_sparse(&query, &dereference_commitment, transcript)?;
        let alpha2 = alpha * alpha;
        let mut operations = Vec::with_capacity(12);
        let mut memory = Vec::with_capacity(4);
        for (axis, mu) in memories.iter().enumerate() {
            let mut reads = Vec::with_capacity(3);
            let mut writes = Vec::with_capacity(3);
            for matrix in 0..3 {
                let read = (0..n)
                    .map(|k| {
                        self.operations[(axis * 6 + matrix) * n + k]
                            + alpha * dereferences[(axis * 3 + matrix) * n + k]
                            + alpha2 * self.operations[(axis * 6 + 3 + matrix) * n + k]
                            - beta
                    })
                    .collect::<Vec<_>>();
                writes.push(read.iter().map(|value| *value + alpha2).collect());
                reads.push(read);
            }
            operations.extend(reads);
            operations.extend(writes);
            let init = mu
                .iter()
                .enumerate()
                .map(|(i, v)| Fr::from_u64(i as u64) + alpha * v - beta)
                .collect::<Vec<_>>();
            let audit = init
                .iter()
                .enumerate()
                .map(|(i, v)| *v + alpha2 * self.memory[axis * l + i])
                .collect();
            memory.push(init);
            memory.push(audit);
        }
        let mut dots = Vec::with_capacity(6);
        let mut halves = [Fr::zero(); 6];
        for matrix in 0..3 {
            for half in 0..2 {
                let begin = half * n / 2;
                let end = begin + n / 2;
                let triple = [
                    dereferences[matrix * n + begin..matrix * n + end].to_vec(),
                    dereferences[(3 + matrix) * n + begin..(3 + matrix) * n + end].to_vec(),
                    self.operations[(12 + matrix) * n + begin..(12 + matrix) * n + end].to_vec(),
                ];
                halves[2 * matrix + half] = triple[0]
                    .iter()
                    .zip(&triple[1])
                    .zip(&triple[2])
                    .map(|((a, b), c)| *a * b * c)
                    .sum();
                dots.push(triple);
            }
        }
        let operations = ProductNetwork::new(operations, Some(dots))?;
        let memory = ProductNetwork::new(memory, None)?;
        let ops_roots = operations.roots()?;
        let mem_roots = memory.roots()?;
        let mut roots = [Fr::zero(); 16];
        for axis in 0..2 {
            roots[axis * 8] = mem_roots[axis * 2];
            roots[axis * 8 + 7] = mem_roots[axis * 2 + 1];
            roots[axis * 8 + 1..axis * 8 + 7].copy_from_slice(&ops_roots[axis * 6..axis * 6 + 6]);
        }
        Ok(SparseWitness {
            dereference_commitment,
            roots,
            halves,
            operations,
            memory,
            dereferences,
        })
    }

    pub fn prove_sparse(
        &self,
        query: SparseQuery<'_>,
        setup: &HyperKZGProverSetup,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<SparseMatrixProof, MatrixError> {
        let witness = self.prepare_sparse(
            SparseQuery {
                rows: query.rows,
                columns: query.columns,
                values: query.values,
            },
            setup,
            transcript,
        )?;
        let _ = self
            .key
            .bind_roots(query.values, &witness.roots, &witness.halves, transcript)?;
        self.finish_sparse(witness, setup, transcript)
    }

    fn finish_sparse(
        &self,
        witness: SparseWitness,
        setup: &HyperKZGProverSetup,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<SparseMatrixProof, MatrixError> {
        let SparseWitness {
            dereference_commitment,
            roots,
            halves,
            operations,
            memory,
            dereferences,
        } = witness;
        let n = self.key.shape().operations;
        let l = self.key.shape().memory;

        let (operations, ops) = operations.prove(Network::Operations, Some(halves), transcript)?;
        let (memory, mem) = memory.prove(Network::Memory, None, transcript)?;
        let dereferences = Self::open_slab(
            &dereferences,
            n,
            6,
            &ops.point,
            Slab::Dereferences,
            setup,
            transcript,
        )?;
        let operation_values = Self::open_slab(
            &self.operations,
            n,
            16,
            &ops.point,
            Slab::Operations,
            setup,
            transcript,
        )?;
        let audit_values = Self::open_slab(
            &self.memory,
            l,
            2,
            &mem.point,
            Slab::Audit,
            setup,
            transcript,
        )?;
        Ok(SparseMatrixProof {
            dereference_commitment,
            roots,
            dot_halves: halves,
            operations,
            memory,
            dereferences,
            operation_values,
            audit_values,
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "bounded adversarial fixtures fail loudly"
)]
mod tests {
    use super::super::tests::{ids, relation, setup};
    use super::*;
    use jolt_field::One;
    use jolt_transcript::Transcript;

    #[test]
    fn product_network_accepts_zero_factors_without_division() {
        let leaves = vec![
            vec![0, 2, 3, 4],
            vec![5, 0, 7, 8],
            vec![9, 10, 0, 12],
            vec![13, 14, 15, 0],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(Fr::from_u64).collect())
        .collect::<Vec<Vec<_>>>();
        let network = ProductNetwork::new(leaves.clone(), None).unwrap();
        assert_eq!(network.roots().unwrap(), vec![Fr::zero(); 4]);
        let mut pt = Bn254WideBlake2bTranscript::new(b"zero-product-control");
        let (proof, result) = network.prove(Network::Memory, None, &mut pt).unwrap();
        let mut vt = Bn254WideBlake2bTranscript::new(b"zero-product-control");
        let checked = ProductReduction::new(Network::Memory, 2, vec![Fr::zero(); 4], None, &mut vt)
            .unwrap()
            .verify(&proof, &mut vt)
            .unwrap();
        assert_eq!(pt.state(), vt.state());
        assert_eq!(result.point, checked.point);
        let eq = EqPolynomial::new(checked.point).evaluations();
        let expected = leaves
            .iter()
            .map(|v| v.iter().zip(&eq).map(|(a, b)| *a * b).sum::<Fr>())
            .collect::<Vec<_>>();
        assert_eq!(checked.tree_evaluations, expected);
    }

    #[test]
    fn coherently_recommitted_reset_timestamps_and_unshifted_addresses_fail_algebra() {
        let (pk, vk) = setup();
        let x = [2, 3].map(Fr::from_u64);
        let y = [4, 5, 6].map(Fr::from_u64);
        let direct = relation();
        let rw = EqPolynomial::new(x.to_vec()).evaluations();
        let cw = EqPolynomial::new(y.to_vec()).evaluations();
        let mut values = [Fr::zero(); 3];
        for (i, v) in values.iter_mut().enumerate() {
            let mut w = [Fr::zero(); 3];
            w[i] = Fr::one();
            *v = direct
                .matrices()
                .linear_form_bilinear_eval(&rw, &cw[..5], 3, 5, w)
                .unwrap();
        }
        for reset in [true, false] {
            let mut tables = PreprocessedMatrices::new(&direct, ids(), &pk).unwrap();
            let original_id = tables.key.id();
            let shape = tables.key.shape();
            let n = shape.operations;
            let l = shape.memory;
            let mut counters = [vec![0u64; l], vec![0u64; l]];
            for matrix in 0..3 {
                if reset {
                    for axis in &mut counters {
                        axis.fill(0);
                    }
                }
                for k in 0..n {
                    if !reset {
                        tables.addresses[matrix][k].1 += shape.public_columns;
                    }
                    let (row, col) = tables.addresses[matrix][k];
                    for (axis, address) in [row, col].into_iter().enumerate() {
                        tables.operations[(axis * 6 + matrix) * n + k] =
                            Fr::from_u64(address as u64);
                        tables.operations[(axis * 6 + 3 + matrix) * n + k] =
                            Fr::from_u64(counters[axis][address]);
                        counters[axis][address] += 1;
                    }
                }
            }
            tables.memory = counters.into_iter().flatten().map(Fr::from_u64).collect();
            let commitments = [&tables.public, &tables.operations, &tables.memory].map(|table| {
                HyperKZGScheme::commit(&Polynomial::new(table.clone()), &pk)
                    .unwrap()
                    .0
            });
            // This is deliberately invalid trusted preprocessing under a NEW key,
            // not a claim of replacing an authenticated key in an application.
            tables.key = ComputationKey::new(shape, ids(), [66; 32], commitments, &vk).unwrap();
            assert_ne!(tables.key.id(), original_id);
            let mut pt = Bn254WideBlake2bTranscript::new(b"coherent-memory-control");
            let query = || SparseQuery {
                rows: &x,
                columns: &y,
                values,
            };
            let witness = tables.prepare_sparse(query(), &pk, &mut pt).unwrap();
            // Malicious prover skips its local assertion but uses the unchanged
            // wire order and real downstream product proofs and PCS openings.
            pt.append_values(b"memory-roots", &witness.roots);
            pt.append_values(b"dot-halves", &witness.halves);
            let proof = tables.finish_sparse(witness, &pk, &mut pt).unwrap();
            let mut vt = Bn254WideBlake2bTranscript::new(b"coherent-memory-control");
            let error = tables
                .key
                .verify_sparse(query(), &proof, &vk, &mut vt)
                .unwrap_err();
            assert!(
                matches!(error,MatrixError::Relation(message) if message==if reset {"memory roots"} else {"matrix half sums"})
            );
        }
    }
}
