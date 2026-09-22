//! Conditional SPARK product/memory reductions; not a joint-extraction theorem.
use jolt_crypto::Bn254G1;
use jolt_field::{Fr, One, Zero};
use jolt_hyperkzg::{HyperKZGProof, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::CommitmentScheme;
use jolt_poly::{EqPolynomial, IdentityPolynomial, MultilinearEvaluation};
use jolt_sumcheck::{
    BooleanHypercube, CompressedSumcheckProof, SumcheckClaim, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{Bn254WideBlake2bTranscript, Label, Transcript, U64Word};
use serde::{Deserialize, Serialize};

use super::{ComputationKey, MatrixError};

#[derive(Clone, Copy)]
pub enum Network {
    Operations,
    Memory,
}
impl Network {
    pub fn width(self) -> usize {
        match self {
            Self::Operations => 12,
            Self::Memory => 4,
        }
    }
    pub fn begin(self, transcript: &mut Bn254WideBlake2bTranscript) {
        transcript.append(&Label(match self {
            Self::Operations => b"product-ops-v2",
            Self::Memory => b"product-mem-v2",
        }));
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProductLayerProof {
    pub sumcheck: CompressedSumcheckProof<Fr>,
    pub ends: Vec<[Fr; 2]>,
    pub dot_ends: Vec<[Fr; 3]>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProductNetworkProof {
    pub layers: Vec<ProductLayerProof>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlabOpening {
    pub evaluations: Vec<Fr>,
    pub opening: HyperKZGProof,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseMatrixProof {
    pub dereference_commitment: Bn254G1,
    pub roots: [Fr; 16],
    pub dot_halves: [Fr; 6],
    pub operations: ProductNetworkProof,
    pub memory: ProductNetworkProof,
    pub dereferences: SlabOpening,
    pub operation_values: SlabOpening,
    pub audit_values: SlabOpening,
}

/// A query in private-column coordinates; rows and columns use MSB-first MLEs.
pub struct SparseQuery<'a> {
    pub rows: &'a [Fr],
    pub columns: &'a [Fr],
    pub values: [Fr; 3],
}

pub struct NetworkResult {
    pub point: Vec<Fr>,
    pub tree_evaluations: Vec<Fr>,
    pub dot_evaluations: Option<[[Fr; 3]; 3]>,
}

/// Shared prover/verifier transcript and scalar checks for one fixed product network.
pub struct ProductReduction {
    network: Network,
    layers: usize,
    claims: Vec<Fr>,
    point: Vec<Fr>,
    dot_halves: Option<[Fr; 6]>,
    dot_evaluations: Option<[[Fr; 3]; 3]>,
}
impl ProductReduction {
    pub fn new(
        network: Network,
        layers: usize,
        claims: Vec<Fr>,
        dot_halves: Option<[Fr; 6]>,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<Self, MatrixError> {
        if layers == 0
            || claims.len() != network.width()
            || matches!(network, Network::Operations) != dot_halves.is_some()
        {
            return Err(MatrixError::Shape);
        }
        network.begin(transcript);
        Ok(Self {
            network,
            layers,
            claims,
            point: Vec::new(),
            dot_halves,
            dot_evaluations: None,
        })
    }
    pub fn point(&self) -> &[Fr] {
        &self.point
    }
    pub fn begin_layer(
        &self,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<(Vec<Fr>, Fr), MatrixError> {
        let layer = self.point.len();
        if layer >= self.layers {
            return Err(MatrixError::Shape);
        }
        transcript.append_labeled(b"product-layer", &U64Word(layer as u64));
        transcript.append(&Label(b"layer-weights"));
        let bottom = layer + 1 == self.layers;
        let dot = self.dot_halves.filter(|_| bottom);
        let weights = transcript.challenge_vector(self.claims.len() + dot.map_or(0, |_| 6));
        let claim = weights
            .iter()
            .zip(self.claims.iter().copied().chain(dot.into_iter().flatten()))
            .map(|(w, c)| *w * c)
            .sum();
        Ok((weights, claim))
    }
    pub fn finish_layer(
        &mut self,
        weights: &[Fr],
        point: &[Fr],
        final_claim: Fr,
        ends: &[[Fr; 2]],
        dot_ends: &[[Fr; 3]],
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<(), MatrixError> {
        let layer = self.point.len();
        let dot_count = if layer + 1 == self.layers && self.dot_halves.is_some() {
            6
        } else {
            0
        };
        if layer >= self.layers
            || point.len() != layer
            || ends.len() != self.network.width()
            || dot_ends.len() != dot_count
            || weights.len() != ends.len() + dot_count
        {
            return Err(MatrixError::Shape);
        }
        let eq = EqPolynomial::new(self.point.clone()).evaluate(point);
        let expected = weights
            .iter()
            .take(ends.len())
            .zip(ends)
            .map(|(w, [a, b])| *w * eq * a * b)
            .sum::<Fr>()
            + weights
                .iter()
                .skip(ends.len())
                .zip(dot_ends)
                .map(|(w, [a, b, c])| *w * a * b * c)
                .sum::<Fr>();
        if final_claim != expected {
            return Err(MatrixError::Relation("product layer terminal"));
        }
        transcript.append_values(
            b"product-ends",
            &ends.iter().flatten().copied().collect::<Vec<_>>(),
        );
        if dot_count != 0 {
            transcript.append_values(
                b"dot-ends",
                &dot_ends.iter().flatten().copied().collect::<Vec<_>>(),
            );
        }
        transcript.append(&Label(b"layer-eta"));
        let eta: Fr = transcript.challenge();
        self.claims = ends
            .iter()
            .map(|[a, b]| (Fr::one() - eta) * a + eta * b)
            .collect();
        self.point = std::iter::once(eta).chain(point.iter().copied()).collect();
        if dot_count != 0 {
            let mut values = [[Fr::zero(); 3]; 3];
            for (value, halves) in values.iter_mut().zip(dot_ends.chunks_exact(2)) {
                let [left, right] = halves else {
                    return Err(MatrixError::Shape);
                };
                for (v, (a, b)) in value.iter_mut().zip(left.iter().zip(right)) {
                    *v = (Fr::one() - eta) * a + eta * b;
                }
            }
            self.dot_evaluations = Some(values);
        }
        Ok(())
    }
    pub fn finish(self) -> Result<NetworkResult, MatrixError> {
        if self.point.len() != self.layers {
            return Err(MatrixError::Shape);
        }
        Ok(NetworkResult {
            point: self.point,
            tree_evaluations: self.claims,
            dot_evaluations: self.dot_evaluations,
        })
    }
    /// Verifies all key-fixed layers, including zero-round terminal checks.
    pub fn verify(
        mut self,
        proof: &ProductNetworkProof,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<NetworkResult, MatrixError> {
        if proof.layers.len() != self.layers {
            return Err(MatrixError::Shape);
        }
        for (j, layer) in proof.layers.iter().enumerate() {
            let (weights, claim) = self.begin_layer(transcript)?;
            let result = layer.sumcheck.verify(
                &SumcheckClaim {
                    num_vars: j,
                    degree: 3,
                    claimed_sum: claim,
                },
                BooleanHypercube,
                SUMCHECK_ROUND_TRANSCRIPT_LABEL,
                transcript,
            )?;
            self.finish_layer(
                &weights,
                result.point.as_slice(),
                result.value,
                &layer.ends,
                &layer.dot_ends,
                transcript,
            )?;
        }
        self.finish()
    }
}

impl ComputationKey {
    /// Extends the shorter query with leading zeros before memory dereferencing.
    pub fn sparse_points(
        &self,
        query: &SparseQuery<'_>,
    ) -> Result<(Vec<Fr>, Vec<Fr>), MatrixError> {
        let rows = self.shape.padded_rows.trailing_zeros() as usize;
        let cols = self.shape.padded_private.trailing_zeros() as usize;
        let memory = self.shape.memory.trailing_zeros() as usize;
        if query.rows.len() != rows || query.columns.len() != cols {
            return Err(MatrixError::Query);
        }
        Ok((
            std::iter::repeat_n(Fr::zero(), memory - rows)
                .chain(query.rows.iter().copied())
                .collect(),
            std::iter::repeat_n(Fr::zero(), memory - cols)
                .chain(query.columns.iter().copied())
                .collect(),
        ))
    }
    pub fn begin_sparse(
        &self,
        query: &SparseQuery<'_>,
        commitment: &Bn254G1,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<(Fr, Fr), MatrixError> {
        let _ = self.sparse_points(query)?;
        transcript.append(&Label(b"matrix-query-v2"));
        transcript.append_values(b"matrix-x", query.rows);
        transcript.append_values(b"matrix-y", query.columns);
        transcript.append_values(b"matrix-values", &query.values);
        transcript.append(&Label(b"matrix-derefs"));
        transcript.append(commitment);
        transcript.append(&Label(b"memory-hash"));
        Ok((transcript.challenge(), transcript.challenge()))
    }
    pub fn bind_roots(
        &self,
        values: [Fr; 3],
        roots: &[Fr; 16],
        halves: &[Fr; 6],
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<(Vec<Fr>, Vec<Fr>), MatrixError> {
        for axis in roots.chunks_exact(8) {
            let [init, ra, rb, rc, wa, wb, wc, audit] = axis else {
                return Err(MatrixError::Shape);
            };
            if *init * wa * wb * wc != *ra * rb * rc * audit {
                return Err(MatrixError::Relation("memory roots"));
            }
        }
        for (value, pair) in values.iter().zip(halves.chunks_exact(2)) {
            if pair.iter().copied().sum::<Fr>() != *value {
                return Err(MatrixError::Relation("matrix half sums"));
            }
        }
        transcript.append_values(b"memory-roots", roots);
        transcript.append_values(b"dot-halves", halves);
        let mut operations = Vec::with_capacity(12);
        let mut memory = Vec::with_capacity(4);
        for axis in roots.chunks_exact(8) {
            let [init, ra, rb, rc, wa, wb, wc, audit] = axis else {
                return Err(MatrixError::Shape);
            };
            operations.extend([*ra, *rb, *rc, *wa, *wb, *wc]);
            memory.extend([*init, *audit]);
        }
        Ok((operations, memory))
    }
    /// Batch selector prefix after exact ordered claims. Known zero slabs are included.
    pub fn slab_query(
        evaluations: &[Fr],
        point: &[Fr],
        phase: Slab,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<(Vec<Fr>, Fr), MatrixError> {
        let (count, padded, label, selector) = match phase {
            Slab::Dereferences => (
                6,
                8,
                b"deref-evals".as_slice(),
                b"deref-selector".as_slice(),
            ),
            Slab::Operations => (16, 16, b"ops-evals".as_slice(), b"ops-selector".as_slice()),
            Slab::Audit => (
                2,
                2,
                b"audit-evals".as_slice(),
                b"audit-selector".as_slice(),
            ),
        };
        if evaluations.len() != count
            || matches!(phase, Slab::Operations) && evaluations.last() != Some(&Fr::zero())
        {
            return Err(MatrixError::Shape);
        }
        transcript.append_values(label, evaluations);
        transcript.append(&Label(selector));
        let mut query = transcript.challenge_vector((padded as usize).trailing_zeros() as usize);
        let weights = EqPolynomial::new(query.clone()).evaluations();
        let value = weights.iter().zip(evaluations).map(|(a, b)| *a * b).sum();
        query.extend_from_slice(point);
        Ok((query, value))
    }
    fn verify_slab(
        commitment: &Bn254G1,
        proof: &SlabOpening,
        point: &[Fr],
        phase: Slab,
        setup: &HyperKZGVerifierSetup,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<(), MatrixError> {
        let (point, value) = Self::slab_query(&proof.evaluations, point, phase, transcript)?;
        HyperKZGScheme::verify(commitment, &point, value, &proof.opening, setup, transcript)?;
        Ok(())
    }
    /// Authenticates private matrix evaluations for this application-authenticated
    /// key. Full relation acceptance must use `verify`, which owns the query
    /// derivation and transcript. Conditional on the stated PCS/ROM gates.
    #[expect(
        clippy::indexing_slicing,
        reason = "network and slab dimensions are checked before fixed-index terminal equations"
    )]
    pub fn verify_sparse(
        &self,
        query: SparseQuery<'_>,
        proof: &SparseMatrixProof,
        setup: &HyperKZGVerifierSetup,
        transcript: &mut Bn254WideBlake2bTranscript,
    ) -> Result<(), MatrixError> {
        self.authenticate(&self.id(), setup)?;
        let (x, y) = self.sparse_points(&query)?;
        let (alpha, beta) = self.begin_sparse(&query, &proof.dereference_commitment, transcript)?;
        let (ops, mem) =
            self.bind_roots(query.values, &proof.roots, &proof.dot_halves, transcript)?;
        let ops = ProductReduction::new(
            Network::Operations,
            self.shape.operations.trailing_zeros() as usize,
            ops,
            Some(proof.dot_halves),
            transcript,
        )?
        .verify(&proof.operations, transcript)?;
        let mem = ProductReduction::new(
            Network::Memory,
            self.shape.memory.trailing_zeros() as usize,
            mem,
            None,
            transcript,
        )?
        .verify(&proof.memory, transcript)?;
        Self::verify_slab(
            &proof.dereference_commitment,
            &proof.dereferences,
            &ops.point,
            Slab::Dereferences,
            setup,
            transcript,
        )?;
        Self::verify_slab(
            &self.commitments[1],
            &proof.operation_values,
            &ops.point,
            Slab::Operations,
            setup,
            transcript,
        )?;
        Self::verify_slab(
            &self.commitments[2],
            &proof.audit_values,
            &mem.point,
            Slab::Audit,
            setup,
            transcript,
        )?;
        let dots = ops.dot_evaluations.ok_or(MatrixError::Shape)?;
        let dr = &proof.dereferences.evaluations;
        let op = &proof.operation_values.evaluations;
        let alpha2 = alpha * alpha;
        for matrix in 0..3 {
            if dots[matrix] != [dr[matrix], dr[3 + matrix], op[12 + matrix]] {
                return Err(MatrixError::Relation("dot leaf"));
            }
            for axis in 0..2 {
                let read = op[axis * 6 + matrix]
                    + alpha * dr[axis * 3 + matrix]
                    + alpha2 * op[axis * 6 + 3 + matrix]
                    - beta;
                if ops.tree_evaluations[axis * 6 + matrix] != read
                    || ops.tree_evaluations[axis * 6 + 3 + matrix] != read + alpha2
                {
                    return Err(MatrixError::Relation("operation hash leaf"));
                }
            }
        }
        let id = IdentityPolynomial::new(mem.point.len()).evaluate(&mem.point);
        for (axis, point) in [&x, &y].into_iter().enumerate() {
            let init = id + alpha * EqPolynomial::new(point.clone()).evaluate(&mem.point) - beta;
            if mem.tree_evaluations[axis * 2] != init
                || mem.tree_evaluations[axis * 2 + 1]
                    != init + alpha2 * proof.audit_values.evaluations[axis]
            {
                return Err(MatrixError::Relation("memory hash leaf"));
            }
        }
        Ok(())
    }
}
#[derive(Clone, Copy)]
pub enum Slab {
    Dereferences,
    Operations,
    Audit,
}
