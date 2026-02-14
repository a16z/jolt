//! Relaxed R1CS for BlindFold Protocol
//!
//! Standard R1CS: (A·Z) ∘ (B·Z) = C·Z
//!
//! This doesn't fold nicely due to cross-terms. Relaxed R1CS:
//!   (A·Z) ∘ (B·Z) = u·(C·Z) + E
//!
//! Where:
//! - u ∈ F is a scalar (u=1 for non-relaxed)
//! - E ∈ F^m is an error vector (E=0 for non-relaxed)
//!
//! This allows folding two satisfying instances into one.
//!
//! Instance and witness use row-based commitments for Hyrax-style opening:
//! - W is arranged as an R' × C grid; coefficient rows reuse sumcheck round commitments
//! - E is arranged as an R_E × C_E grid; row commitments enable Hyrax opening at rx

use crate::curve::{JoltCurve, JoltGroupElement};
use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use super::protocol::BlindFoldVerifyError;
use super::r1cs::VerifierR1CS;

/// Relaxed R1CS Instance (public data)
///
/// Public inputs are baked into R1CS matrix coefficients, so there is no `x` field.
/// Row commitments replace monolithic W_bar and E_bar:
/// - `round_commitments`: coefficient row commitments (reuse existing sumcheck round commitments)
/// - `noncoeff_row_commitments`: non-coefficient row commitments (prover sends in proof)
/// - `e_row_commitments`: E row commitments (derived from cross-term and random instance)
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct RelaxedR1CSInstance<F: JoltField, C: JoltCurve> {
    pub u: F,
    /// Per-round commitments from ZK sumcheck (= coefficient row commitments)
    pub round_commitments: Vec<C::G1>,
    /// Non-coefficient W row commitments
    pub noncoeff_row_commitments: Vec<C::G1>,
    /// E row commitments
    pub e_row_commitments: Vec<C::G1>,
    /// Evaluation commitments for extra constraints
    pub eval_commitments: Vec<C::G1>,
}

/// Relaxed R1CS Witness (private data)
///
/// W is in grid layout (R' × C). Row blindings cover all W rows.
/// E is flat but has per-row blindings for Hyrax opening.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct RelaxedR1CSWitness<F: JoltField> {
    /// Error vector (zeros for non-relaxed)
    pub E: Vec<F>,
    /// Witness values in grid layout (R' × C)
    pub W: Vec<F>,
    /// One blinding per W row (R' elements: coefficient row blindings, then non-coeff, then padding zeros)
    pub w_row_blindings: Vec<F>,
    /// One blinding per E row (R_E elements)
    pub e_row_blindings: Vec<F>,
}

impl<F: JoltField, C: JoltCurve> RelaxedR1CSInstance<F, C> {
    /// Create a non-relaxed instance (u=1, E=0) from standard R1CS witness.
    pub fn new_non_relaxed(
        witness: &[F],
        num_constraints: usize,
        hyrax_C: usize,
        round_commitments: Vec<C::G1>,
        noncoeff_row_commitments: Vec<C::G1>,
        eval_commitments: Vec<C::G1>,
        w_row_blindings: Vec<F>,
    ) -> (Self, RelaxedR1CSWitness<F>) {
        let (R_E, _C_E) = super::HyraxParams::e_grid_static(hyrax_C, num_constraints);

        let instance = Self {
            u: F::one(),
            round_commitments,
            noncoeff_row_commitments,
            e_row_commitments: vec![C::G1::zero(); R_E],
            eval_commitments,
        };

        let witness_struct = RelaxedR1CSWitness {
            E: vec![F::zero(); num_constraints],
            W: witness.to_vec(),
            w_row_blindings,
            e_row_blindings: vec![F::zero(); R_E],
        };

        (instance, witness_struct)
    }

    /// Fold two instances into one.
    ///
    /// E row commitments fold with cross-term T row commitments:
    ///   e_row_i' = e_row_i_1 + r·t_row_i + r²·e_row_i_2
    pub fn fold(
        &self,
        other: &Self,
        t_row_commitments: &[C::G1],
        r: F,
    ) -> Result<Self, BlindFoldVerifyError> {
        if self.round_commitments.len() != other.round_commitments.len()
            || self.noncoeff_row_commitments.len() != other.noncoeff_row_commitments.len()
            || self.eval_commitments.len() != other.eval_commitments.len()
            || self.e_row_commitments.len() != other.e_row_commitments.len()
            || self.e_row_commitments.len() != t_row_commitments.len()
        {
            return Err(BlindFoldVerifyError::MalformedProof);
        }

        let r_squared = r * r;
        let u = self.u + r * other.u;

        let round_commitments: Vec<C::G1> = self
            .round_commitments
            .iter()
            .zip(&other.round_commitments)
            .map(|(c1, c2)| *c1 + c2.scalar_mul(&r))
            .collect();

        let noncoeff_row_commitments: Vec<C::G1> = self
            .noncoeff_row_commitments
            .iter()
            .zip(&other.noncoeff_row_commitments)
            .map(|(c1, c2)| *c1 + c2.scalar_mul(&r))
            .collect();

        let e_row_commitments: Vec<C::G1> = self
            .e_row_commitments
            .iter()
            .zip(t_row_commitments)
            .zip(&other.e_row_commitments)
            .map(|((e1, t), e2)| *e1 + t.scalar_mul(&r) + e2.scalar_mul(&r_squared))
            .collect();

        let eval_commitments: Vec<C::G1> = self
            .eval_commitments
            .iter()
            .zip(&other.eval_commitments)
            .map(|(c1, c2)| *c1 + c2.scalar_mul(&r))
            .collect();

        Ok(Self {
            u,
            round_commitments,
            noncoeff_row_commitments,
            e_row_commitments,
            eval_commitments,
        })
    }

    /// All W row commitments in order: coefficient rows (padded to R_coeff), then non-coeff rows,
    /// then padding to R'.
    pub fn all_w_row_commitments(
        &self,
        total_rounds: usize,
        R_coeff: usize,
        R_prime: usize,
    ) -> Vec<C::G1> {
        let mut rows = Vec::with_capacity(R_prime);
        rows.extend_from_slice(&self.round_commitments);
        for _ in total_rounds..R_coeff {
            rows.push(C::G1::zero());
        }
        rows.extend_from_slice(&self.noncoeff_row_commitments);
        while rows.len() < R_prime {
            rows.push(C::G1::zero());
        }
        rows
    }
}

impl<F: JoltField> RelaxedR1CSWitness<F> {
    /// Fold two witnesses into one.
    ///
    /// E folds with cross-term: E' = E₁ + r·T + r²·E₂
    /// W folds linearly: W' = W₁ + r·W₂
    pub fn fold(&self, other: &Self, T: &[F], t_row_blindings: &[F], r: F) -> Self {
        let r_squared = r * r;

        let E: Vec<F> = self
            .E
            .iter()
            .zip(T.iter())
            .zip(&other.E)
            .map(|((e1, t), e2)| *e1 + r * *t + r_squared * *e2)
            .collect();

        let W: Vec<F> = self
            .W
            .iter()
            .zip(&other.W)
            .map(|(w1, w2)| *w1 + r * *w2)
            .collect();

        let w_row_blindings: Vec<F> = self
            .w_row_blindings
            .iter()
            .zip(&other.w_row_blindings)
            .map(|(b1, b2)| *b1 + r * *b2)
            .collect();

        let e_row_blindings: Vec<F> = self
            .e_row_blindings
            .iter()
            .zip(t_row_blindings)
            .zip(&other.e_row_blindings)
            .map(|((b1, tb), b2)| *b1 + r * *tb + r_squared * *b2)
            .collect();

        Self {
            E,
            W,
            w_row_blindings,
            e_row_blindings,
        }
    }

    /// Check if the witness satisfies the relaxed R1CS: (AZ)∘(BZ) = u·(CZ) + E
    /// Z = [u, W...]
    pub fn check_satisfaction(&self, r1cs: &VerifierR1CS<F>, u: F) -> Result<(), usize> {
        let mut z = Vec::with_capacity(r1cs.num_vars);
        z.push(u);
        z.extend_from_slice(&self.W);

        assert_eq!(
            z.len(),
            r1cs.num_vars,
            "Z vector size mismatch: {} vs {}",
            z.len(),
            r1cs.num_vars
        );

        let az = r1cs.a.mul_vector(&z);
        let bz = r1cs.b.mul_vector(&z);
        let cz = r1cs.c.mul_vector(&z);

        for i in 0..r1cs.num_constraints {
            let lhs = az[i] * bz[i];
            let rhs = u * cz[i] + self.E[i];
            if lhs != rhs {
                return Err(i);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::Bn254Curve;
    use crate::poly::commitment::pedersen::PedersenGenerators;
    use crate::subprotocols::blindfold::r1cs::VerifierR1CSBuilder;
    use crate::subprotocols::blindfold::witness::{BlindFoldWitness, RoundWitness, StageWitness};
    use crate::subprotocols::blindfold::{BakedPublicInputs, StageConfig};
    use ark_bn254::Fr;
    use ark_std::{One, UniformRand, Zero};

    use rand::thread_rng;

    #[test]
    fn test_non_relaxed_instance_creation() {
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
        let round = RoundWitness::new(
            vec![
                F::from_u64(40),
                F::from_u64(5),
                F::from_u64(10),
                F::from_u64(5),
            ],
            F::from_u64(3),
        );
        let initial_claim = F::from_u64(100);
        let blindfold_witness =
            BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round])]);

        let baked = BakedPublicInputs::from_witness(&blindfold_witness, &configs);
        let builder = VerifierR1CSBuilder::<F>::new(&configs, &baked);
        let r1cs = builder.build();

        let z = blindfold_witness.assign(&r1cs);
        assert!(r1cs.is_satisfied(&z));

        // witness_start = 1 (no public inputs)
        let witness: Vec<F> = z[1..].to_vec();

        let gens = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.hyrax.C + 1);

        let hyrax_C = r1cs.hyrax.C;
        let coeffs_row = &witness[0..hyrax_C];
        let blinding = F::rand(&mut thread_rng());
        let round_commitment = gens.commit(coeffs_row, &blinding);

        let R_coeff = r1cs.hyrax.R_coeff;
        let R_prime = r1cs.hyrax.R_prime;
        let noncoeff_start = R_coeff * hyrax_C;
        let noncoeff_rows_count = r1cs.hyrax.noncoeff_count.div_ceil(hyrax_C);

        let mut noncoeff_row_commitments = Vec::new();
        let mut w_row_blindings = vec![F::zero(); R_prime];
        w_row_blindings[0] = blinding;

        let mut rng = thread_rng();
        for row in 0..noncoeff_rows_count {
            let start = noncoeff_start + row * hyrax_C;
            let end = (start + hyrax_C).min(witness.len());
            let row_data = &witness[start..end];
            let row_blinding = F::rand(&mut rng);
            noncoeff_row_commitments.push(gens.commit(row_data, &row_blinding));
            w_row_blindings[R_coeff + row] = row_blinding;
        }

        let (instance, relaxed_witness) = RelaxedR1CSInstance::<F, Bn254Curve>::new_non_relaxed(
            &witness,
            r1cs.num_constraints,
            hyrax_C,
            vec![round_commitment],
            noncoeff_row_commitments,
            Vec::new(),
            w_row_blindings,
        );

        assert_eq!(instance.u, F::one());
        assert_eq!(relaxed_witness.W, witness);
        assert!(relaxed_witness.E.iter().all(|e| e.is_zero()));
    }

    #[test]
    fn test_relaxed_satisfaction_non_relaxed() {
        let mut rng = thread_rng();
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
        let round = RoundWitness::new(
            vec![
                F::from_u64(40),
                F::from_u64(5),
                F::from_u64(10),
                F::from_u64(5),
            ],
            F::from_u64(3),
        );
        let initial_claim = F::from_u64(100);
        let blindfold_witness =
            BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round])]);

        let baked = BakedPublicInputs::from_witness(&blindfold_witness, &configs);
        let builder = VerifierR1CSBuilder::<F>::new(&configs, &baked);
        let r1cs = builder.build();

        let z = blindfold_witness.assign(&r1cs);
        let witness: Vec<F> = z[1..].to_vec();

        let relaxed_witness = RelaxedR1CSWitness {
            E: vec![F::zero(); r1cs.num_constraints],
            W: witness,
            w_row_blindings: vec![F::random(&mut rng); r1cs.hyrax.R_prime],
            e_row_blindings: Vec::new(),
        };

        let result = relaxed_witness.check_satisfaction(&r1cs, F::one());
        assert!(result.is_ok(), "Relaxed R1CS should be satisfied");
    }

    #[test]
    fn test_witness_folding() {
        let mut rng = thread_rng();
        type F = Fr;

        let n = 10;
        let m = 5;

        let w1: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let w2: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let e1: Vec<F> = (0..m).map(|_| F::rand(&mut rng)).collect();
        let e2: Vec<F> = (0..m).map(|_| F::rand(&mut rng)).collect();
        let t: Vec<F> = (0..m).map(|_| F::rand(&mut rng)).collect();

        let w_blinds1: Vec<F> = (0..4).map(|_| F::rand(&mut rng)).collect();
        let w_blinds2: Vec<F> = (0..4).map(|_| F::rand(&mut rng)).collect();

        let e_blinds1: Vec<F> = (0..2).map(|_| F::rand(&mut rng)).collect();
        let e_blinds2: Vec<F> = (0..2).map(|_| F::rand(&mut rng)).collect();
        let t_blinds: Vec<F> = (0..2).map(|_| F::rand(&mut rng)).collect();

        let wit1 = RelaxedR1CSWitness {
            E: e1.clone(),
            W: w1.clone(),
            w_row_blindings: w_blinds1.clone(),
            e_row_blindings: e_blinds1.clone(),
        };

        let wit2 = RelaxedR1CSWitness {
            E: e2.clone(),
            W: w2.clone(),
            w_row_blindings: w_blinds2.clone(),
            e_row_blindings: e_blinds2.clone(),
        };

        let r = F::rand(&mut rng);
        let folded = wit1.fold(&wit2, &t, &t_blinds, r);

        let r_sq = r * r;
        for i in 0..m {
            assert_eq!(folded.E[i], e1[i] + r * t[i] + r_sq * e2[i]);
        }

        for i in 0..n {
            assert_eq!(folded.W[i], w1[i] + r * w2[i]);
        }

        for i in 0..4 {
            assert_eq!(folded.w_row_blindings[i], w_blinds1[i] + r * w_blinds2[i]);
        }

        for i in 0..2 {
            assert_eq!(
                folded.e_row_blindings[i],
                e_blinds1[i] + r * t_blinds[i] + r_sq * e_blinds2[i]
            );
        }
    }

    #[test]
    fn test_instance_folding() {
        let mut rng = thread_rng();
        type F = Fr;

        let gens = PedersenGenerators::<Bn254Curve>::deterministic(20);

        let u1 = F::rand(&mut rng);
        let u2 = F::rand(&mut rng);

        let rc1 = vec![gens.message_generators[0].scalar_mul(&F::rand(&mut rng))];
        let rc2 = vec![gens.message_generators[1].scalar_mul(&F::rand(&mut rng))];
        let nc1 = vec![gens.message_generators[2].scalar_mul(&F::rand(&mut rng))];
        let nc2 = vec![gens.message_generators[3].scalar_mul(&F::rand(&mut rng))];
        let e1 = vec![gens.message_generators[4].scalar_mul(&F::rand(&mut rng))];
        let e2 = vec![gens.message_generators[5].scalar_mul(&F::rand(&mut rng))];
        let t_rows = vec![gens.message_generators[6].scalar_mul(&F::rand(&mut rng))];

        let inst1 = RelaxedR1CSInstance::<F, Bn254Curve> {
            u: u1,
            round_commitments: rc1.clone(),
            noncoeff_row_commitments: nc1.clone(),
            e_row_commitments: e1.clone(),
            eval_commitments: Vec::new(),
        };

        let inst2 = RelaxedR1CSInstance::<F, Bn254Curve> {
            u: u2,
            round_commitments: rc2.clone(),
            noncoeff_row_commitments: nc2.clone(),
            e_row_commitments: e2.clone(),
            eval_commitments: Vec::new(),
        };

        let r = F::rand(&mut rng);
        let folded = inst1.fold(&inst2, &t_rows, r).unwrap();

        assert_eq!(folded.u, u1 + r * u2);

        let r_sq = r * r;
        let expected_e_row = e1[0] + t_rows[0].scalar_mul(&r) + e2[0].scalar_mul(&r_sq);
        assert_eq!(folded.e_row_commitments[0], expected_e_row);

        assert_eq!(folded.round_commitments[0], rc1[0] + rc2[0].scalar_mul(&r));
    }
}
