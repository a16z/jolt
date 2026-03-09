//! Stage 4: Read-write checking sumchecks.
//!
//! Contains two sumcheck instances:
//!
//! 1. **Register read-write checking** (degree 4):
//!    `eq·rd_wa·inc + eq·rd_wa·val + eq·γ·rs1_ra·val + eq·γ²·rs2_ra·val`
//!
//! 2. **RAM value check** (degree 3):
//!    `c0·inc·wa` where `c0 = LT(r_cycle', r_cycle) + γ`

use std::sync::Arc;

use jolt_compute::CpuBackend;
use jolt_field::Field;
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_sumcheck::prover::SumcheckCompute;
use jolt_transcript::Transcript;

use crate::claims::{ram, registers};
use crate::evaluators::formula::{formula_descriptor, FormulaEvaluator, Term};
use crate::stage::{ProverStage, StageBatch};

/// Register read-write checking + RAM value check prover stage.
///
/// The register RW checking formula has degree 4 (products of up to 3
/// polynomials plus eq). The RAM value check has degree 3 (c0·inc·wa).
/// Both use [`FormulaEvaluator`] as the sumcheck evaluator.
pub struct RwCheckingStage<F: Field> {
    // Register RW checking polynomials
    reg_val: Option<Vec<F>>,
    rs1_ra: Option<Vec<F>>,
    rs2_ra: Option<Vec<F>>,
    rd_wa: Option<Vec<F>>,
    rd_inc: Option<Vec<F>>,
    // RAM value check polynomials
    ram_inc: Option<Vec<F>>,
    ram_wa: Option<Vec<F>>,
    /// Eq point for register RW checking.
    reg_eq_point: Vec<F>,
    /// Challenges for register RW: [eq_eval, γ, γ²].
    reg_challenges: Vec<F>,
    /// Eq point for RAM value check.
    ram_eq_point: Vec<F>,
    /// Challenge for RAM val check: c0 = LT + γ.
    ram_c0: F,
    num_vars: usize,
}

impl<F: Field> RwCheckingStage<F> {
    /// Creates a new stage with all polynomial tables and challenges.
    ///
    /// # Arguments
    ///
    /// * `reg_polys` — `(val, rs1_ra, rs2_ra, rd_wa, rd_inc)` tables
    /// * `reg_eq_point` — Eq point for register RW checking
    /// * `reg_challenges` — `[eq_eval, γ, γ²]` for register formula
    /// * `ram_polys` — `(inc, wa)` tables for RAM value check
    /// * `ram_eq_point` — Eq point for RAM value check
    /// * `ram_c0` — Combined `LT + γ` challenge
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn new(
        reg_polys: (Vec<F>, Vec<F>, Vec<F>, Vec<F>, Vec<F>),
        reg_eq_point: Vec<F>,
        reg_challenges: Vec<F>,
        ram_polys: (Vec<F>, Vec<F>),
        ram_eq_point: Vec<F>,
        ram_c0: F,
    ) -> Self {
        let num_vars = reg_eq_point.len();
        let expected = 1usize << num_vars;
        assert_eq!(ram_eq_point.len(), num_vars);
        assert_eq!(reg_polys.0.len(), expected);
        assert_eq!(reg_polys.1.len(), expected);
        assert_eq!(reg_polys.2.len(), expected);
        assert_eq!(reg_polys.3.len(), expected);
        assert_eq!(reg_polys.4.len(), expected);
        assert_eq!(ram_polys.0.len(), expected);
        assert_eq!(ram_polys.1.len(), expected);

        Self {
            reg_val: Some(reg_polys.0),
            rs1_ra: Some(reg_polys.1),
            rs2_ra: Some(reg_polys.2),
            rd_wa: Some(reg_polys.3),
            rd_inc: Some(reg_polys.4),
            ram_inc: Some(ram_polys.0),
            ram_wa: Some(ram_polys.1),
            reg_eq_point,
            reg_challenges,
            ram_eq_point,
            ram_c0,
            num_vars,
        }
    }

    fn build_register_rw(&self) -> (SumcheckClaim<F>, Box<dyn SumcheckCompute<F>>) {
        let val = self.reg_val.as_ref().unwrap();
        let rs1_ra = self.rs1_ra.as_ref().unwrap();
        let rs2_ra = self.rs2_ra.as_ref().unwrap();
        let rd_wa = self.rd_wa.as_ref().unwrap();
        let inc = self.rd_inc.as_ref().unwrap();

        let eq_eval = self.reg_challenges[0];
        let gamma = self.reg_challenges[1];
        let gamma_sq = self.reg_challenges[2];

        let poly_tables = vec![
            val.clone(),
            rs1_ra.clone(),
            rs2_ra.clone(),
            rd_wa.clone(),
            inc.clone(),
        ];

        let terms = vec![
            Term {
                coeff: eq_eval,
                factors: vec![3, 4], // rd_wa · inc
            },
            Term {
                coeff: eq_eval,
                factors: vec![3, 0], // rd_wa · val
            },
            Term {
                coeff: eq_eval * gamma,
                factors: vec![1, 0], // γ · rs1_ra · val
            },
            Term {
                coeff: eq_eval * gamma_sq,
                factors: vec![2, 0], // γ² · rs2_ra · val
            },
        ];

        let eq_table = EqPolynomial::new(self.reg_eq_point.clone()).evaluations();
        let n = 1usize << self.num_vars;

        let claimed_sum: F = (0..n)
            .map(|x| {
                let mut formula_val = F::zero();
                for term in &terms {
                    let mut product = term.coeff;
                    for &idx in &term.factors {
                        product *= poly_tables[idx][x];
                    }
                    formula_val += product;
                }
                eq_table[x] * formula_val
            })
            .sum();

        let degree = 3;

        let backend = Arc::new(CpuBackend);
        let (desc, challenges) = formula_descriptor(&terms, poly_tables.len(), degree);
        let kernel = jolt_cpu_kernels::compile_with_challenges::<F>(&desc, &challenges);
        let poly_bufs: Vec<_> = poly_tables.iter().map(|t| backend.upload(t)).collect();
        let witness = FormulaEvaluator::new(
            backend.upload(&eq_table),
            poly_bufs,
            kernel,
            degree,
            backend,
        );

        let claim = SumcheckClaim {
            num_vars: self.num_vars,
            degree,
            claimed_sum,
        };

        (claim, Box::new(witness))
    }

    fn build_ram_val_check(&self) -> (SumcheckClaim<F>, Box<dyn SumcheckCompute<F>>) {
        let inc = self.ram_inc.as_ref().unwrap();
        let wa = self.ram_wa.as_ref().unwrap();

        let poly_tables = vec![inc.clone(), wa.clone()];

        let terms = vec![Term {
            coeff: self.ram_c0,
            factors: vec![0, 1], // inc · wa
        }];

        let eq_table = EqPolynomial::new(self.ram_eq_point.clone()).evaluations();
        let n = 1usize << self.num_vars;

        let claimed_sum: F = (0..n)
            .map(|x| eq_table[x] * self.ram_c0 * poly_tables[0][x] * poly_tables[1][x])
            .sum();

        let degree = 3;

        let backend = Arc::new(CpuBackend);
        let (desc, challenges) = formula_descriptor(&terms, poly_tables.len(), degree);
        let kernel = jolt_cpu_kernels::compile_with_challenges::<F>(&desc, &challenges);
        let poly_bufs: Vec<_> = poly_tables.iter().map(|t| backend.upload(t)).collect();
        let witness = FormulaEvaluator::new(
            backend.upload(&eq_table),
            poly_bufs,
            kernel,
            degree,
            backend,
        );

        let claim = SumcheckClaim {
            num_vars: self.num_vars,
            degree,
            claimed_sum,
        };

        (claim, Box::new(witness))
    }
}

impl<F: Field, T: Transcript> ProverStage<F, T> for RwCheckingStage<F> {
    fn build(&mut self, _prior_claims: &[ProverClaim<F>], _transcript: &mut T) -> StageBatch<F> {
        let (reg_claim, reg_witness) = self.build_register_rw();
        let (ram_claim, ram_witness) = self.build_ram_val_check();

        StageBatch {
            claims: vec![reg_claim, ram_claim],
            witnesses: vec![reg_witness, ram_witness],
        }
    }

    fn extract_claims(&mut self, challenges: &[F], _final_eval: F) -> Vec<ProverClaim<F>> {
        let tables: Vec<Vec<F>> = vec![
            self.reg_val.take().unwrap(),
            self.rs1_ra.take().unwrap(),
            self.rs2_ra.take().unwrap(),
            self.rd_wa.take().unwrap(),
            self.rd_inc.take().unwrap(),
            self.ram_inc.take().unwrap(),
            self.ram_wa.take().unwrap(),
        ];

        tables
            .into_iter()
            .map(|evals| {
                let poly = Polynomial::new(evals.clone());
                let eval = poly.evaluate(challenges);
                ProverClaim {
                    evaluations: evals,
                    point: challenges.to_vec(),
                    eval,
                }
            })
            .collect()
    }

    fn claim_definitions(&self) -> Vec<ClaimDefinition> {
        vec![
            registers::registers_read_write_checking(),
            ram::ram_val_check(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::Zero;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn random_table(n: usize, rng: &mut ChaCha20Rng) -> Vec<Fr> {
        (0..n).map(|_| Fr::random(rng)).collect()
    }

    #[test]
    fn stage_produces_two_claims() {
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let reg_eq: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let ram_eq: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let gamma = Fr::random(&mut rng);
        let lt_plus_gamma = Fr::random(&mut rng);

        let mut stage = RwCheckingStage::new(
            (
                random_table(n, &mut rng),
                random_table(n, &mut rng),
                random_table(n, &mut rng),
                random_table(n, &mut rng),
                random_table(n, &mut rng),
            ),
            reg_eq,
            vec![Fr::from_u64(1), gamma, gamma * gamma],
            (random_table(n, &mut rng), random_table(n, &mut rng)),
            ram_eq,
            lt_plus_gamma,
        );

        let mut t = Blake2bTranscript::new(b"test_s4");
        let batch = stage.build(&[], &mut t);

        assert_eq!(batch.claims.len(), 2);
        assert_eq!(batch.witnesses.len(), 2);
        assert_eq!(batch.claims[0].degree, 3); // register RW
        assert_eq!(batch.claims[1].degree, 3); // RAM val check
    }

    #[test]
    fn full_prove_verify_round_trip() {
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(99);

        let reg_eq: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let ram_eq: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let gamma = Fr::random(&mut rng);
        let eq_eval = Fr::random(&mut rng);
        let lt_plus_gamma = Fr::random(&mut rng);

        let mut stage: RwCheckingStage<Fr> = RwCheckingStage::new(
            (
                random_table(n, &mut rng),
                random_table(n, &mut rng),
                random_table(n, &mut rng),
                random_table(n, &mut rng),
                random_table(n, &mut rng),
            ),
            reg_eq,
            vec![eq_eval, eq_eval * gamma, eq_eval * gamma * gamma],
            (random_table(n, &mut rng), random_table(n, &mut rng)),
            ram_eq,
            lt_plus_gamma,
        );

        let mut pt = Blake2bTranscript::new(b"s4_roundtrip");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot: Vec<_> = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"s4_roundtrip");
        let (final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &claims_snapshot,
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        )
        .expect("verification should succeed");

        let prover_claims =
            <RwCheckingStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
                &mut stage,
                &challenges,
                final_eval,
            );
        // 5 register polys + 2 RAM polys = 7
        assert_eq!(prover_claims.len(), 7);

        for pc in &prover_claims {
            let poly = Polynomial::new(pc.evaluations.clone());
            assert_eq!(poly.evaluate(&challenges), pc.eval);
        }
    }

    #[test]
    fn claim_definitions_match() {
        let n = 8;
        let stage = RwCheckingStage::<Fr>::new(
            (
                vec![Fr::zero(); n],
                vec![Fr::zero(); n],
                vec![Fr::zero(); n],
                vec![Fr::zero(); n],
                vec![Fr::zero(); n],
            ),
            vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)],
            vec![Fr::from_u64(1); 3],
            (vec![Fr::zero(); n], vec![Fr::zero(); n]),
            vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)],
            Fr::from_u64(7),
        );

        let defs =
            <RwCheckingStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::claim_definitions(&stage);
        assert_eq!(defs.len(), 2);
    }
}
