#![allow(dead_code)]
use super::dense_mlpoly::DensePolynomial;
use super::dense_mlpoly::EqPolynomial;
use super::math::Math;
use super::sumcheck::SumcheckInstanceProof;
use super::transcript::ProofTranscript;
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::*;
use merlin::Transcript;

#[derive(Debug)]
pub struct GrandProductCircuit<F> {
  left_vec: Vec<DensePolynomial<F>>,
  right_vec: Vec<DensePolynomial<F>>,
}

impl<F: PrimeField> GrandProductCircuit<F> {
  fn compute_layer(
    inp_left: &DensePolynomial<F>,
    inp_right: &DensePolynomial<F>,
  ) -> (DensePolynomial<F>, DensePolynomial<F>) {
    let len = inp_left.len() + inp_right.len();
    let outp_left = (0..len / 4)
      .map(|i| inp_left[i] * inp_right[i])
      .collect::<Vec<F>>();
    let outp_right = (len / 4..len / 2)
      .map(|i| inp_left[i] * inp_right[i])
      .collect::<Vec<F>>();

    (
      DensePolynomial::new(outp_left),
      DensePolynomial::new(outp_right),
    )
  }

  pub fn new(poly: &DensePolynomial<F>) -> Self {
    let mut left_vec: Vec<DensePolynomial<F>> = Vec::new();
    let mut right_vec: Vec<DensePolynomial<F>> = Vec::new();

    let num_layers = poly.len().log_2() as usize;
    let (outp_left, outp_right) = poly.split(poly.len() / 2);

    left_vec.push(outp_left);
    right_vec.push(outp_right);

    for i in 0..num_layers - 1 {
      let (outp_left, outp_right) = GrandProductCircuit::compute_layer(&left_vec[i], &right_vec[i]);
      left_vec.push(outp_left);
      right_vec.push(outp_right);
    }

    GrandProductCircuit {
      left_vec,
      right_vec,
    }
  }

  pub fn evaluate(&self) -> F {
    let len = self.left_vec.len();
    assert_eq!(self.left_vec[len - 1].get_num_vars(), 0);
    assert_eq!(self.right_vec[len - 1].get_num_vars(), 0);
    self.left_vec[len - 1][0] * self.right_vec[len - 1][0]
  }
}

pub struct GeneralizedScalarProduct<F> {
  operands: Vec<DensePolynomial<F>>,
}

impl<F: PrimeField> GeneralizedScalarProduct<F> {
  pub fn new(operands: Vec<DensePolynomial<F>>) -> Self {
    assert!(operands.iter().all(|poly| poly.len() == operands[0].len()));
    GeneralizedScalarProduct { operands }
  }

  /// Evaluate operand polynomials over boolean hypercube, summing products of all evaluations.
  pub fn evaluate(&self) -> F {
    (0..self.operands[0].len())
      .map(|i| {
        self
          .operands
          .iter()
          .map(|polynomial| polynomial[i])
          .product::<F>()
      })
      .sum()
  }

  /// Split this GeneralizedScalarProduct into two by splitting each of the multi-linear polynomials to half the length.
  pub fn split(self) -> (GeneralizedScalarProduct<F>, GeneralizedScalarProduct<F>) {
    let idx = self.operands[0].len() / 2;
    assert_eq!(idx * 2, self.operands[0].len());
    let (left, right): (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>) = self
      .operands
      .iter()
      .map(|polynomial| polynomial.split(idx))
      .unzip();
    (
      GeneralizedScalarProduct { operands: left },
      GeneralizedScalarProduct { operands: right },
    )
  }
}

#[cfg(test)]
mod generalized_scalar_product_tests {
  use ark_bls12_381::{Fr};
  use super::*;
  use crate::utils::index_to_field_bitvector;

  #[test]
  fn evaluate() {
    // Create three dense polynomials, evaluate each over every point on the boolean hypercube and sum the products of each term
    let A = DensePolynomial::new(vec![Fr::from(3), Fr::from(3), Fr::from(3), Fr::from(3)]);
    let B = DensePolynomial::new(vec![Fr::from(5), Fr::from(5), Fr::from(5), Fr::from(5)]);
    let C = DensePolynomial::new(vec![Fr::from(7), Fr::from(7), Fr::from(7), Fr::from(7)]);

    let gsp = GeneralizedScalarProduct::new(vec![A.clone(), B.clone(), C.clone()]);

    // Calculate manually: Evaluate each at every point on the boolean hypercube and sum the products
    let mut manual_eval = Fr::from(0u64);
    for i in 0..4 {
      let a = A.evaluate(&index_to_field_bitvector(i, 2));
      let b = B.evaluate(&index_to_field_bitvector(i, 2));
      let c = C.evaluate(&index_to_field_bitvector(i, 2));

      manual_eval += a * b * c;
    }

    assert_eq!(gsp.evaluate(), manual_eval);
  }

  #[test]
  fn split() {
    let A = DensePolynomial::new(vec![Fr::from(3), Fr::from(3), Fr::from(3), Fr::from(3)]);
    let B = DensePolynomial::new(vec![Fr::from(5), Fr::from(5), Fr::from(7), Fr::from(7)]);
    let C = DensePolynomial::new(vec![Fr::from(9), Fr::from(9), Fr::from(9), Fr::from(9)]);

    let gsp = GeneralizedScalarProduct::new(vec![A.clone(), B.clone(), C.clone()]);

    let (left, right) = gsp.split();

    assert_eq!(left.operands.len(), 3);
    assert_eq!(right.operands.len(), 3);
    assert_eq!(left.operands[0].len(), 2);
    assert_eq!(right.operands[0].len(), 2);

    assert_eq!(left.operands[1].evaluate(&index_to_field_bitvector(1, 1)), Fr::from(5));
    assert_eq!(right.operands[1].evaluate(&index_to_field_bitvector(1, 1)), Fr::from(7));
  }
}

#[allow(dead_code)]
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LayerProof<F: PrimeField> {
  pub proof: SumcheckInstanceProof<F>,
  pub claims: Vec<F>,
}

#[allow(dead_code)]
impl<F: PrimeField> LayerProof<F> {
  pub fn verify<G, T: ProofTranscript<G>>(
    &self,
    claim: F,
    num_rounds: usize,
    degree_bound: usize,
    transcript: &mut T,
  ) -> (F, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    self
      .proof
      .verify::<G, T>(claim, num_rounds, degree_bound, transcript)
      .unwrap()
  }
}

#[allow(dead_code)]
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LayerProofBatched<F: PrimeField> {
  pub proof: SumcheckInstanceProof<F>,
  pub claims_prod_left: Vec<F>,
  pub claims_prod_right: Vec<F>,
}

#[allow(dead_code)]
impl<F: PrimeField> LayerProofBatched<F> {
  pub fn verify<G, T: ProofTranscript<G>>(
    &self,
    claim: F,
    num_rounds: usize,
    degree_bound: usize,
    transcript: &mut T,
  ) -> (F, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    self
      .proof
      .verify::<G, T>(claim, num_rounds, degree_bound, transcript)
      .unwrap()
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct GrandProductArgument<F: PrimeField> {
  proof: Vec<LayerProof<F>>,
}

impl<F: PrimeField> GrandProductArgument<F> {
  #![allow(dead_code)]
  pub fn prove<G, T: ProofTranscript<G>>(
    circuit: &mut GrandProductCircuit<F>,
    transcript: &mut T,
  ) -> (Self, F, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    let mut proof: Vec<LayerProof<F>> = Vec::new();
    let num_layers = circuit.left_vec.len();

    let mut claim = circuit.evaluate();
    let mut rand = Vec::new();
    for layer_id in (0..num_layers).rev() {
      let len = circuit.left_vec[layer_id].len() + circuit.right_vec[layer_id].len();

      let mut poly_C = DensePolynomial::new(EqPolynomial::new(rand.clone()).evals());
      assert_eq!(poly_C.len(), len / 2);

      let num_rounds_prod = poly_C.len().log_2() as usize;
      let comb_func_prod = |poly_A_comp: &F, poly_B_comp: &F, poly_C_comp: &F| -> F {
        *poly_A_comp * *poly_B_comp * *poly_C_comp
      };
      let (proof_prod, rand_prod, claims_prod) = SumcheckInstanceProof::<F>::prove_cubic::<_, G, T>(
        &claim,
        num_rounds_prod,
        &mut circuit.left_vec[layer_id],
        &mut circuit.right_vec[layer_id],
        &mut poly_C,
        comb_func_prod,
        transcript,
      );
      transcript.append_scalar(b"claim_prod_left", &claims_prod[0]);
      transcript.append_scalar(b"claim_prod_right", &claims_prod[1]);

      // produce a random challenge
      let r_layer = transcript.challenge_scalar(b"challenge_r_layer");

      claim = claims_prod[0] + r_layer * (claims_prod[1] - claims_prod[0]);

      let mut ext = vec![r_layer];
      ext.extend(rand_prod);
      rand = ext;

      proof.push(LayerProof {
        proof: proof_prod,
        claims: claims_prod[0..claims_prod.len() - 1].to_vec(),
      });
    }

    (GrandProductArgument { proof }, claim, rand)
  }

  pub fn verify<G, T: ProofTranscript<G>>(
    &self,
    eval: F,
    len: usize,
    transcript: &mut T,
  ) -> (F, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    let num_layers = len.log_2() as usize;
    let mut claim = eval;
    let mut rand: Vec<F> = Vec::new();
    assert_eq!(self.proof.len(), num_layers);

    for (num_rounds, i) in (0..num_layers).enumerate() {
      let (claim_last, rand_prod) = self.proof[i].verify::<G, T>(claim, num_rounds, 3, transcript);

      let claims_prod = &self.proof[i].claims;
      transcript.append_scalar(b"claim_prod_left", &claims_prod[0]);
      transcript.append_scalar(b"claim_prod_right", &claims_prod[1]);

      assert_eq!(rand.len(), rand_prod.len());
      let eq: F = (0..rand.len())
        .map(|i| rand[i] * rand_prod[i] + (F::one() - rand[i]) * (F::one() - rand_prod[i]))
        .product();
      assert_eq!(claims_prod[0] * claims_prod[1] * eq, claim_last);

      // produce a random challenge
      let r_layer = transcript.challenge_scalar(b"challenge_r_layer");

      claim = (F::one() - r_layer) * claims_prod[0] + r_layer * claims_prod[1];
      let mut ext = vec![r_layer];
      ext.extend(rand_prod);
      rand = ext;
    }

    (claim, rand)
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedGrandProductArgument<F: PrimeField> {
  proof: Vec<LayerProofBatched<F>>,
}

impl<F: PrimeField> BatchedGrandProductArgument<F> {
  pub fn prove<G>(
    grand_product_circuits: &mut Vec<&mut GrandProductCircuit<F>>,
    transcript: &mut Transcript,
  ) -> (Self, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    assert!(!grand_product_circuits.is_empty());

    let mut proof_layers: Vec<LayerProofBatched<F>> = Vec::new();
    let num_layers = grand_product_circuits[0].left_vec.len();
    let mut claims_to_verify = (0..grand_product_circuits.len())
      .map(|i| grand_product_circuits[i].evaluate())
      .collect::<Vec<F>>();

    let mut rand = Vec::new();
    for layer_id in (0..num_layers).rev() {
      // prepare parallel instance that share poly_C first
      let len = grand_product_circuits[0].left_vec[layer_id].len()
        + grand_product_circuits[0].right_vec[layer_id].len();

      let mut poly_C_par = DensePolynomial::new(EqPolynomial::<F>::new(rand.clone()).evals());
      assert_eq!(poly_C_par.len(), len / 2);

      let num_rounds_prod = poly_C_par.len().log_2() as usize;
      let comb_func_prod = |poly_A_comp: &F, poly_B_comp: &F, poly_C_comp: &F| -> F {
        *poly_A_comp * *poly_B_comp * *poly_C_comp
      };

      let mut poly_A_batched_par: Vec<&mut DensePolynomial<F>> = Vec::new();
      let mut poly_B_batched_par: Vec<&mut DensePolynomial<F>> = Vec::new();
      for prod_circuit in grand_product_circuits.iter_mut() {
        poly_A_batched_par.push(&mut prod_circuit.left_vec[layer_id]);
        poly_B_batched_par.push(&mut prod_circuit.right_vec[layer_id])
      }
      let poly_vec_par = (
        &mut poly_A_batched_par,
        &mut poly_B_batched_par,
        &mut poly_C_par,
      );

      // produce a fresh set of coeffs and a joint claim
      let coeff_vec = <Transcript as ProofTranscript<G>>::challenge_vector(
        transcript,
        b"rand_coeffs_next_layer",
        claims_to_verify.len(),
      );
      let claim = (0..claims_to_verify.len())
        .map(|i| claims_to_verify[i] * coeff_vec[i])
        .sum();

      let (proof, rand_prod, claims_prod) =
        SumcheckInstanceProof::<F>::prove_cubic_batched::<_, G>(
          &claim,
          num_rounds_prod,
          poly_vec_par,
          &coeff_vec,
          comb_func_prod,
          transcript,
        );

      let (claims_prod_left, claims_prod_right, _claims_eq) = claims_prod;
      for i in 0..grand_product_circuits.len() {
        <Transcript as ProofTranscript<G>>::append_scalar(
          transcript,
          b"claim_prod_left",
          &claims_prod_left[i],
        );

        <Transcript as ProofTranscript<G>>::append_scalar(
          transcript,
          b"claim_prod_right",
          &claims_prod_right[i],
        );
      }

      // produce a random challenge to condense two claims into a single claim
      let r_layer =
        <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"challenge_r_layer");

      claims_to_verify = (0..grand_product_circuits.len())
        .map(|i| claims_prod_left[i] + r_layer * (claims_prod_right[i] - claims_prod_left[i]))
        .collect::<Vec<F>>();

      let mut ext = vec![r_layer];
      ext.extend(rand_prod);
      rand = ext;

      proof_layers.push(LayerProofBatched {
        proof,
        claims_prod_left,
        claims_prod_right,
      });
    }

    (
      BatchedGrandProductArgument {
        proof: proof_layers,
      },
      rand,
    )
  }

  pub fn verify<G, T: ProofTranscript<G>>(
    &self,
    claims_prod_vec: &Vec<F>,
    len: usize,
    transcript: &mut T,
  ) -> (Vec<F>, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    let num_layers = len.log_2() as usize;
    let mut rand: Vec<F> = Vec::new();
    assert_eq!(self.proof.len(), num_layers);

    let mut claims_to_verify = claims_prod_vec.to_owned();
    for (num_rounds, i) in (0..num_layers).enumerate() {
      // produce random coefficients, one for each instance
      let coeff_vec =
        transcript.challenge_vector(b"rand_coeffs_next_layer", claims_to_verify.len());

      // produce a joint claim
      let claim = (0..claims_to_verify.len())
        .map(|i| claims_to_verify[i] * coeff_vec[i])
        .sum();

      let (claim_last, rand_prod) = self.proof[i].verify::<G, T>(claim, num_rounds, 3, transcript);

      let claims_prod_left = &self.proof[i].claims_prod_left;
      let claims_prod_right = &self.proof[i].claims_prod_right;
      assert_eq!(claims_prod_left.len(), claims_prod_vec.len());
      assert_eq!(claims_prod_right.len(), claims_prod_vec.len());

      for i in 0..claims_prod_vec.len() {
        transcript.append_scalar(b"claim_prod_left", &claims_prod_left[i]);
        transcript.append_scalar(b"claim_prod_right", &claims_prod_right[i]);
      }

      assert_eq!(rand.len(), rand_prod.len());
      let eq: F = (0..rand.len())
        .map(|i| rand[i] * rand_prod[i] + (F::one() - rand[i]) * (F::one() - rand_prod[i]))
        .product();
      let claim_expected: F = (0..claims_prod_vec.len())
        .map(|i| coeff_vec[i] * (claims_prod_left[i] * claims_prod_right[i] * eq))
        .sum();

      assert_eq!(claim_expected, claim_last);

      // produce a random challenge
      let r_layer = transcript.challenge_scalar(b"challenge_r_layer");

      claims_to_verify = (0..claims_prod_left.len())
        .map(|i| claims_prod_left[i] + r_layer * (claims_prod_right[i] - claims_prod_left[i]))
        .collect::<Vec<F>>();

      let mut ext = vec![r_layer];
      ext.extend(rand_prod);
      rand = ext;
    }
    (claims_to_verify, rand)
  }
}