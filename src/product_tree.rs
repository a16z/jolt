#![allow(dead_code)]
use super::dense_mlpoly::DensePolynomial;
use super::dense_mlpoly::EqPolynomial;
use super::math::Math;
use super::scalar::Scalar;
use super::sumcheck::SumcheckInstanceProof;
use super::transcript::ProofTranscript;
use ark_serialize::*;
use ark_std::One;
use merlin::Transcript;

#[derive(Debug)]
pub struct ProductCircuit {
  left_vec: Vec<DensePolynomial>,
  right_vec: Vec<DensePolynomial>,
}

impl ProductCircuit {
  fn compute_layer(
    inp_left: &DensePolynomial,
    inp_right: &DensePolynomial,
  ) -> (DensePolynomial, DensePolynomial) {
    let len = inp_left.len() + inp_right.len();
    let outp_left = (0..len / 4)
      .map(|i| inp_left[i] * inp_right[i])
      .collect::<Vec<Scalar>>();
    let outp_right = (len / 4..len / 2)
      .map(|i| inp_left[i] * inp_right[i])
      .collect::<Vec<Scalar>>();

    (
      DensePolynomial::new(outp_left),
      DensePolynomial::new(outp_right),
    )
  }

  pub fn new(poly: &DensePolynomial) -> Self {
    let mut left_vec: Vec<DensePolynomial> = Vec::new();
    let mut right_vec: Vec<DensePolynomial> = Vec::new();

    let num_layers = poly.len().log_2() as usize;
    let (outp_left, outp_right) = poly.split(poly.len() / 2);

    left_vec.push(outp_left);
    right_vec.push(outp_right);

    for i in 0..num_layers - 1 {
      let (outp_left, outp_right) = ProductCircuit::compute_layer(&left_vec[i], &right_vec[i]);
      left_vec.push(outp_left);
      right_vec.push(outp_right);
    }

    ProductCircuit {
      left_vec,
      right_vec,
    }
  }

  pub fn evaluate(&self) -> Scalar {
    let len = self.left_vec.len();
    assert_eq!(self.left_vec[len - 1].get_num_vars(), 0);
    assert_eq!(self.right_vec[len - 1].get_num_vars(), 0);
    self.left_vec[len - 1][0] * self.right_vec[len - 1][0]
  }
}

pub struct DotProductCircuit {
  left: DensePolynomial,
  right: DensePolynomial,
  weight: DensePolynomial,
}

impl DotProductCircuit {
  pub fn new(left: DensePolynomial, right: DensePolynomial, weight: DensePolynomial) -> Self {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), weight.len());
    DotProductCircuit {
      left,
      right,
      weight,
    }
  }

  pub fn evaluate(&self) -> Scalar {
    (0..self.left.len())
      .map(|i| self.left[i] * self.right[i] * self.weight[i])
      .sum()
  }

  pub fn split(&mut self) -> (DotProductCircuit, DotProductCircuit) {
    let idx = self.left.len() / 2;
    assert_eq!(idx * 2, self.left.len());
    let (l1, l2) = self.left.split(idx);
    let (r1, r2) = self.right.split(idx);
    let (w1, w2) = self.weight.split(idx);
    (
      DotProductCircuit {
        left: l1,
        right: r1,
        weight: w1,
      },
      DotProductCircuit {
        left: l2,
        right: r2,
        weight: w2,
      },
    )
  }
}

#[allow(dead_code)]
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LayerProof {
  pub proof: SumcheckInstanceProof,
  pub claims: Vec<Scalar>,
}

#[allow(dead_code)]
impl LayerProof {
  pub fn verify(
    &self,
    claim: Scalar,
    num_rounds: usize,
    degree_bound: usize,
    transcript: &mut Transcript,
  ) -> (Scalar, Vec<Scalar>) {
    self
      .proof
      .verify(claim, num_rounds, degree_bound, transcript)
      .unwrap()
  }
}

#[allow(dead_code)]
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LayerProofBatched {
  pub proof: SumcheckInstanceProof,
  pub claims_prod_left: Vec<Scalar>,
  pub claims_prod_right: Vec<Scalar>,
}

#[allow(dead_code)]
impl LayerProofBatched {
  pub fn verify(
    &self,
    claim: Scalar,
    num_rounds: usize,
    degree_bound: usize,
    transcript: &mut Transcript,
  ) -> (Scalar, Vec<Scalar>) {
    self
      .proof
      .verify(claim, num_rounds, degree_bound, transcript)
      .unwrap()
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProductCircuitEvalProof {
  proof: Vec<LayerProof>,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProductCircuitEvalProofBatched {
  proof: Vec<LayerProofBatched>,
  claims_dotp: (Vec<Scalar>, Vec<Scalar>, Vec<Scalar>),
}

impl ProductCircuitEvalProof {
  #![allow(dead_code)]
  pub fn prove(
    circuit: &mut ProductCircuit,
    transcript: &mut Transcript,
  ) -> (Self, Scalar, Vec<Scalar>) {
    let mut proof: Vec<LayerProof> = Vec::new();
    let num_layers = circuit.left_vec.len();

    let mut claim = circuit.evaluate();
    let mut rand = Vec::new();
    for layer_id in (0..num_layers).rev() {
      let len = circuit.left_vec[layer_id].len() + circuit.right_vec[layer_id].len();

      let mut poly_C = DensePolynomial::new(EqPolynomial::new(rand.clone()).evals());
      assert_eq!(poly_C.len(), len / 2);

      let num_rounds_prod = poly_C.len().log_2() as usize;
      let comb_func_prod = |poly_A_comp: &Scalar,
                            poly_B_comp: &Scalar,
                            poly_C_comp: &Scalar|
       -> Scalar { *poly_A_comp * *poly_B_comp * *poly_C_comp };
      let (proof_prod, rand_prod, claims_prod) = SumcheckInstanceProof::prove_cubic(
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

    (ProductCircuitEvalProof { proof }, claim, rand)
  }

  pub fn verify(
    &self,
    eval: Scalar,
    len: usize,
    transcript: &mut Transcript,
  ) -> (Scalar, Vec<Scalar>) {
    let num_layers = len.log_2() as usize;
    let mut claim = eval;
    let mut rand: Vec<Scalar> = Vec::new();
    //let mut num_rounds = 0;
    assert_eq!(self.proof.len(), num_layers);
    for (num_rounds, i) in (0..num_layers).enumerate() {
      let (claim_last, rand_prod) = self.proof[i].verify(claim, num_rounds, 3, transcript);

      let claims_prod = &self.proof[i].claims;
      transcript.append_scalar(b"claim_prod_left", &claims_prod[0]);
      transcript.append_scalar(b"claim_prod_right", &claims_prod[1]);

      assert_eq!(rand.len(), rand_prod.len());
      let eq: Scalar = (0..rand.len())
        .map(|i| {
          rand[i] * rand_prod[i] + (Scalar::one() - rand[i]) * (Scalar::one() - rand_prod[i])
        })
        .product();
      assert_eq!(claims_prod[0] * claims_prod[1] * eq, claim_last);

      // produce a random challenge
      let r_layer = transcript.challenge_scalar(b"challenge_r_layer");
      claim = (Scalar::one() - r_layer) * claims_prod[0] + r_layer * claims_prod[1];
      let mut ext = vec![r_layer];
      ext.extend(rand_prod);
      rand = ext;
    }

    (claim, rand)
  }
}

impl ProductCircuitEvalProofBatched {
  pub fn prove(
    prod_circuit_vec: &mut Vec<&mut ProductCircuit>,
    dotp_circuit_vec: &mut Vec<&mut DotProductCircuit>,
    transcript: &mut Transcript,
  ) -> (Self, Vec<Scalar>) {
    assert!(!prod_circuit_vec.is_empty());

    let mut claims_dotp_final = (Vec::new(), Vec::new(), Vec::new());

    let mut proof_layers: Vec<LayerProofBatched> = Vec::new();
    let num_layers = prod_circuit_vec[0].left_vec.len();
    let mut claims_to_verify = (0..prod_circuit_vec.len())
      .map(|i| prod_circuit_vec[i].evaluate())
      .collect::<Vec<Scalar>>();
    let mut rand = Vec::new();
    for layer_id in (0..num_layers).rev() {
      // prepare paralell instance that share poly_C first
      let len = prod_circuit_vec[0].left_vec[layer_id].len()
        + prod_circuit_vec[0].right_vec[layer_id].len();

      let mut poly_C_par = DensePolynomial::new(EqPolynomial::new(rand.clone()).evals());
      assert_eq!(poly_C_par.len(), len / 2);

      let num_rounds_prod = poly_C_par.len().log_2() as usize;
      let comb_func_prod = |poly_A_comp: &Scalar,
                            poly_B_comp: &Scalar,
                            poly_C_comp: &Scalar|
       -> Scalar { *poly_A_comp * *poly_B_comp * *poly_C_comp };

      let mut poly_A_batched_par: Vec<&mut DensePolynomial> = Vec::new();
      let mut poly_B_batched_par: Vec<&mut DensePolynomial> = Vec::new();
      for prod_circuit in prod_circuit_vec.iter_mut() {
        poly_A_batched_par.push(&mut prod_circuit.left_vec[layer_id]);
        poly_B_batched_par.push(&mut prod_circuit.right_vec[layer_id])
      }
      let poly_vec_par = (
        &mut poly_A_batched_par,
        &mut poly_B_batched_par,
        &mut poly_C_par,
      );

      // prepare sequential instances that don't share poly_C
      let mut poly_A_batched_seq: Vec<&mut DensePolynomial> = Vec::new();
      let mut poly_B_batched_seq: Vec<&mut DensePolynomial> = Vec::new();
      let mut poly_C_batched_seq: Vec<&mut DensePolynomial> = Vec::new();
      if layer_id == 0 && !dotp_circuit_vec.is_empty() {
        // add additional claims
        for item in dotp_circuit_vec.iter() {
          claims_to_verify.push(item.evaluate());
          assert_eq!(len / 2, item.left.len());
          assert_eq!(len / 2, item.right.len());
          assert_eq!(len / 2, item.weight.len());
        }

        for dotp_circuit in dotp_circuit_vec.iter_mut() {
          poly_A_batched_seq.push(&mut dotp_circuit.left);
          poly_B_batched_seq.push(&mut dotp_circuit.right);
          poly_C_batched_seq.push(&mut dotp_circuit.weight);
        }
      }
      let poly_vec_seq = (
        &mut poly_A_batched_seq,
        &mut poly_B_batched_seq,
        &mut poly_C_batched_seq,
      );

      // produce a fresh set of coeffs and a joint claim
      let coeff_vec =
        transcript.challenge_vector(b"rand_coeffs_next_layer", claims_to_verify.len());
      let claim = (0..claims_to_verify.len())
        .map(|i| claims_to_verify[i] * coeff_vec[i])
        .sum();

      let (proof, rand_prod, claims_prod, claims_dotp) = SumcheckInstanceProof::prove_cubic_batched(
        &claim,
        num_rounds_prod,
        poly_vec_par,
        poly_vec_seq,
        &coeff_vec,
        comb_func_prod,
        transcript,
      );

      let (claims_prod_left, claims_prod_right, _claims_eq) = claims_prod;
      for i in 0..prod_circuit_vec.len() {
        transcript.append_scalar(b"claim_prod_left", &claims_prod_left[i]);
        transcript.append_scalar(b"claim_prod_right", &claims_prod_right[i]);
      }

      if layer_id == 0 && !dotp_circuit_vec.is_empty() {
        let (claims_dotp_left, claims_dotp_right, claims_dotp_weight) = claims_dotp;
        for i in 0..dotp_circuit_vec.len() {
          transcript.append_scalar(b"claim_dotp_left", &claims_dotp_left[i]);
          transcript.append_scalar(b"claim_dotp_right", &claims_dotp_right[i]);
          transcript.append_scalar(b"claim_dotp_weight", &claims_dotp_weight[i]);
        }
        claims_dotp_final = (claims_dotp_left, claims_dotp_right, claims_dotp_weight);
      }

      // produce a random challenge to condense two claims into a single claim
      let r_layer = transcript.challenge_scalar(b"challenge_r_layer");

      claims_to_verify = (0..prod_circuit_vec.len())
        .map(|i| claims_prod_left[i] + r_layer * (claims_prod_right[i] - claims_prod_left[i]))
        .collect::<Vec<Scalar>>();

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
      ProductCircuitEvalProofBatched {
        proof: proof_layers,
        claims_dotp: claims_dotp_final,
      },
      rand,
    )
  }

  pub fn verify(
    &self,
    claims_prod_vec: &[Scalar],
    claims_dotp_vec: &[Scalar],
    len: usize,
    transcript: &mut Transcript,
  ) -> (Vec<Scalar>, Vec<Scalar>, Vec<Scalar>) {
    let num_layers = len.log_2() as usize;
    let mut rand: Vec<Scalar> = Vec::new();
    //let mut num_rounds = 0;
    assert_eq!(self.proof.len(), num_layers);

    let mut claims_to_verify = claims_prod_vec.to_owned();
    let mut claims_to_verify_dotp: Vec<Scalar> = Vec::new();
    for (num_rounds, i) in (0..num_layers).enumerate() {
      if i == num_layers - 1 {
        claims_to_verify.extend(claims_dotp_vec);
      }

      // produce random coefficients, one for each instance
      let coeff_vec =
        transcript.challenge_vector(b"rand_coeffs_next_layer", claims_to_verify.len());

      // produce a joint claim
      let claim = (0..claims_to_verify.len())
        .map(|i| claims_to_verify[i] * coeff_vec[i])
        .sum();

      let (claim_last, rand_prod) = self.proof[i].verify(claim, num_rounds, 3, transcript);

      let claims_prod_left = &self.proof[i].claims_prod_left;
      let claims_prod_right = &self.proof[i].claims_prod_right;
      assert_eq!(claims_prod_left.len(), claims_prod_vec.len());
      assert_eq!(claims_prod_right.len(), claims_prod_vec.len());

      for i in 0..claims_prod_vec.len() {
        transcript.append_scalar(b"claim_prod_left", &claims_prod_left[i]);
        transcript.append_scalar(b"claim_prod_right", &claims_prod_right[i]);
      }

      assert_eq!(rand.len(), rand_prod.len());
      let eq: Scalar = (0..rand.len())
        .map(|i| {
          rand[i] * rand_prod[i] + (Scalar::one() - rand[i]) * (Scalar::one() - rand_prod[i])
        })
        .product();
      let mut claim_expected: Scalar = (0..claims_prod_vec.len())
        .map(|i| coeff_vec[i] * (claims_prod_left[i] * claims_prod_right[i] * eq))
        .sum();

      // add claims from the dotp instances
      if i == num_layers - 1 {
        let num_prod_instances = claims_prod_vec.len();
        let (claims_dotp_left, claims_dotp_right, claims_dotp_weight) = &self.claims_dotp;
        for i in 0..claims_dotp_left.len() {
          transcript.append_scalar(b"claim_dotp_left", &claims_dotp_left[i]);
          transcript.append_scalar(b"claim_dotp_right", &claims_dotp_right[i]);
          transcript.append_scalar(b"claim_dotp_weight", &claims_dotp_weight[i]);

          claim_expected += coeff_vec[i + num_prod_instances]
            * claims_dotp_left[i]
            * claims_dotp_right[i]
            * claims_dotp_weight[i];
        }
      }

      assert_eq!(claim_expected, claim_last);

      // produce a random challenge
      let r_layer = transcript.challenge_scalar(b"challenge_r_layer");

      claims_to_verify = (0..claims_prod_left.len())
        .map(|i| claims_prod_left[i] + r_layer * (claims_prod_right[i] - claims_prod_left[i]))
        .collect::<Vec<Scalar>>();

      // add claims to verify for dotp circuit
      if i == num_layers - 1 {
        let (claims_dotp_left, claims_dotp_right, claims_dotp_weight) = &self.claims_dotp;

        for i in 0..claims_dotp_vec.len() / 2 {
          // combine left claims
          let claim_left = claims_dotp_left[2 * i]
            + r_layer * (claims_dotp_left[2 * i + 1] - claims_dotp_left[2 * i]);

          let claim_right = claims_dotp_right[2 * i]
            + r_layer * (claims_dotp_right[2 * i + 1] - claims_dotp_right[2 * i]);

          let claim_weight = claims_dotp_weight[2 * i]
            + r_layer * (claims_dotp_weight[2 * i + 1] - claims_dotp_weight[2 * i]);
          claims_to_verify_dotp.push(claim_left);
          claims_to_verify_dotp.push(claim_right);
          claims_to_verify_dotp.push(claim_weight);
        }
      }

      let mut ext = vec![r_layer];
      ext.extend(rand_prod);
      rand = ext;
    }
    (claims_to_verify, claims_to_verify_dotp, rand)
  }
}
