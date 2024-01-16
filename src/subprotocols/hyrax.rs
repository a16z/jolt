use super::traits::PCS;


#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PolyEvalProof<G: CurveGroup> {
  proof: DotProductProofLog<G>,
}

impl<G: CurveGroup> PolyEvalProof<G> {
  fn protocol_name() -> &'static [u8] {
    b"polynomial evaluation proof"
  }
}

impl<G: CurveGroup> PCS for PolyEvalProof<G> {
  type Commitment;
  type Evaluation;
  type Challenge;
  type Proof;
  type Error;

  type ProverKey;
  type VerifierKey;

  #[tracing::instrument(skip_all, name = "DensePolynomial.commit")]
  fn commit<G>(
    &self,
    gens: &PolyCommitmentGens<G>,
    random_tape: Option<&mut RandomTape<G>>,
  ) -> (PolyCommitment<G>, PolyCommitmentBlinds<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    let n = self.Z.len();
    let ell = self.get_num_vars();
    assert_eq!(n, ell.pow2());

    let (left_num_vars, right_num_vars) = EqPolynomial::<F>::compute_factored_lens(ell);
    let L_size = left_num_vars.pow2();
    let R_size = right_num_vars.pow2();
    assert_eq!(L_size * R_size, n);

    let blinds = if let Some(t) = random_tape {
      PolyCommitmentBlinds {
        blinds: t.random_vector(b"poly_blinds", L_size),
      }
    } else {
      PolyCommitmentBlinds {
        blinds: vec![F::zero(); L_size],
      }
    };


  #[tracing::instrument(skip_all, name = "DensePolyEval.prove")]
  fn prove(
    poly: &DensePolynomial<G::ScalarField>,
    blinds_opt: Option<&PolyCommitmentBlinds<G::ScalarField>>,
    r: &[G::ScalarField], // point at which the polynomial is evaluated
    Zr: &G::ScalarField,  // evaluation of \widetilde{Z}(r)
    blind_Zr_opt: Option<&G::ScalarField>, // specifies a blind for Zr
    gens: &PolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> (PolyEvalProof<G>, G) {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      PolyEvalProof::<G>::protocol_name(),
    );

    // assert vectors are of the right size
    assert_eq!(poly.get_num_vars(), r.len());

    let (left_num_vars, right_num_vars) =
      EqPolynomial::<G::ScalarField>::compute_factored_lens(r.len());
    let L_size = left_num_vars.pow2();
    let R_size = right_num_vars.pow2();

    let default_blinds = PolyCommitmentBlinds {
      blinds: vec![G::ScalarField::zero(); L_size],
    };
    let blinds = blinds_opt.map_or(&default_blinds, |p| p);

    assert_eq!(blinds.blinds.len(), L_size);

    let zero = G::ScalarField::zero();
    let blind_Zr = blind_Zr_opt.map_or(&zero, |p| p);

    // compute the L and R vectors
    let eq = EqPolynomial::new(r.to_vec());
    let (L, R) = eq.compute_factored_evals();
    assert_eq!(L.len(), L_size);
    assert_eq!(R.len(), R_size);

    // compute the vector underneath L*Z and the L*blinds
    // compute vector-matrix product between L and Z viewed as a matrix
    let LZ = poly.bound(&L);
    let LZ_blind: G::ScalarField = (0..L.len()).map(|i| blinds.blinds[i] * L[i]).sum();

    // a dot product proof of size R_size
    let (proof, _C_LR, C_Zr_prime) = DotProductProofLog::prove(
      &gens.gens,
      transcript,
      random_tape,
      &LZ,
      &LZ_blind,
      &R,
      Zr,
      blind_Zr,
    );

    (PolyEvalProof { proof }, C_Zr_prime)
  }

  fn verify(
    &self,
    gens: &PolyCommitmentGens<G>,
    transcript: &mut Transcript,
    r: &[G::ScalarField], // point at which the polynomial is evaluated
    C_Zr: &G,             // commitment to \widetilde{Z}(r)
    comm: &PolyCommitment<G>,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      PolyEvalProof::<G>::protocol_name(),
    );

    // compute L and R
    let eq = EqPolynomial::new(r.to_vec());
    let (L, R) = eq.compute_factored_evals();

    // compute a weighted sum of commitments and L
    let C_affine = G::normalize_batch(&comm.C);

    let C_LZ = VariableBaseMSM::msm(C_affine.as_ref(), L.as_ref()).unwrap();

    self
      .proof
      .verify(R.len(), &gens.gens, transcript, &R, &C_LZ, C_Zr)
  }
}