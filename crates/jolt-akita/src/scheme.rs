use akita_pcs::{CommitmentProver, ComputeBackendSetup, CpuBackend, RootPolyShape};
use jolt_crypto::Commitment;
use jolt_field::CanonicalBytes;
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, OpeningsError, VerifierOpeningClaim,
    ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::adapters::{
    akita_error, akita_ordered_evaluations, backend_stack, commit_failed, dense_polynomials,
    domain_size, invalid_batch, one_hot_polynomial, serialize_akita, sparse_unit_polynomial,
    transparent_zk_error, AkitaBackendCommitment, AkitaBackendDensePoly, AkitaBackendHint,
    AkitaBackendScheme, AkitaBatchProof, AkitaCommitment, AkitaField, AkitaHidingCommitment,
    AkitaHintPolynomials, AkitaLayoutDigest, AkitaOneHotBackendScheme, AkitaProverHint,
    AkitaProverSetup, AkitaSetupParams, AkitaVerifierSetup, AKITA_D, AKITA_ONE_HOT_LOG_K,
};
use crate::native_batching::{AkitaNativeBatchWitness, AkitaNativeBatching};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaScheme;

impl AkitaScheme {
    /// Returns true when the Akita backend sparse-ring path can represent a
    /// unit-valued sparse polynomial with this multilinear dimension.
    pub fn supports_unit_sparse_dimension(num_vars: usize) -> bool {
        domain_size(num_vars).is_some_and(|size| size >= AKITA_D)
    }

    pub fn commit_group(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        polynomials: &[Polynomial<AkitaField>],
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let first = polynomials
            .first()
            .ok_or_else(|| invalid_batch("Akita commitment group must contain a polynomial"))?;
        let num_vars = first.num_vars();

        Self::validate_commit_shape(setup, num_vars, polynomials.len())?;
        for polynomial in polynomials {
            if polynomial.num_vars() != num_vars {
                return Err(invalid_batch(format!(
                    "Akita commitment group mixes {}-variable and {num_vars}-variable polynomials",
                    polynomial.num_vars()
                )));
            }
        }

        let dense = dense_polynomials(polynomials)?;
        Self::commit_dense_backend(setup, layout_digest, num_vars, dense)
    }

    /// Validates the commitment shape before handing values to Akita.
    fn validate_commit_shape(
        setup: &AkitaProverSetup,
        num_vars: usize,
        poly_count: usize,
    ) -> Result<(), OpeningsError> {
        if num_vars != setup.max_num_vars() {
            return Err(invalid_batch(format!(
                "Akita commitment dimension {num_vars} does not match exact setup dimension {}",
                setup.max_num_vars()
            )));
        }
        if poly_count > setup.max_num_polys_per_commitment_group() {
            return Err(invalid_batch(format!(
                "Akita commitment group has {poly_count} polynomials but setup supports {}",
                setup.max_num_polys_per_commitment_group()
            )));
        }
        Ok(())
    }

    /// Wraps a backend commitment and its opening data into the adapter's
    /// commitment/hint pair; the flavor and polynomial count come from the
    /// hint polynomials themselves.
    fn package_commitment(
        layout_digest: AkitaLayoutDigest,
        num_vars: usize,
        backend_commitment: AkitaBackendCommitment,
        backend_hint: AkitaBackendHint,
        polynomials: AkitaHintPolynomials,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let commitment = AkitaCommitment {
            backend_flavor: polynomials.backend_flavor(),
            layout_digest,
            num_vars,
            poly_count: polynomials.len(),
            serialized_backend_bytes: serialize_akita(&backend_commitment)?,
        };
        Ok((
            commitment.clone(),
            AkitaProverHint {
                commitment,
                backend: Some((backend_commitment, backend_hint)),
                polynomials,
            },
        ))
    }

    fn commit_dense_backend(
        setup: &AkitaProverSetup,
        layout_digest: AkitaLayoutDigest,
        num_vars: usize,
        dense: Vec<AkitaBackendDensePoly>,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let stack = backend_stack(&setup.backend_prover_setup, &setup.prepared_backend_setup)?;
        let (backend_commitment, backend_hint) =
            AkitaBackendScheme::commit(&setup.backend_prover_setup, dense.as_slice(), &stack)
                .map_err(commit_failed)?;
        Self::package_commitment(
            layout_digest,
            num_vars,
            backend_commitment,
            backend_hint,
            AkitaHintPolynomials::Dense(dense.into()),
        )
    }
}

impl Commitment for AkitaScheme {
    type Output = AkitaCommitment;
}

impl CommitmentScheme for AkitaScheme {
    type Field = AkitaField;
    type Proof = AkitaBatchProof;
    type ProverSetup = AkitaProverSetup;
    type VerifierSetup = AkitaVerifierSetup;
    type Polynomial = Polynomial<AkitaField>;
    type OpeningHint = AkitaProverHint;
    type SetupParams = AkitaSetupParams;

    fn setup(
        params: Self::SetupParams,
    ) -> Result<(Self::ProverSetup, Self::VerifierSetup), OpeningsError> {
        let invalid_setup =
            |err: &dyn std::fmt::Display| OpeningsError::InvalidSetup(err.to_string());
        let backend_prover_setup = AkitaBackendScheme::setup_prover(
            params.max_num_vars,
            params.max_num_polys_per_commitment_group,
        )
        .map_err(|err| invalid_setup(&err))?;
        let prepared_backend_setup = CpuBackend
            .prepare_setup(&backend_prover_setup)
            .map_err(|err| invalid_setup(&err))?;
        let backend_verifier_setup = AkitaBackendScheme::setup_verifier(&backend_prover_setup);
        let (one_hot_backend_prover_setup, prepared_one_hot_backend_setup, one_hot_verifier_bytes) =
            if params.max_num_vars >= AKITA_ONE_HOT_LOG_K {
                let backend_prover_setup = AkitaOneHotBackendScheme::setup_prover(
                    params.max_num_vars,
                    params.max_num_polys_per_commitment_group,
                )
                .map_err(|err| invalid_setup(&err))?;
                let prepared_backend_setup = CpuBackend
                    .prepare_setup(&backend_prover_setup)
                    .map_err(|err| invalid_setup(&err))?;
                let backend_verifier_setup =
                    AkitaOneHotBackendScheme::setup_verifier(&backend_prover_setup);
                let verifier_bytes = serialize_akita(&backend_verifier_setup)?;
                (
                    Some(backend_prover_setup),
                    Some(prepared_backend_setup),
                    Some(verifier_bytes),
                )
            } else {
                (None, None, None)
            };
        let verifier = AkitaVerifierSetup {
            max_num_vars: params.max_num_vars,
            max_num_polys_per_commitment_group: params.max_num_polys_per_commitment_group,
            default_layout_digest: params.default_layout_digest,
            serialized_backend_bytes: serialize_akita(&backend_verifier_setup)?,
            serialized_one_hot_backend_bytes: one_hot_verifier_bytes,
            backend_cache: Default::default(),
        };
        let prover = AkitaProverSetup {
            backend_prover_setup,
            prepared_backend_setup,
            one_hot_backend_prover_setup,
            prepared_one_hot_backend_setup,
            verifier: verifier.clone(),
        };
        Ok((prover, verifier))
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        prover_setup.verifier.clone()
    }

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        if let Some(one_hot) = one_hot_polynomial(poly)? {
            let num_vars = one_hot.num_vars();
            Self::validate_commit_shape(setup, num_vars, 1)?;
            let (backend_prover_setup, prepared_backend_setup) = setup.one_hot_backend()?;
            let stack = backend_stack(backend_prover_setup, prepared_backend_setup)?;
            let (backend_commitment, backend_hint) = AkitaOneHotBackendScheme::commit(
                backend_prover_setup,
                std::slice::from_ref(&one_hot),
                &stack,
            )
            .map_err(commit_failed)?;
            return Self::package_commitment(
                setup.default_layout_digest(),
                num_vars,
                backend_commitment,
                backend_hint,
                AkitaHintPolynomials::OneHot(vec![one_hot].into()),
            );
        }

        if poly.is_one_hot() && Self::supports_unit_sparse_dimension(poly.num_vars()) {
            let mut indices = Vec::new();
            poly.for_each_one(&mut |index| indices.push(index));
            let sparse = sparse_unit_polynomial(poly.num_vars(), indices)?;
            let num_vars = sparse.num_vars();
            Self::validate_commit_shape(setup, num_vars, 1)?;
            let stack = backend_stack(&setup.backend_prover_setup, &setup.prepared_backend_setup)?;
            let (backend_commitment, backend_hint) = AkitaBackendScheme::commit(
                &setup.backend_prover_setup,
                std::slice::from_ref(&sparse),
                &stack,
            )
            .map_err(commit_failed)?;
            return Self::package_commitment(
                setup.default_layout_digest(),
                num_vars,
                backend_commitment,
                backend_hint,
                AkitaHintPolynomials::SparseUnit(vec![sparse].into()),
            );
        }

        let num_vars = poly.num_vars();
        Self::validate_commit_shape(setup, num_vars, 1)?;
        let evals = akita_ordered_evaluations(poly)?;
        let dense =
            vec![AkitaBackendDensePoly::from_field_evals(num_vars, &evals).map_err(akita_error)?];
        Self::commit_dense_backend(setup, setup.default_layout_digest(), num_vars, dense)
    }

    fn open<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::Proof, OpeningsError> {
        let hint = match hint {
            Some(hint) => hint,
            None => Self::commit(poly, setup)?.1,
        };
        let statement = vec![VerifierOpeningClaim {
            commitment: hint.commitment.clone(),
            evaluation: EvaluationClaim::new(point.to_vec(), eval),
        }];
        let witness: AkitaNativeBatchWitness<'_> =
            (vec![&poly as &(dyn MultilinearPoly<AkitaField> + '_)], hint);
        <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            setup, statement, witness, transcript,
        )
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        let statement = vec![VerifierOpeningClaim {
            commitment: commitment.clone(),
            evaluation: EvaluationClaim::new(point.to_vec(), eval),
        }];
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            setup, statement, proof, transcript,
        )
    }
}

impl ZkOpeningScheme for AkitaScheme {
    type HidingCommitment = AkitaHidingCommitment;
    type Blind = ();

    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Output, Self::OpeningHint), OpeningsError> {
        Self::commit(poly, setup)
    }

    fn open_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError> {
        let proof = Self::open(poly, point, eval, setup, Some(hint), transcript)?;
        Ok((
            proof,
            AkitaHidingCommitment::new(eval.to_bytes_le_vec()),
            (),
        ))
    }

    fn verify_zk(
        _commitment: &Self::Output,
        _point: &[Self::Field],
        _proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        Err(transparent_zk_error())
    }
}

impl ZkBatchOpeningScheme for AkitaNativeBatching {
    type Commitment = AkitaCommitment;
    type HidingCommitment = AkitaHidingCommitment;
    type Blind = ();
    type ZkBatchingWitness<'a>
        = AkitaNativeBatchWitness<'a>
    where
        Self: 'a;

    fn prove_batch_zk<'a, T>(
        _setup: &Self::ProverSetup,
        _point: jolt_poly::Point<{ jolt_poly::HIGH_TO_LOW }, Self::Field>,
        _commitments: Vec<Self::Commitment>,
        _witness: Self::ZkBatchingWitness<'a>,
        _transcript: &mut T,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError>
    where
        Self: 'a,
        T: Transcript<Challenge = Self::Field>,
    {
        Err(transparent_zk_error())
    }

    fn verify_batch_zk<T>(
        _setup: &Self::VerifierSetup,
        _point: jolt_poly::Point<{ jolt_poly::HIGH_TO_LOW }, Self::Field>,
        _commitments: Vec<Self::Commitment>,
        _proof: &Self::Proof,
        _transcript: &mut T,
    ) -> Result<Self::HidingCommitment, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        Err(transparent_zk_error())
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests unwrap successful PCS operations")]
    #![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

    use super::*;
    use crate::adapters::{append_verifier_setup, AkitaBackendFlavor};
    use jolt_transcript::Blake2bTranscript;

    #[test]
    fn setup_key_transcript_binds_backend_shape_and_bytes() {
        let setup = AkitaVerifierSetup {
            max_num_vars: 4,
            max_num_polys_per_commitment_group: 1,
            default_layout_digest: [7; 32],
            serialized_backend_bytes: vec![1, 2, 3],
            serialized_one_hot_backend_bytes: None,
            backend_cache: Default::default(),
        };
        let mut baseline = Blake2bTranscript::<AkitaField>::new(b"akita-setup-key-test");
        let initial_state = baseline.state();

        append_verifier_setup(&mut baseline, &setup, AkitaBackendFlavor::Full);
        assert_ne!(baseline.state(), initial_state);

        let mut same = Blake2bTranscript::<AkitaField>::new(b"akita-setup-key-test");
        append_verifier_setup(&mut same, &setup, AkitaBackendFlavor::Full);
        assert_eq!(baseline.state(), same.state());

        let mut changed_shape = setup.clone();
        changed_shape.max_num_vars = 5;
        let mut shape_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-setup-key-test");
        append_verifier_setup(
            &mut shape_transcript,
            &changed_shape,
            AkitaBackendFlavor::Full,
        );
        assert_ne!(baseline.state(), shape_transcript.state());

        let mut changed_backend_bytes = setup;
        changed_backend_bytes.serialized_backend_bytes.push(4);
        let mut backend_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-setup-key-test");
        append_verifier_setup(
            &mut backend_transcript,
            &changed_backend_bytes,
            AkitaBackendFlavor::Full,
        );
        assert_ne!(baseline.state(), backend_transcript.state());
    }

    #[test]
    fn direct_opening_requires_statement_commitment_layout_digest() {
        let setup_params = AkitaSetupParams::new(1, 1, [7; 32]);
        let (prover_setup, verifier_setup) = AkitaScheme::setup(setup_params).unwrap();
        let polynomial = Polynomial::new(vec![AkitaField::from_u64(2), AkitaField::from_u64(5)]);
        let commitment_digest = [9; 32];
        let (commitment, hint) = AkitaScheme::commit_group(
            &prover_setup,
            commitment_digest,
            std::slice::from_ref(&polynomial),
        )
        .expect("direct commitment may use its own layout digest");
        assert_eq!(commitment.layout_digest, commitment_digest);

        let point = vec![AkitaField::from_u64(3)];
        let claim = polynomial.evaluate(&point);
        let statement = vec![VerifierOpeningClaim {
            commitment: commitment.clone(),
            evaluation: EvaluationClaim::new(point.clone(), claim),
        }];

        let mut prover_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-direct-layout");
        let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement.clone(),
            (vec![&polynomial], hint),
            &mut prover_transcript,
        )
        .expect("direct proof should prove");

        let mut verifier_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-direct-layout");
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            statement.clone(),
            &proof,
            &mut verifier_transcript,
        )
        .expect("direct proof should verify");
        assert_eq!(prover_transcript.state(), verifier_transcript.state());

        let mut changed_commitment_statement = statement.clone();
        changed_commitment_statement[0].commitment.layout_digest = [15; 32];
        let mut verifier_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-direct-layout");
        let _error = <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            changed_commitment_statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect_err("changed direct commitment digest should reject");

        let mut changed_setup = verifier_setup;
        changed_setup.default_layout_digest = commitment_digest;
        let mut verifier_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-direct-layout");
        let _error = <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &changed_setup,
            statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect_err("direct commitment layout must not be accepted through setup default");
    }
}
