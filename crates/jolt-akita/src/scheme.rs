use akita_pcs::{CommitmentProver, ComputeBackendSetup, CpuBackend, RootPolyShape};
use jolt_crypto::Commitment;
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, OpeningsError, VerifierOpeningClaim,
    ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::adapters::{
    akita_error, dense_polynomials, field_bytes, invalid_batch, one_hot_polynomial,
    polynomial_evaluations, serialize_akita, transparent_zk_error, AkitaBackendFlavor,
    AkitaBackendScheme, AkitaBatchProof, AkitaCommitment, AkitaField, AkitaHidingCommitment,
    AkitaHintPolynomials, AkitaOneHotBackendScheme, AkitaProverHint, AkitaProverSetup,
    AkitaSetupParams, AkitaSourceKind, AkitaSparsePolynomial, AkitaVerifierSetup, AKITA_D,
    AKITA_ONE_HOT_LOG_K,
};
use crate::black_box_batching::{AkitaBlackBoxBatchWitness, AkitaBlackBoxBatching};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaScheme;

impl AkitaScheme {
    /// Returns true when the Akita backend sparse-ring path can represent a
    /// unit-valued sparse polynomial with this multilinear dimension.
    pub fn supports_unit_sparse_dimension(num_vars: usize) -> bool {
        let Some(domain_size) = u32::try_from(num_vars)
            .ok()
            .and_then(|shift| 1usize.checked_shl(shift))
        else {
            return false;
        };
        domain_size >= AKITA_D
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

        // Validate the commitment shape before handing values to Akita.
        if num_vars > setup.max_num_vars {
            return Err(OpeningsError::PolynomialTooLarge {
                poly_size: num_vars,
                setup_max: setup.max_num_vars,
            });
        }
        if num_vars != setup.max_num_vars {
            return Err(invalid_batch(format!(
                "Akita commitment dimension {num_vars} does not match exact setup dimension {}",
                setup.max_num_vars
            )));
        }
        if polynomials.len() > setup.max_num_polys_per_commitment_group {
            return Err(invalid_batch(format!(
                "Akita commitment group has {} polynomials but setup supports {}",
                polynomials.len(),
                setup.max_num_polys_per_commitment_group
            )));
        }
        for polynomial in polynomials {
            if polynomial.num_vars() != num_vars {
                return Err(invalid_batch(format!(
                    "Akita commitment group mixes {}-variable and {num_vars}-variable polynomials",
                    polynomial.num_vars()
                )));
            }
        }

        let dense = dense_polynomials(polynomials)?;
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &setup.prepared_backend_setup,
            setup.backend_prover_setup.expanded.as_ref(),
        )
        .map_err(akita_error)?;
        let (backend_commitment, backend_hint) =
            AkitaBackendScheme::commit(&setup.backend_prover_setup, dense.as_slice(), &stack)
                .map_err(akita_error)?;
        let commitment = AkitaCommitment {
            backend_flavor: AkitaBackendFlavor::Full,
            layout_digest,
            num_vars,
            poly_count: polynomials.len(),
            serialized_backend_bytes: serialize_akita(&backend_commitment)?,
        };
        Ok((
            commitment.clone(),
            AkitaProverHint {
                commitment,
                backend_commitment: Some(backend_commitment),
                backend_hint: Some(backend_hint),
                backend_polynomials: Some(AkitaHintPolynomials::Dense(dense.into())),
                source_kind: AkitaSourceKind::Dense,
            },
        ))
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

    #[expect(
        clippy::panic,
        reason = "CommitmentScheme::setup cannot return Akita backend setup errors"
    )]
    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
        let backend_prover_setup = AkitaBackendScheme::setup_prover(
            params.max_num_vars,
            params.max_num_polys_per_commitment_group,
        )
        .unwrap_or_else(|err| panic!("Akita setup failed: {err}"));
        let prepared_backend_setup = CpuBackend
            .prepare_setup(&backend_prover_setup)
            .unwrap_or_else(|err| panic!("Akita setup preparation failed: {err}"));
        let backend_verifier_setup = AkitaBackendScheme::setup_verifier(&backend_prover_setup);
        let (one_hot_backend_prover_setup, prepared_one_hot_backend_setup, one_hot_verifier_bytes) =
            if params.max_num_vars >= AKITA_ONE_HOT_LOG_K {
                let backend_prover_setup = AkitaOneHotBackendScheme::setup_prover(
                    params.max_num_vars,
                    params.max_num_polys_per_commitment_group,
                )
                .unwrap_or_else(|err| panic!("Akita one-hot setup failed: {err}"));
                let prepared_backend_setup = CpuBackend
                    .prepare_setup(&backend_prover_setup)
                    .unwrap_or_else(|err| panic!("Akita one-hot setup preparation failed: {err}"));
                let backend_verifier_setup =
                    AkitaOneHotBackendScheme::setup_verifier(&backend_prover_setup);
                let verifier_bytes =
                    serialize_akita(&backend_verifier_setup).unwrap_or_else(|err| {
                        panic!("Akita one-hot verifier setup serialization failed: {err}")
                    });
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
            serialized_backend_bytes: serialize_akita(&backend_verifier_setup)
                .unwrap_or_else(|err| panic!("Akita verifier setup serialization failed: {err}")),
            serialized_one_hot_backend_bytes: one_hot_verifier_bytes,
        };
        let prover = AkitaProverSetup {
            max_num_vars: params.max_num_vars,
            max_num_polys_per_commitment_group: params.max_num_polys_per_commitment_group,
            default_layout_digest: params.default_layout_digest,
            backend_prover_setup,
            prepared_backend_setup,
            one_hot_backend_prover_setup,
            prepared_one_hot_backend_setup,
            verifier: verifier.clone(),
        };
        (prover, verifier)
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        prover_setup.verifier.clone()
    }

    #[expect(
        clippy::panic,
        reason = "CommitmentScheme::commit cannot return Akita backend commit errors"
    )]
    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        if let Some(one_hot) = one_hot_polynomial(poly)
            .unwrap_or_else(|err| panic!("Akita one-hot commit failed: {err}"))
        {
            assert!(
                one_hot.num_vars() <= setup.max_num_vars,
                "Akita one-hot commit failed: polynomial dimension {} exceeds setup dimension {}",
                one_hot.num_vars(),
                setup.max_num_vars
            );
            assert_eq!(
                one_hot.num_vars(),
                setup.max_num_vars,
                "Akita one-hot commit failed: commitment dimension {} does not match exact setup dimension {}",
                one_hot.num_vars(),
                setup.max_num_vars
            );
            assert_ne!(
                setup.max_num_polys_per_commitment_group,
                0,
                "Akita one-hot commit failed: commitment group has 1 polynomial but setup supports 0"
            );
            let backend_prover_setup =
                setup
                    .one_hot_backend_prover_setup
                    .as_ref()
                    .unwrap_or_else(|| {
                        panic!("Akita one-hot commit failed: setup has no one-hot backend")
                    });
            let prepared_backend_setup = setup
                .prepared_one_hot_backend_setup
                .as_ref()
                .unwrap_or_else(|| {
                    panic!("Akita one-hot commit failed: setup has no prepared one-hot backend")
                });
            let one_hot_num_vars = one_hot.num_vars();
            let backend_polynomials = std::slice::from_ref(&one_hot);
            let stack = akita_prover::UniformProverStack::uniform(
                &CpuBackend,
                prepared_backend_setup,
                backend_prover_setup.expanded.as_ref(),
            )
            .unwrap_or_else(|err| panic!("Akita one-hot commit failed: {err}"));
            let (backend_commitment, backend_hint) =
                AkitaOneHotBackendScheme::commit(backend_prover_setup, backend_polynomials, &stack)
                    .unwrap_or_else(|err| panic!("Akita one-hot commit failed: {err}"));
            let commitment = AkitaCommitment {
                backend_flavor: AkitaBackendFlavor::OneHot,
                layout_digest: setup.default_layout_digest,
                num_vars: one_hot_num_vars,
                poly_count: 1,
                serialized_backend_bytes: serialize_akita(&backend_commitment)
                    .unwrap_or_else(|err| panic!("Akita one-hot commit failed: {err}")),
            };
            return (
                commitment.clone(),
                AkitaProverHint {
                    commitment,
                    backend_commitment: Some(backend_commitment),
                    backend_hint: Some(backend_hint),
                    backend_polynomials: Some(AkitaHintPolynomials::OneHot(vec![one_hot].into())),
                    source_kind: AkitaSourceKind::OneHot,
                },
            );
        }

        if poly.is_one_hot() && Self::supports_unit_sparse_dimension(poly.num_vars()) {
            let mut indices = Vec::new();
            poly.for_each_one(&mut |index| indices.push(index));
            let sparse = AkitaSparsePolynomial::from_jolt_unit_indices(poly.num_vars(), indices)
                .unwrap_or_else(|err| panic!("Akita sparse commit failed: {err}"));

            // Validate the sparse commitment shape before handing values to Akita.
            assert!(
                sparse.num_vars() <= setup.max_num_vars,
                "Akita sparse commit failed: polynomial dimension {} exceeds setup dimension {}",
                sparse.num_vars(),
                setup.max_num_vars
            );
            assert_eq!(
                sparse.num_vars(),
                setup.max_num_vars,
                "Akita sparse commit failed: commitment dimension {} does not match exact setup dimension {}",
                sparse.num_vars(),
                setup.max_num_vars
            );
            assert_ne!(
                setup.max_num_polys_per_commitment_group,
                0,
                "Akita sparse commit failed: commitment group has 1 polynomial but setup supports 0"
            );
            let sparse_num_vars = sparse.num_vars();
            let backend_polynomial = sparse.backend_polynomial;
            let backend_polynomials = std::slice::from_ref(&backend_polynomial);
            let stack = akita_prover::UniformProverStack::uniform(
                &CpuBackend,
                &setup.prepared_backend_setup,
                setup.backend_prover_setup.expanded.as_ref(),
            )
            .unwrap_or_else(|err| panic!("Akita sparse commit failed: {err}"));
            let (backend_commitment, backend_hint) = AkitaBackendScheme::commit(
                &setup.backend_prover_setup,
                backend_polynomials,
                &stack,
            )
            .unwrap_or_else(|err| panic!("Akita sparse commit failed: {err}"));
            let commitment = AkitaCommitment {
                backend_flavor: AkitaBackendFlavor::Full,
                layout_digest: setup.default_layout_digest,
                num_vars: sparse_num_vars,
                poly_count: 1,
                serialized_backend_bytes: serialize_akita(&backend_commitment)
                    .unwrap_or_else(|err| panic!("Akita sparse commit failed: {err}")),
            };
            return (
                commitment.clone(),
                AkitaProverHint {
                    commitment,
                    backend_commitment: Some(backend_commitment),
                    backend_hint: Some(backend_hint),
                    backend_polynomials: Some(AkitaHintPolynomials::SparseUnit(
                        vec![backend_polynomial].into(),
                    )),
                    source_kind: AkitaSourceKind::SparseUnit,
                },
            );
        }

        let polynomial = Polynomial::from(polynomial_evaluations(poly));
        Self::commit_group(setup, setup.default_layout_digest, &[polynomial])
            .unwrap_or_else(|err| panic!("Akita commit failed: {err}"))
    }

    #[expect(
        clippy::panic,
        reason = "CommitmentScheme::open cannot return Akita backend prove errors"
    )]
    fn open<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        let hint = hint.unwrap_or_else(|| Self::commit(poly, setup).1);
        let statement = vec![VerifierOpeningClaim {
            commitment: hint.commitment.clone(),
            evaluation: EvaluationClaim::new(point.to_vec(), eval),
        }];
        let witness: AkitaBlackBoxBatchWitness<'_> =
            (vec![&poly as &(dyn MultilinearPoly<AkitaField> + '_)], hint);
        <AkitaBlackBoxBatching as BatchOpeningScheme>::prove_batch(
            setup, statement, witness, transcript,
        )
        .unwrap_or_else(|err| panic!("Akita open failed: {err}"))
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
        <AkitaBlackBoxBatching as BatchOpeningScheme>::verify_batch(
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
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit(poly, setup)
    }

    fn open_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        let proof = Self::open(poly, point, eval, setup, Some(hint), transcript);
        (proof, AkitaHidingCommitment::new(field_bytes(eval)), ())
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

impl ZkBatchOpeningScheme for AkitaBlackBoxBatching {
    type Commitment = AkitaCommitment;
    type HidingCommitment = AkitaHidingCommitment;
    type Blind = ();
    type ZkBatchingWitness<'a>
        = (
        Vec<&'a (dyn MultilinearPoly<AkitaField> + 'a)>,
        AkitaProverHint,
    )
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

    #![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

    use super::*;
    use jolt_transcript::Blake2bTranscript;

    #[test]
    fn setup_key_transcript_binds_backend_shape_and_bytes() {
        let setup = AkitaVerifierSetup {
            max_num_vars: 4,
            max_num_polys_per_commitment_group: 1,
            default_layout_digest: [7; 32],
            serialized_backend_bytes: vec![1, 2, 3],
            serialized_one_hot_backend_bytes: None,
        };
        let mut baseline = Blake2bTranscript::<AkitaField>::new(b"akita-setup-key-test");
        let initial_state = baseline.state();

        baseline.append(&setup);
        assert_ne!(baseline.state(), initial_state);

        let mut same = Blake2bTranscript::<AkitaField>::new(b"akita-setup-key-test");
        same.append(&setup);
        assert_eq!(baseline.state(), same.state());

        let mut changed_shape = setup.clone();
        changed_shape.max_num_vars = 5;
        let mut shape_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-setup-key-test");
        shape_transcript.append(&changed_shape);
        assert_ne!(baseline.state(), shape_transcript.state());

        let mut changed_backend_bytes = setup;
        changed_backend_bytes.serialized_backend_bytes.push(4);
        let mut backend_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-setup-key-test");
        backend_transcript.append(&changed_backend_bytes);
        assert_ne!(baseline.state(), backend_transcript.state());
    }

    #[test]
    fn direct_opening_requires_statement_commitment_layout_digest() {
        let setup_params = AkitaSetupParams::new(1, 1, [7; 32]);
        let (prover_setup, verifier_setup) = AkitaScheme::setup(setup_params);
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
        let proof = <AkitaBlackBoxBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement.clone(),
            (vec![&polynomial], hint),
            &mut prover_transcript,
        )
        .expect("direct proof should prove");

        let mut verifier_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-direct-layout");
        <AkitaBlackBoxBatching as BatchOpeningScheme>::verify_batch(
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
        let _error = <AkitaBlackBoxBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            changed_commitment_statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect_err("changed direct commitment digest should reject");

        let mut changed_setup = verifier_setup;
        changed_setup.default_layout_digest = commitment_digest;
        let mut verifier_transcript = Blake2bTranscript::<AkitaField>::new(b"akita-direct-layout");
        let _error = <AkitaBlackBoxBatching as BatchOpeningScheme>::verify_batch(
            &changed_setup,
            statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect_err("direct commitment layout must not be accepted through setup default");
    }
}
