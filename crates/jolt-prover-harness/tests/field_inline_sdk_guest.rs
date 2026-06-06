#[cfg(all(feature = "core-fixtures", feature = "field-inline"))]
mod tests {
    #[cfg(feature = "zk")]
    use common::constants::MAX_BLINDFOLD_GENERATORS;
    use jolt_backends::cpu::{CpuBackend, CpuBackendConfig};
    use jolt_claims::protocols::field_inline::FieldInlineCommittedPolynomial;
    use jolt_claims::protocols::jolt::{
        formulas::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions,
        JoltFormulaDimensions, JoltOneHotConfig,
    };
    #[cfg(feature = "zk")]
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_crypto::{Commitment, HomomorphicCommitment, VectorCommitment};
    #[cfg(feature = "zk")]
    use jolt_dory::DoryScheme;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_openings::{
        mock::{MockCommitmentScheme, MockHidingCommitment},
        CommitmentScheme,
    };
    use jolt_openings::{
        AdditivelyHomomorphic, OpeningsError, StreamingCommitment, ZkOpeningScheme,
        ZkStreamingCommitment,
    };
    use jolt_poly::{sparse_segments_mle_msb, Polynomial};
    use jolt_program::preprocess::{JoltProgramPreprocessing, PublicInitialRam};
    use jolt_prover::stages::stage0::{prove, CommitmentStageConfig, CommitmentStageInput};
    use jolt_prover::stages::stage1::{
        input::{Stage1ProverConfig, Stage1ProverInput},
        output::{stage1_claims_from_r1cs_inputs, Stage1SumcheckOutput},
        prove::prove as prove_stage1,
    };
    use jolt_prover::stages::stage2::{
        input::{Stage2BatchProverConfig, Stage2ProverInput},
        output::Stage2ProverOutput,
        prove::prove as prove_stage2,
    };
    #[cfg(feature = "zk")]
    use jolt_prover::stages::stage3::prove::prove_committed_boundary as prove_stage3_committed_boundary;
    use jolt_prover::stages::stage3::{
        input::{Stage3ProverConfig, Stage3ProverInput},
        output::Stage3ProverOutput,
        prove::prove as prove_stage3,
    };
    #[cfg(feature = "zk")]
    use jolt_prover::stages::stage4::prove::prove_committed_boundary as prove_stage4_committed_boundary;
    use jolt_prover::stages::stage4::{
        input::{Stage4ProverConfig, Stage4ProverInput},
        output::{Stage4ProverOutput, Stage4RamValCheckInitialEvaluation},
        prove::prove as prove_stage4,
    };
    #[cfg(feature = "zk")]
    use jolt_prover::stages::stage5::prove::prove_committed_boundary as prove_stage5_committed_boundary;
    use jolt_prover::stages::stage5::{
        input::{Stage5ProverConfig, Stage5ProverInput},
        output::Stage5ProverOutput,
        prove::prove as prove_stage5,
    };
    #[cfg(feature = "zk")]
    use jolt_prover::stages::stage6::prove::prove_committed_boundary as prove_stage6_committed_boundary;
    use jolt_prover::stages::stage6::{
        input::{Stage6ProverConfig, Stage6ProverInput},
        output::Stage6ProverOutput,
        prove::prove as prove_stage6,
    };
    #[cfg(feature = "zk")]
    use jolt_prover::stages::stage7::prove::prove_committed_boundary as prove_stage7_committed_boundary;
    use jolt_prover::stages::stage7::{
        input::{Stage7ProverConfig, Stage7ProverInput},
        output::Stage7ProverOutput,
        prove::prove as prove_stage7,
    };
    #[cfg(feature = "zk")]
    use jolt_prover::stages::stage8::prove::prove_stage8_zk;
    use jolt_prover::stages::stage8::{input::Stage8ProverConfig, prove::prove_stage8};
    use jolt_prover_harness::{trace_sdk_guest, SdkGuestTraceRequest};
    #[cfg(feature = "zk")]
    use jolt_r1cs::constraints::jolt::{
        SPARTAN_OUTER_REMAINDER_DEGREE, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
        SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
    };
    use jolt_riscv::JoltInstructionKind;
    use jolt_sumcheck::{ClearProof, ClearSumcheckProof, SumcheckProof};
    #[cfg(feature = "zk")]
    use jolt_sumcheck::{CommittedOutputClaims, CommittedRound, CommittedSumcheckProof};
    use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};
    use jolt_verifier::{
        proof::{JoltProof, JoltProofClaims, JoltStageProofs},
        stages::{
            stage1::Stage1Output, stage2::Stage2Output, stage3::Stage3Output, stage4::Stage4Output,
            stage5::Stage5Output, stage6::Stage6Output, stage7::Stage7Output,
        },
        verify_until_stage1, JoltVerifierPreprocessing,
    };
    use jolt_witness::protocols::jolt_vm::{
        field_inline::{FieldInlineNamespace, TraceBackedFieldInlineWitness},
        JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackedJoltVmWitness,
        RV64_LOOKUP_ADDRESS_BITS,
    };
    use jolt_witness::{CommittedWitnessProvider, OracleRef, PolynomialEncoding, WitnessProvider};

    type MockPcs = MockCommitmentScheme<Fr>;

    #[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct Stage8RoundHidingMockPcs;

    impl Commitment for Stage8RoundHidingMockPcs {
        type Output = <MockPcs as Commitment>::Output;
    }

    impl CommitmentScheme for Stage8RoundHidingMockPcs {
        type Field = Fr;
        type Proof = <MockPcs as CommitmentScheme>::Proof;
        type ProverSetup = <MockPcs as CommitmentScheme>::ProverSetup;
        type VerifierSetup = <MockPcs as CommitmentScheme>::VerifierSetup;
        type Polynomial = <MockPcs as CommitmentScheme>::Polynomial;
        type OpeningHint = <MockPcs as CommitmentScheme>::OpeningHint;
        type SetupParams = <MockPcs as CommitmentScheme>::SetupParams;

        fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
            <MockPcs as CommitmentScheme>::setup(params)
        }

        fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
            <MockPcs as CommitmentScheme>::verifier_setup(prover_setup);
        }

        fn commit<P: jolt_poly::MultilinearPoly<Self::Field> + ?Sized>(
            poly: &P,
            setup: &Self::ProverSetup,
        ) -> (Self::Output, Self::OpeningHint) {
            <MockPcs as CommitmentScheme>::commit(poly, setup)
        }

        fn open(
            poly: &Self::Polynomial,
            point: &[Self::Field],
            eval: Self::Field,
            setup: &Self::ProverSetup,
            hint: Option<Self::OpeningHint>,
            transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Self::Proof {
            <MockPcs as CommitmentScheme>::open(poly, point, eval, setup, hint, transcript)
        }

        fn verify(
            commitment: &Self::Output,
            point: &[Self::Field],
            eval: Self::Field,
            proof: &Self::Proof,
            setup: &Self::VerifierSetup,
            transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Result<(), OpeningsError> {
            <MockPcs as CommitmentScheme>::verify(commitment, point, eval, proof, setup, transcript)
        }

        fn bind_opening_inputs(
            transcript: &mut impl Transcript<Challenge = Self::Field>,
            point: &[Self::Field],
            eval: &Self::Field,
        ) {
            <MockPcs as CommitmentScheme>::bind_opening_inputs(transcript, point, eval);
        }
    }

    impl AdditivelyHomomorphic for Stage8RoundHidingMockPcs {
        fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output {
            <MockPcs as AdditivelyHomomorphic>::combine(commitments, scalars)
        }

        fn combine_hints(
            hints: Vec<Self::OpeningHint>,
            scalars: &[Self::Field],
        ) -> Self::OpeningHint {
            <MockPcs as AdditivelyHomomorphic>::combine_hints(hints, scalars);
        }
    }

    impl StreamingCommitment for Stage8RoundHidingMockPcs {
        type PartialCommitment = <MockPcs as StreamingCommitment>::PartialCommitment;

        fn begin(setup: &Self::ProverSetup) -> Self::PartialCommitment {
            <MockPcs as StreamingCommitment>::begin(setup)
        }

        fn feed(
            partial: &mut Self::PartialCommitment,
            chunk: &[Self::Field],
            setup: &Self::ProverSetup,
        ) {
            <MockPcs as StreamingCommitment>::feed(partial, chunk, setup);
        }

        fn finish(partial: Self::PartialCommitment, setup: &Self::ProverSetup) -> Self::Output {
            <MockPcs as StreamingCommitment>::finish(partial, setup)
        }

        fn finish_with_hint(
            partial: Self::PartialCommitment,
            setup: &Self::ProverSetup,
        ) -> (Self::Output, Self::OpeningHint) {
            <MockPcs as StreamingCommitment>::finish_with_hint(partial, setup)
        }
    }

    impl ZkOpeningScheme for Stage8RoundHidingMockPcs {
        type HidingCommitment = MockRoundCommitment;
        type Blind = ();

        fn commit_zk<P: jolt_poly::MultilinearPoly<Self::Field> + ?Sized>(
            poly: &P,
            setup: &Self::ProverSetup,
        ) -> (Self::Output, Self::OpeningHint) {
            <MockPcs as ZkOpeningScheme>::commit_zk(poly, setup)
        }

        fn open_zk(
            poly: &Self::Polynomial,
            point: &[Self::Field],
            eval: Self::Field,
            setup: &Self::ProverSetup,
            hint: Self::OpeningHint,
            transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
            let proof = <MockPcs as CommitmentScheme>::open(
                poly,
                point,
                eval,
                setup,
                Some(hint),
                transcript,
            );
            (proof, MockRoundCommitment(point.len() as u64), ())
        }

        fn verify_zk(
            _commitment: &Self::Output,
            point: &[Self::Field],
            _proof: &Self::Proof,
            _setup: &Self::VerifierSetup,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Result<Self::HidingCommitment, OpeningsError> {
            Ok(MockRoundCommitment(point.len() as u64))
        }

        fn bind_zk_opening_inputs(
            transcript: &mut impl Transcript<Challenge = Self::Field>,
            point: &[Self::Field],
            hiding_commitment: &Self::HidingCommitment,
        ) {
            transcript.append_bytes(b"stage8_round_hiding_mock_zk_opening");
            transcript.append_bytes(&(point.len() as u64).to_be_bytes());
            for value in point {
                value.append_to_transcript(transcript);
            }
            hiding_commitment.append_to_transcript(transcript);
        }
    }

    impl ZkStreamingCommitment for Stage8RoundHidingMockPcs {
        fn finish_zk_with_hint(
            partial: Self::PartialCommitment,
            setup: &Self::ProverSetup,
        ) -> (Self::Output, Self::OpeningHint) {
            <MockPcs as ZkStreamingCommitment>::finish_zk_with_hint(partial, setup)
        }
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct MockRoundCommitment(u64);

    impl AppendToTranscript for MockRoundCommitment {
        fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
            transcript.append_bytes(&self.0.to_be_bytes());
        }
    }

    impl HomomorphicCommitment<Fr> for MockRoundCommitment {
        fn add(c1: &Self, c2: &Self) -> Self {
            Self(c1.0.wrapping_add(c2.0))
        }

        fn linear_combine(c1: &Self, c2: &Self, scalar: &Fr) -> Self {
            let _ = scalar;
            Self(c1.0.wrapping_add(c2.0))
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct MockVectorCommitment;

    impl Commitment for MockVectorCommitment {
        type Output = MockRoundCommitment;
    }

    impl VectorCommitment for MockVectorCommitment {
        type Field = Fr;
        type Setup = usize;

        fn capacity(setup: &Self::Setup) -> usize {
            *setup
        }

        fn commit(
            _setup: &Self::Setup,
            values: &[Self::Field],
            _blinding: &Self::Field,
        ) -> Self::Output {
            MockRoundCommitment(values.len() as u64)
        }

        fn verify(
            _setup: &Self::Setup,
            _commitment: &Self::Output,
            _values: &[Self::Field],
            _blinding: &Self::Field,
        ) -> bool {
            true
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct MockHidingVectorCommitment;

    impl Commitment for MockHidingVectorCommitment {
        type Output = MockHidingCommitment<Fr>;
    }

    impl VectorCommitment for MockHidingVectorCommitment {
        type Field = Fr;
        type Setup = usize;

        fn capacity(setup: &Self::Setup) -> usize {
            *setup
        }

        fn commit(
            _setup: &Self::Setup,
            values: &[Self::Field],
            _blinding: &Self::Field,
        ) -> Self::Output {
            MockHidingCommitment {
                eval: Fr::from_u64(values.len() as u64),
            }
        }

        fn verify(
            _setup: &Self::Setup,
            _commitment: &Self::Output,
            _values: &[Self::Field],
            _blinding: &Self::Field,
        ) -> bool {
            true
        }
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn top_level_field_inline_clear_prover_outputs_verify() -> Result<(), String> {
        let inputs =
            postcard::to_stdvec(&7u32).map_err(|error| format!("serialize input: {error}"))?;
        let fixture = trace_sdk_guest(
            SdkGuestTraceRequest::new("field-inline-eq-poly-guest", inputs)
                .with_field_inline(true)
                .with_max_padded_trace_length(1 << 9),
        )
        .map_err(|error| error.to_string())?;

        let one_hot = JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        };
        let log_t = fixture.padded_trace_length.trailing_zeros() as usize;
        let bytecode_start = fixture
            .preprocessing
            .memory_layout
            .remapped_word_address(fixture.preprocessing.ram.min_bytecode_address)
            .map_err(|error| error.to_string())? as usize;
        let ram_k = (bytecode_start + fixture.preprocessing.ram.bytecode_words.len())
            .next_power_of_two()
            .max(1);
        let witness_config = JoltVmWitnessConfig::new(log_t, ram_k, one_hot);
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&fixture.program, &fixture.preprocessing, fixture.trace),
        );
        let field_witness = witness
            .field_inline_witness()
            .map_err(|error| error.to_string())?;
        let verifier_bytecode = verifier_field_inline_bytecode_rows(&fixture.preprocessing)?;
        let verifier_preprocessing = JoltVerifierPreprocessing::<
            Stage8RoundHidingMockPcs,
            MockVectorCommitment,
        >::new(
            fixture.preprocessing.clone(), [7; 32], (), None
        )
        .with_field_inline_bytecode(verifier_bytecode);
        let prover_preprocessing =
            jolt_prover::JoltProverPreprocessing::new(verifier_preprocessing.clone(), ());
        let public_io = witness.trace.device.clone();
        let proof_shape = jolt_prover::ProverProofShape::new(
            fixture.padded_trace_length,
            ram_k,
            jolt_claims::protocols::jolt::JoltReadWriteConfig {
                ram_rw_phase1_num_rounds: 1,
                ram_rw_phase2_num_rounds: 1,
                registers_rw_phase1_num_rounds: 1,
                registers_rw_phase2_num_rounds: 1,
            },
            one_hot,
            jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
        );
        let config = jolt_prover::ProverConfig::default().with_proof_shape(proof_shape);
        let mut backend = CpuBackend::new(CpuBackendConfig {
            preserve_core_fast_path: true,
            commitment_chunk_size: 1 << 10,
        });

        let output = jolt_prover::prove_with_output(
            &prover_preprocessing,
            &public_io,
            &witness,
            &field_witness,
            config,
            &mut backend,
        )
        .map_err(|error| error.to_string())?;
        assert!(output.trusted_advice_commitment.is_none());
        jolt_verifier::verify::<
            Fr,
            Stage8RoundHidingMockPcs,
            MockVectorCommitment,
            Blake2bTranscript<Fr>,
        >(
            &verifier_preprocessing,
            &public_io,
            &output.proof,
            None,
            false,
        )
        .map_err(|error| error.to_string())?;

        Ok(())
    }

    #[test]
    #[cfg(feature = "zk")]
    fn top_level_field_inline_zk_prover_outputs_verify() -> Result<(), String> {
        with_zk_field_inline_stack(
            "top_level_field_inline_zk_prover_outputs_verify",
            top_level_field_inline_zk_prover_outputs_verify_inner,
        )
    }

    #[cfg(feature = "zk")]
    fn top_level_field_inline_zk_prover_outputs_verify_inner() -> Result<(), String> {
        let inputs =
            postcard::to_stdvec(&7u32).map_err(|error| format!("serialize input: {error}"))?;
        let fixture = trace_sdk_guest(
            SdkGuestTraceRequest::new("field-inline-eq-poly-guest", inputs)
                .with_field_inline(true)
                .with_max_padded_trace_length(1 << 9),
        )
        .map_err(|error| error.to_string())?;

        let one_hot = JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        };
        let log_t = fixture.padded_trace_length.trailing_zeros() as usize;
        let committed_chunk_bits = one_hot.committed_chunk_bits();
        let bytecode_start = fixture
            .preprocessing
            .memory_layout
            .remapped_word_address(fixture.preprocessing.ram.min_bytecode_address)
            .map_err(|error| error.to_string())? as usize;
        let ram_k = (bytecode_start + fixture.preprocessing.ram.bytecode_words.len())
            .next_power_of_two()
            .max(1);
        let witness_config = JoltVmWitnessConfig::new(log_t, ram_k, one_hot);
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&fixture.program, &fixture.preprocessing, fixture.trace),
        );
        let field_witness = witness
            .field_inline_witness()
            .map_err(|error| error.to_string())?;
        let verifier_bytecode = verifier_field_inline_bytecode_rows(&fixture.preprocessing)?;
        let vc_capacity = MAX_BLINDFOLD_GENERATORS.max(64);
        let max_num_vars = (log_t + committed_chunk_bits).max(vc_capacity.ilog2() as usize);
        let pcs_setup = DoryScheme::setup_prover(max_num_vars);
        let verifier_preprocessing =
            JoltVerifierPreprocessing::<DoryScheme, Pedersen<Bn254G1>>::from_pcs_prover_setup(
                fixture.preprocessing.clone(),
                [7; 32],
                &pcs_setup,
                vc_capacity,
            )
            .with_field_inline_bytecode(verifier_bytecode);
        let prover_preprocessing =
            jolt_prover::JoltProverPreprocessing::new(verifier_preprocessing.clone(), pcs_setup);
        let public_io = witness.trace.device.clone();
        let proof_shape = jolt_prover::ProverProofShape::new(
            fixture.padded_trace_length,
            ram_k,
            jolt_claims::protocols::jolt::JoltReadWriteConfig {
                ram_rw_phase1_num_rounds: 1,
                ram_rw_phase2_num_rounds: 1,
                registers_rw_phase1_num_rounds: 1,
                registers_rw_phase2_num_rounds: 1,
            },
            one_hot,
            jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
        );
        let config = jolt_prover::ProverConfig::default().with_proof_shape(proof_shape);
        let mut backend = CpuBackend::new(CpuBackendConfig {
            preserve_core_fast_path: true,
            commitment_chunk_size: 1 << 10,
        });

        let output = jolt_prover::prove_with_output::<DoryScheme, Pedersen<Bn254G1>, _, _, _>(
            &prover_preprocessing,
            &public_io,
            &witness,
            &field_witness,
            config,
            &mut backend,
        )
        .map_err(|error| error.to_string())?;
        assert!(output.trusted_advice_commitment.is_none());

        jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript<Fr>>(
            &verifier_preprocessing,
            &public_io,
            &output.proof,
            None,
            true,
        )
        .map_err(|error| error.to_string())?;

        Ok(())
    }

    #[test]
    fn field_inline_eq_poly_guest_replays_modular_frontier_with_jolt_verifier() -> Result<(), String>
    {
        let inputs =
            postcard::to_stdvec(&7u32).map_err(|error| format!("serialize input: {error}"))?;
        let fixture = trace_sdk_guest(
            SdkGuestTraceRequest::new("field-inline-eq-poly-guest", inputs)
                .with_field_inline(true)
                .with_max_padded_trace_length(1 << 9),
        )
        .map_err(|error| error.to_string())?;

        let field_rows = fixture
            .trace
            .trace
            .rows()
            .iter()
            .filter(|row| row.field_inline.is_some())
            .count();
        assert_eq!(field_rows, 14);
        let metadata = fixture
            .preprocessing
            .bytecode
            .field_inline
            .as_ref()
            .ok_or_else(|| "field-inline preprocessing metadata is missing".to_owned())?;
        assert!(metadata.rows.iter().any(|row| row.active));

        let one_hot = JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        };
        let log_t = fixture.padded_trace_length.trailing_zeros() as usize;
        let committed_chunk_bits = one_hot.committed_chunk_bits();
        let bytecode_start = fixture
            .preprocessing
            .memory_layout
            .remapped_word_address(fixture.preprocessing.ram.min_bytecode_address)
            .map_err(|error| error.to_string())? as usize;
        let ram_k = (bytecode_start + fixture.preprocessing.ram.bytecode_words.len())
            .next_power_of_two()
            .max(1);
        let log_k = ram_k.trailing_zeros() as usize;
        let witness_config = JoltVmWitnessConfig::new(log_t, ram_k, one_hot);
        let dimensions = JoltFormulaDimensions::try_from(one_hot.dimensions(
            witness_config.log_t,
            RV64_LOOKUP_ADDRESS_BITS,
            fixture.preprocessing.bytecode.code_size,
            witness_config.ram_k,
        ))
        .map_err(|error| error.to_string())?;
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&fixture.program, &fixture.preprocessing, fixture.trace),
        );
        let field_witness = witness
            .field_inline_witness()
            .map_err(|error| error.to_string())?;

        assert_eq!(
            <TraceBackedFieldInlineWitness<'_, '_, _> as CommittedWitnessProvider<
                Fr,
                FieldInlineNamespace,
            >>::committed_oracle_order(&field_witness)
            .map_err(|error| error.to_string())?,
            vec![FieldInlineCommittedPolynomial::FieldRdInc]
        );
        let descriptor = <TraceBackedFieldInlineWitness<'_, '_, _> as WitnessProvider<
            Fr,
            FieldInlineNamespace,
        >>::describe_oracle(
            &field_witness,
            OracleRef::committed(FieldInlineCommittedPolynomial::FieldRdInc),
        )
        .map_err(|error| error.to_string())?;
        assert_eq!(descriptor.dimensions.rows, fixture.padded_trace_length);
        assert_eq!(descriptor.encoding, PolynomialEncoding::Dense);

        let mut backend = CpuBackend::new(CpuBackendConfig {
            preserve_core_fast_path: true,
            commitment_chunk_size: 1 << 10,
        });
        let stage0 = prove::<Fr, _, _, MockPcs>(
            CommitmentStageInput::new(
                &witness,
                &(),
                CommitmentStageConfig::new(dimensions.ra_layout, false, false)
                    .with_final_opening_trace_embedding(
                        log_t,
                        committed_chunk_bits,
                        jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
                    ),
                jolt_verifier::JoltProtocolConfig::for_zk(false),
                &field_witness,
            ),
            &mut backend,
        )
        .map_err(|error| error.to_string())?;

        let verifier_bytecode = verifier_field_inline_bytecode_rows(&fixture.preprocessing)?;
        let verifier_preprocessing =
            JoltVerifierPreprocessing::<MockPcs, MockVectorCommitment>::new(
                fixture.preprocessing.clone(),
                [7; 32],
                (),
                None,
            )
            .with_field_inline_bytecode(verifier_bytecode.clone());
        let public_io = witness.trace.device.clone();
        let empty_proof = frontier_proof(
            stage0.commitments.clone(),
            empty_stage_proofs(),
            jolt_verifier::compat::claims::empty_clear_opening_claims(fixture.padded_trace_length),
            fixture.padded_trace_length,
            ram_k,
        );
        let mut prover_state =
            verify_until_stage1::<MockPcs, MockVectorCommitment, Blake2bTranscript<Fr>, _>(
                &verifier_preprocessing,
                &public_io,
                &empty_proof,
                None,
                false,
            )
            .map_err(|error| error.to_string())?;

        let stage1 = prove_stage1::<Fr, _, _, _, _, MockRoundCommitment>(
            Stage1ProverInput::new(Stage1ProverConfig::new(log_t), &witness, &field_witness),
            &mut backend,
            &mut prover_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let stage1_claims = stage1_claims_from_r1cs_inputs(
            stage1.uniskip_output_claim,
            &stage1.r1cs_input_claims,
            &stage1.field_inline_r1cs_input_claims,
        )
        .map_err(|error| error.to_string())?;
        let proof = frontier_proof(
            stage0.commitments,
            stage_proofs(stage1.clone()),
            {
                let mut claims = jolt_verifier::compat::claims::empty_clear_opening_claims(
                    fixture.padded_trace_length,
                );
                claims.stage1 = stage1_claims.clone();
                claims
            },
            fixture.padded_trace_length,
            ram_k,
        );

        let mut verifier_state = verify_until_stage1::<
            MockPcs,
            MockVectorCommitment,
            Blake2bTranscript<Fr>,
            _,
        >(&verifier_preprocessing, &public_io, &proof, None, false)
        .map_err(|error| error.to_string())?;
        let stage1_output = jolt_verifier::stages::stage1::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
        )
        .map_err(|error| error.to_string())?;

        let Stage1Output::Clear(stage1_clear) = stage1_output else {
            return Err("field-inline Stage 1 verifier did not return clear output".to_owned());
        };
        assert_eq!(
            verifier_state.transcript.state(),
            prover_state.transcript.state()
        );

        let stage2 = prove_stage2::<Fr, _, _, _, _, MockRoundCommitment>(
            Stage2ProverInput::new(
                Stage2BatchProverConfig::new(
                    log_t,
                    log_k,
                    jolt_claims::protocols::jolt::JoltReadWriteConfig {
                        ram_rw_phase1_num_rounds: 1,
                        ram_rw_phase2_num_rounds: 1,
                        registers_rw_phase1_num_rounds: 1,
                        registers_rw_phase2_num_rounds: 1,
                    },
                ),
                &verifier_state.checked,
                &stage1_clear,
                &witness,
                &field_witness,
            ),
            &mut backend,
            &mut prover_state.transcript,
        )
        .map_err(|error| error.to_string())?;

        let proof = frontier_proof(
            proof.commitments,
            stage1_stage2_proofs(stage1.clone(), stage2.clone()),
            {
                let mut claims = jolt_verifier::compat::claims::empty_clear_opening_claims(
                    fixture.padded_trace_length,
                );
                claims.stage1 = stage1_claims.clone();
                claims.stage2 = stage2.claims.clone();
                claims
            },
            fixture.padded_trace_length,
            ram_k,
        );
        let mut verifier_state = verify_until_stage1::<
            MockPcs,
            MockVectorCommitment,
            Blake2bTranscript<Fr>,
            _,
        >(&verifier_preprocessing, &public_io, &proof, None, false)
        .map_err(|error| error.to_string())?;
        let stage1_output = jolt_verifier::stages::stage1::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let stage2_output = jolt_verifier::stages::stage2::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage2::inputs::deps(&stage1_output),
        )
        .map_err(|error| error.to_string())?;

        let Stage2Output::Clear(stage2_verified) = stage2_output else {
            return Err("field-inline Stage 2 verifier did not return clear output".to_owned());
        };
        assert_eq!(
            stage2_verified.output_claims,
            stage2.verifier_output.output_claims
        );
        let Stage1Output::Clear(stage1_verified) = stage1_output else {
            return Err("field-inline Stage 1 verifier did not return clear output".to_owned());
        };
        assert_eq!(
            verifier_state.transcript.state(),
            prover_state.transcript.state()
        );

        let stage3 = prove_stage3::<Fr, _, _, _, MockRoundCommitment>(
            Stage3ProverInput::new(
                Stage3ProverConfig::new(log_t),
                &verifier_state.checked,
                &stage1_verified,
                &stage2_verified,
                &witness,
            ),
            &mut backend,
            &mut prover_state.transcript,
        )
        .map_err(|error| error.to_string())?;

        let proof = frontier_proof(
            proof.commitments,
            stage1_stage2_stage3_proofs(stage1.clone(), stage2.clone(), stage3.clone()),
            {
                let mut claims = jolt_verifier::compat::claims::empty_clear_opening_claims(
                    fixture.padded_trace_length,
                );
                claims.stage1 = stage1_claims.clone();
                claims.stage2 = stage2.claims.clone();
                claims.stage3 = stage3.claims.clone();
                claims
            },
            fixture.padded_trace_length,
            ram_k,
        );
        let mut verifier_state = verify_until_stage1::<
            MockPcs,
            MockVectorCommitment,
            Blake2bTranscript<Fr>,
            _,
        >(&verifier_preprocessing, &public_io, &proof, None, false)
        .map_err(|error| error.to_string())?;
        let stage1_output = jolt_verifier::stages::stage1::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let stage2_output = jolt_verifier::stages::stage2::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage2::inputs::deps(&stage1_output),
        )
        .map_err(|error| error.to_string())?;
        let stage3_output = jolt_verifier::stages::stage3::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage3::inputs::deps(&stage1_output, &stage2_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let Stage2Output::Clear(stage2_verified) = stage2_output else {
            return Err("field-inline Stage 2 verifier did not return clear output".to_owned());
        };
        let Stage3Output::Clear(stage3_verified) = stage3_output else {
            return Err("field-inline Stage 3 verifier did not return clear output".to_owned());
        };
        assert_eq!(
            stage3_verified.output_claims,
            stage3.verifier_output.output_claims
        );
        assert_eq!(
            verifier_state.transcript.state(),
            prover_state.transcript.state()
        );

        let ram_val_check_init = stage4_ram_val_check_initial_evaluation(
            &fixture.preprocessing,
            &public_io,
            log_k,
            &stage2_verified,
        )?;
        let stage4 = prove_stage4::<Fr, _, _, _, _, MockRoundCommitment>(
            Stage4ProverInput::new(
                Stage4ProverConfig::new(
                    log_t,
                    log_k,
                    jolt_claims::protocols::jolt::JoltReadWriteConfig {
                        ram_rw_phase1_num_rounds: 1,
                        ram_rw_phase2_num_rounds: 1,
                        registers_rw_phase1_num_rounds: 1,
                        registers_rw_phase2_num_rounds: 1,
                    },
                ),
                &verifier_state.checked,
                &stage2_verified,
                &stage3_verified,
                ram_val_check_init,
                &witness,
                &field_witness,
            ),
            &mut backend,
            &mut prover_state.transcript,
        )
        .map_err(|error| error.to_string())?;

        let proof = frontier_proof(
            proof.commitments,
            stage1_stage2_stage3_stage4_proofs(
                stage1.clone(),
                stage2.clone(),
                stage3.clone(),
                stage4.clone(),
            ),
            {
                let mut claims = jolt_verifier::compat::claims::empty_clear_opening_claims(
                    fixture.padded_trace_length,
                );
                claims.stage1 = stage1_claims.clone();
                claims.stage2 = stage2.claims.clone();
                claims.stage3 = stage3.claims.clone();
                claims.stage4 = stage4.claims.clone();
                claims
            },
            fixture.padded_trace_length,
            ram_k,
        );

        let mut verifier_state = verify_until_stage1::<
            MockPcs,
            MockVectorCommitment,
            Blake2bTranscript<Fr>,
            _,
        >(&verifier_preprocessing, &public_io, &proof, None, false)
        .map_err(|error| error.to_string())?;
        let stage1_output = jolt_verifier::stages::stage1::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let stage2_output = jolt_verifier::stages::stage2::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage2::inputs::deps(&stage1_output),
        )
        .map_err(|error| error.to_string())?;
        let stage3_output = jolt_verifier::stages::stage3::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage3::inputs::deps(&stage1_output, &stage2_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage4_output = jolt_verifier::stages::stage4::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage4::inputs::deps(&stage2_output, &stage3_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let Stage4Output::Clear(stage4_verified) = stage4_output else {
            return Err("field-inline Stage 4 verifier did not return clear output".to_owned());
        };
        assert_eq!(stage4_verified, stage4.verifier_output);
        assert_eq!(
            verifier_state.transcript.state(),
            prover_state.transcript.state()
        );

        let stage5 = prove_stage5::<Fr, _, _, _, _, MockRoundCommitment>(
            Stage5ProverInput::new(
                Stage5ProverConfig::new(log_t, log_k, dimensions.instruction_read_raf),
                &verifier_state.checked,
                &stage2_verified,
                &stage4_verified,
                &witness,
                &field_witness,
            ),
            &mut backend,
            &mut prover_state.transcript,
        )
        .map_err(|error| error.to_string())?;

        let proof = frontier_proof(
            proof.commitments,
            stage1_stage2_stage3_stage4_stage5_proofs(
                stage1.clone(),
                stage2.clone(),
                stage3.clone(),
                stage4.clone(),
                stage5.clone(),
            ),
            {
                let mut claims = jolt_verifier::compat::claims::empty_clear_opening_claims(
                    fixture.padded_trace_length,
                );
                claims.stage1 = stage1_claims.clone();
                claims.stage2 = stage2.claims.clone();
                claims.stage3 = stage3.claims.clone();
                claims.stage4 = stage4.claims.clone();
                claims.stage5 = stage5.claims.clone();
                claims
            },
            fixture.padded_trace_length,
            ram_k,
        );
        let mut verifier_state = verify_until_stage1::<
            MockPcs,
            MockVectorCommitment,
            Blake2bTranscript<Fr>,
            _,
        >(&verifier_preprocessing, &public_io, &proof, None, false)
        .map_err(|error| error.to_string())?;
        let stage1_output = jolt_verifier::stages::stage1::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let stage2_output = jolt_verifier::stages::stage2::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage2::inputs::deps(&stage1_output),
        )
        .map_err(|error| error.to_string())?;
        let stage3_output = jolt_verifier::stages::stage3::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage3::inputs::deps(&stage1_output, &stage2_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage4_output = jolt_verifier::stages::stage4::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage4::inputs::deps(&stage2_output, &stage3_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage5_output = jolt_verifier::stages::stage5::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage5::inputs::deps(&stage2_output, &stage4_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let Stage1Output::Clear(stage1_verified) = stage1_output else {
            return Err("field-inline Stage 1 verifier did not return clear output".to_owned());
        };
        let Stage2Output::Clear(stage2_verified) = stage2_output else {
            return Err("field-inline Stage 2 verifier did not return clear output".to_owned());
        };
        let Stage3Output::Clear(stage3_verified) = stage3_output else {
            return Err("field-inline Stage 3 verifier did not return clear output".to_owned());
        };
        let Stage4Output::Clear(stage4_verified) = stage4_output else {
            return Err("field-inline Stage 4 verifier did not return clear output".to_owned());
        };
        let Stage5Output::Clear(stage5_verified) = stage5_output else {
            return Err("field-inline Stage 5 verifier did not return clear output".to_owned());
        };
        assert_eq!(stage5_verified, stage5.verifier_output);
        assert_eq!(
            verifier_state.transcript.state(),
            prover_state.transcript.state()
        );

        let entry_bytecode_index = fixture
            .preprocessing
            .bytecode
            .entry_bytecode_index()
            .ok_or_else(|| "entry address was not found in bytecode preprocessing".to_owned())?;
        let stage6_config = Stage6ProverConfig::new(
            log_t,
            log_k,
            committed_chunk_bits,
            dimensions.bytecode_read_raf,
            jolt_claims::protocols::jolt::formulas::booleanity::BooleanityDimensions::new(
                dimensions.ra_layout,
                log_t,
                committed_chunk_bits,
            ),
            dimensions.ram_ra_virtualization,
            dimensions.instruction_ra_virtualization,
            None,
            None,
        )
        .with_bytecode_context(
            fixture.preprocessing.bytecode.bytecode.clone(),
            entry_bytecode_index,
        );
        let stage6 = prove_stage6::<Fr, _, _, _, _, MockRoundCommitment>(
            Stage6ProverInput::new(
                &stage6_config,
                &verifier_state.checked,
                &stage1_verified,
                &stage2_verified,
                &stage3_verified,
                &stage4_verified,
                &stage5_verified,
                &witness,
                &field_witness,
            ),
            &mut backend,
            &mut prover_state.transcript,
        )
        .map_err(|error| error.to_string())?;

        let proof = frontier_proof(
            proof.commitments,
            stage1_stage2_stage3_stage4_stage5_stage6_proofs(
                stage1.clone(),
                stage2.clone(),
                stage3.clone(),
                stage4.clone(),
                stage5.clone(),
                stage6.clone(),
            ),
            {
                let mut claims = jolt_verifier::compat::claims::empty_clear_opening_claims(
                    fixture.padded_trace_length,
                );
                claims.stage1 = stage1_claims.clone();
                claims.stage2 = stage2.claims.clone();
                claims.stage3 = stage3.claims.clone();
                claims.stage4 = stage4.claims.clone();
                claims.stage5 = stage5_verified.output_claims.clone();
                claims.stage6 = stage6.claims.clone();
                claims
            },
            fixture.padded_trace_length,
            ram_k,
        );
        let mut verifier_state = verify_until_stage1::<
            MockPcs,
            MockVectorCommitment,
            Blake2bTranscript<Fr>,
            _,
        >(&verifier_preprocessing, &public_io, &proof, None, false)
        .map_err(|error| error.to_string())?;
        let stage1_output = jolt_verifier::stages::stage1::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let stage2_output = jolt_verifier::stages::stage2::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage2::inputs::deps(&stage1_output),
        )
        .map_err(|error| error.to_string())?;
        let stage3_output = jolt_verifier::stages::stage3::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage3::inputs::deps(&stage1_output, &stage2_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage4_output = jolt_verifier::stages::stage4::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage4::inputs::deps(&stage2_output, &stage3_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage5_output = jolt_verifier::stages::stage5::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage5::inputs::deps(&stage2_output, &stage4_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage6_output = jolt_verifier::stages::stage6::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage6::inputs::deps(
                &stage1_output,
                &stage2_output,
                &stage3_output,
                &stage4_output,
                &stage5_output,
            )
            .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let Stage6Output::Clear(stage6_verified) = &stage6_output else {
            return Err("field-inline Stage 6 verifier did not return clear output".to_owned());
        };
        assert_eq!(stage6_verified, &stage6.verifier_output);
        assert_eq!(
            verifier_state.transcript.state(),
            prover_state.transcript.state()
        );

        let stage7_config = Stage7ProverConfig::new(
            log_t,
            HammingWeightClaimReductionDimensions::new(dimensions.ra_layout, committed_chunk_bits),
            None,
            None,
        );
        let stage7 = prove_stage7::<Fr, _, _, _, MockRoundCommitment>(
            Stage7ProverInput::new(
                &stage7_config,
                &verifier_state.checked,
                &stage4_verified,
                stage6_verified,
                &witness,
            ),
            &mut backend,
            &mut prover_state.transcript,
        )
        .map_err(|error| error.to_string())?;

        let proof = frontier_proof(
            proof.commitments,
            stage1_stage2_stage3_stage4_stage5_stage6_stage7_proofs(
                stage1.clone(),
                stage2.clone(),
                stage3.clone(),
                stage4.clone(),
                stage5.clone(),
                stage6.clone(),
                stage7.clone(),
            ),
            {
                let mut claims = jolt_verifier::compat::claims::empty_clear_opening_claims(
                    fixture.padded_trace_length,
                );
                claims.stage1 = stage1_claims;
                claims.stage2 = stage2.claims;
                claims.stage3 = stage3.claims;
                claims.stage4 = stage4.claims;
                claims.stage5 = stage5.claims;
                claims.stage6 = stage6.claims;
                claims.stage7 = stage7.claims.clone();
                claims
            },
            fixture.padded_trace_length,
            ram_k,
        );

        let mut verifier_state = verify_until_stage1::<
            MockPcs,
            MockVectorCommitment,
            Blake2bTranscript<Fr>,
            _,
        >(&verifier_preprocessing, &public_io, &proof, None, false)
        .map_err(|error| error.to_string())?;
        let stage1_output = jolt_verifier::stages::stage1::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let stage2_output = jolt_verifier::stages::stage2::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage2::inputs::deps(&stage1_output),
        )
        .map_err(|error| error.to_string())?;
        let stage3_output = jolt_verifier::stages::stage3::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage3::inputs::deps(&stage1_output, &stage2_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage4_output = jolt_verifier::stages::stage4::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage4::inputs::deps(&stage2_output, &stage3_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage5_output = jolt_verifier::stages::stage5::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage5::inputs::deps(&stage2_output, &stage4_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage6_output = jolt_verifier::stages::stage6::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage6::inputs::deps(
                &stage1_output,
                &stage2_output,
                &stage3_output,
                &stage4_output,
                &stage5_output,
            )
            .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage7_output = jolt_verifier::stages::stage7::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage7::inputs::deps(&stage4_output, &stage6_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let Stage7Output::Clear(stage7_verified) = &stage7_output else {
            return Err("field-inline Stage 7 verifier did not return clear output".to_owned());
        };
        assert_eq!(stage7_verified, &stage7.verifier_output);
        assert_eq!(
            verifier_state.transcript.state(),
            prover_state.transcript.state()
        );

        let stage8_config = Stage8ProverConfig::new(
            log_t,
            committed_chunk_bits,
            dimensions.ra_layout,
            jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
            None,
            None,
        );
        let field_rd_inc_constituent = field_rd_inc_stage8_constituent(
            &field_witness,
            committed_chunk_bits,
            log_t,
            jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
        )?;
        let mut commitments_ordered = vec![
            proof.commitments.ram_inc.clone(),
            proof.commitments.rd_inc.clone(),
            proof
                .commitments
                .field_inline
                .field_registers
                .rd_inc
                .clone(),
        ];
        commitments_ordered.extend(proof.commitments.ra.instruction.iter().cloned());
        commitments_ordered.extend(proof.commitments.ra.bytecode.iter().cloned());
        commitments_ordered.extend(proof.commitments.ra.ram.iter().cloned());
        let hints_ordered = vec![(); commitments_ordered.len()];

        let stage8 = prove_stage8::<Fr, MockPcs, _, _>(
            &stage8_config,
            stage6_verified,
            stage7_verified,
            &witness,
            &field_witness,
            &commitments_ordered,
            hints_ordered,
            &(),
            &mut prover_state.transcript,
        )
        .map_err(|error| error.to_string())?;

        assert!(matches!(
            stage8.structure.opening_ids[2],
            jolt_verifier::stages::stage8::outputs::Stage8OpeningId::FieldInline(_)
        ));
        let field_rd_inc_eval =
            field_rd_inc_constituent.evaluate(stage8.structure.pcs_opening_point.as_slice());
        if field_rd_inc_eval != stage8.structure.scaled_opening_values[2] {
            return Err(format!(
                "field-inline Stage 8 rd_inc opening mismatch: eval={field_rd_inc_eval:?}, expected={:?}",
                stage8.structure.scaled_opening_values[2]
            ));
        }
        let proof = stage8_verifier_proof(&proof, stage8.joint_opening_proof.clone())?;
        let stage8_verifier_preprocessing =
            JoltVerifierPreprocessing::<MockPcs, MockHidingVectorCommitment> {
                program: verifier_preprocessing.program.clone(),
                preprocessing_digest: verifier_preprocessing.preprocessing_digest,
                field_inline_bytecode: verifier_preprocessing.field_inline_bytecode.clone(),
                pcs_setup: verifier_preprocessing.pcs_setup,
                vc_setup: verifier_preprocessing.vc_setup,
            };
        let stage8_output = jolt_verifier::stages::stage8::verify(
            &verifier_state.checked,
            &stage8_verifier_preprocessing,
            &proof,
            None,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage8::inputs::Deps::Clear {
                stage6: stage6_verified,
                stage7: stage7_verified,
            },
        )
        .map_err(|error| error.to_string())?;
        let jolt_verifier::stages::stage8::Stage8Output::Clear(stage8_verified) = stage8_output
        else {
            return Err("field-inline Stage 8 verifier did not return clear output".to_owned());
        };
        assert_eq!(stage8_verified.opening_ids, stage8.structure.opening_ids);
        assert_eq!(
            stage8_verified.pcs_opening_point,
            stage8.structure.pcs_opening_point
        );
        assert_eq!(stage8_verified.joint_claim, stage8.structure.joint_claim);
        assert_eq!(stage8_verified.joint_commitment, stage8.joint_commitment);
        assert_eq!(
            verifier_state.transcript.state(),
            prover_state.transcript.state()
        );

        Ok(())
    }

    #[test]
    #[cfg(feature = "zk")]
    fn field_inline_eq_poly_guest_accepts_zk_committed_stage3_to_stage8_boundaries(
    ) -> Result<(), String> {
        with_zk_field_inline_stack(
            "field_inline_eq_poly_guest_accepts_zk_committed_stage3_to_stage8_boundaries",
            field_inline_eq_poly_guest_accepts_zk_committed_stage3_to_stage8_boundaries_inner,
        )
    }

    #[cfg(feature = "zk")]
    fn field_inline_eq_poly_guest_accepts_zk_committed_stage3_to_stage8_boundaries_inner(
    ) -> Result<(), String> {
        let inputs =
            postcard::to_stdvec(&7u32).map_err(|error| format!("serialize input: {error}"))?;
        let fixture = trace_sdk_guest(
            SdkGuestTraceRequest::new("field-inline-eq-poly-guest", inputs)
                .with_field_inline(true)
                .with_max_padded_trace_length(1 << 16),
        )
        .map_err(|error| error.to_string())?;

        let field_rows = fixture
            .trace
            .trace
            .rows()
            .iter()
            .filter(|row| row.field_inline.is_some())
            .count();
        assert_eq!(field_rows, 14);
        let metadata = fixture
            .preprocessing
            .bytecode
            .field_inline
            .as_ref()
            .ok_or_else(|| "field-inline preprocessing metadata is missing".to_owned())?;
        assert!(metadata.rows.iter().any(|row| row.active));

        let one_hot = JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        };
        let log_t = fixture.padded_trace_length.trailing_zeros() as usize;
        let committed_chunk_bits = one_hot.committed_chunk_bits();
        let bytecode_start = fixture
            .preprocessing
            .memory_layout
            .remapped_word_address(fixture.preprocessing.ram.min_bytecode_address)
            .map_err(|error| error.to_string())? as usize;
        let ram_k = (bytecode_start + fixture.preprocessing.ram.bytecode_words.len())
            .next_power_of_two()
            .max(1);
        let log_k = ram_k.trailing_zeros() as usize;
        let witness_config = JoltVmWitnessConfig::new(log_t, ram_k, one_hot);
        let dimensions = JoltFormulaDimensions::try_from(one_hot.dimensions(
            witness_config.log_t,
            RV64_LOOKUP_ADDRESS_BITS,
            fixture.preprocessing.bytecode.code_size,
            witness_config.ram_k,
        ))
        .map_err(|error| error.to_string())?;
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&fixture.program, &fixture.preprocessing, fixture.trace),
        );
        let field_witness = witness
            .field_inline_witness()
            .map_err(|error| error.to_string())?;

        let mut backend = CpuBackend::new(CpuBackendConfig {
            preserve_core_fast_path: true,
            commitment_chunk_size: 1 << 10,
        });
        let stage0 = prove::<Fr, _, _, MockPcs>(
            CommitmentStageInput::new(
                &witness,
                &(),
                CommitmentStageConfig::new(dimensions.ra_layout, false, false)
                    .with_final_opening_trace_embedding(
                        log_t,
                        committed_chunk_bits,
                        jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
                    ),
                jolt_verifier::JoltProtocolConfig::for_zk(true),
                &field_witness,
            ),
            &mut backend,
        )
        .map_err(|error| error.to_string())?;

        let verifier_bytecode = verifier_field_inline_bytecode_rows(&fixture.preprocessing)?;
        let transparent_verifier_preprocessing = JoltVerifierPreprocessing::<
            MockPcs,
            MockVectorCommitment,
        >::new(
            fixture.preprocessing.clone(), [7; 32], (), None
        )
        .with_field_inline_bytecode(verifier_bytecode.clone());
        let public_io = witness.trace.device.clone();
        let transparent_empty_proof = frontier_proof(
            stage0.commitments.clone(),
            empty_stage_proofs(),
            jolt_verifier::compat::claims::empty_clear_opening_claims(fixture.padded_trace_length),
            fixture.padded_trace_length,
            ram_k,
        );
        let mut clear_prover_state =
            verify_until_stage1::<MockPcs, MockVectorCommitment, Blake2bTranscript<Fr>, _>(
                &transparent_verifier_preprocessing,
                &public_io,
                &transparent_empty_proof,
                None,
                false,
            )
            .map_err(|error| error.to_string())?;
        let clear_stage1 = prove_stage1::<Fr, _, _, _, _, MockRoundCommitment>(
            Stage1ProverInput::new(Stage1ProverConfig::new(log_t), &witness, &field_witness),
            &mut backend,
            &mut clear_prover_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let clear_stage1_claims = stage1_claims_from_r1cs_inputs(
            clear_stage1.uniskip_output_claim,
            &clear_stage1.r1cs_input_claims,
            &clear_stage1.field_inline_r1cs_input_claims,
        )
        .map_err(|error| error.to_string())?;
        let clear_stage1_proof = frontier_proof(
            stage0.commitments.clone(),
            stage_proofs(clear_stage1.clone()),
            {
                let mut claims = jolt_verifier::compat::claims::empty_clear_opening_claims(
                    fixture.padded_trace_length,
                );
                claims.stage1 = clear_stage1_claims.clone();
                claims
            },
            fixture.padded_trace_length,
            ram_k,
        );
        let mut clear_verifier_state =
            verify_until_stage1::<MockPcs, MockVectorCommitment, Blake2bTranscript<Fr>, _>(
                &transparent_verifier_preprocessing,
                &public_io,
                &clear_stage1_proof,
                None,
                false,
            )
            .map_err(|error| error.to_string())?;
        let clear_stage1_output = jolt_verifier::stages::stage1::verify(
            &clear_verifier_state.checked,
            &transparent_verifier_preprocessing,
            &clear_stage1_proof,
            &mut clear_verifier_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let Stage1Output::Clear(clear_stage1_verified) = clear_stage1_output else {
            return Err(
                "field-inline clear Stage 1 verifier did not return clear output".to_owned(),
            );
        };
        let clear_stage2 = prove_stage2::<Fr, _, _, _, _, MockRoundCommitment>(
            Stage2ProverInput::new(
                Stage2BatchProverConfig::new(
                    log_t,
                    log_k,
                    jolt_claims::protocols::jolt::JoltReadWriteConfig {
                        ram_rw_phase1_num_rounds: 1,
                        ram_rw_phase2_num_rounds: 1,
                        registers_rw_phase1_num_rounds: 1,
                        registers_rw_phase2_num_rounds: 1,
                    },
                ),
                &clear_verifier_state.checked,
                &clear_stage1_verified,
                &witness,
                &field_witness,
            ),
            &mut backend,
            &mut clear_prover_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let clear_stage2_proof = frontier_proof(
            stage0.commitments.clone(),
            stage1_stage2_proofs(clear_stage1, clear_stage2.clone()),
            {
                let mut claims = jolt_verifier::compat::claims::empty_clear_opening_claims(
                    fixture.padded_trace_length,
                );
                claims.stage1 = clear_stage1_claims;
                claims.stage2 = clear_stage2.claims.clone();
                claims
            },
            fixture.padded_trace_length,
            ram_k,
        );
        let mut clear_verifier_state =
            verify_until_stage1::<MockPcs, MockVectorCommitment, Blake2bTranscript<Fr>, _>(
                &transparent_verifier_preprocessing,
                &public_io,
                &clear_stage2_proof,
                None,
                false,
            )
            .map_err(|error| error.to_string())?;
        let clear_stage1_output = jolt_verifier::stages::stage1::verify(
            &clear_verifier_state.checked,
            &transparent_verifier_preprocessing,
            &clear_stage2_proof,
            &mut clear_verifier_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let clear_stage2_output = jolt_verifier::stages::stage2::verify(
            &clear_verifier_state.checked,
            &transparent_verifier_preprocessing,
            &clear_stage2_proof,
            &mut clear_verifier_state.transcript,
            jolt_verifier::stages::stage2::inputs::deps(&clear_stage1_output),
        )
        .map_err(|error| error.to_string())?;
        let Stage1Output::Clear(clear_stage1_verified) = clear_stage1_output else {
            return Err(
                "field-inline clear Stage 1 verifier did not return clear output".to_owned(),
            );
        };
        let Stage2Output::Clear(clear_stage2_verified) = clear_stage2_output else {
            return Err(
                "field-inline clear Stage 2 verifier did not return clear output".to_owned(),
            );
        };
        let clear_stage3 = prove_stage3::<Fr, _, _, _, MockRoundCommitment>(
            Stage3ProverInput::new(
                Stage3ProverConfig::new(log_t),
                &clear_verifier_state.checked,
                &clear_stage1_verified,
                &clear_stage2_verified,
                &witness,
            ),
            &mut backend,
            &mut clear_prover_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let ram_val_check_init = stage4_ram_val_check_initial_evaluation(
            &fixture.preprocessing,
            &public_io,
            log_k,
            &clear_stage2_verified,
        )?;
        let clear_stage4 = prove_stage4::<Fr, _, _, _, _, MockRoundCommitment>(
            Stage4ProverInput::new(
                Stage4ProverConfig::new(
                    log_t,
                    log_k,
                    jolt_claims::protocols::jolt::JoltReadWriteConfig {
                        ram_rw_phase1_num_rounds: 1,
                        ram_rw_phase2_num_rounds: 1,
                        registers_rw_phase1_num_rounds: 1,
                        registers_rw_phase2_num_rounds: 1,
                    },
                ),
                &clear_verifier_state.checked,
                &clear_stage2_verified,
                &clear_stage3.verifier_output,
                ram_val_check_init.clone(),
                &witness,
                &field_witness,
            ),
            &mut backend,
            &mut clear_prover_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let entry_bytecode_index = fixture
            .preprocessing
            .bytecode
            .entry_bytecode_index()
            .ok_or_else(|| "entry address was not found in bytecode preprocessing".to_owned())?;
        let stage6_config = Stage6ProverConfig::new(
            log_t,
            log_k,
            committed_chunk_bits,
            dimensions.bytecode_read_raf,
            jolt_claims::protocols::jolt::formulas::booleanity::BooleanityDimensions::new(
                dimensions.ra_layout,
                log_t,
                committed_chunk_bits,
            ),
            dimensions.ram_ra_virtualization,
            dimensions.instruction_ra_virtualization,
            None,
            None,
        )
        .with_bytecode_context(
            fixture.preprocessing.bytecode.bytecode.clone(),
            entry_bytecode_index,
        );
        let stage7_config = Stage7ProverConfig::new(
            log_t,
            HammingWeightClaimReductionDimensions::new(dimensions.ra_layout, committed_chunk_bits),
            None,
            None,
        );
        let stage8_config = Stage8ProverConfig::new(
            log_t,
            committed_chunk_bits,
            dimensions.ra_layout,
            jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
            None,
            None,
        );

        let vc_capacity = fixture.padded_trace_length.max(MAX_BLINDFOLD_GENERATORS);
        let verifier_preprocessing =
            JoltVerifierPreprocessing::<MockPcs, MockVectorCommitment>::new(
                fixture.preprocessing.clone(),
                [7; 32],
                (),
                Some(vc_capacity),
            )
            .with_field_inline_bytecode(verifier_bytecode.clone());
        let mut zk_stages = zk_stage1_stage2_boundary_proofs(log_t, log_k);
        let proof = zk_frontier_proof(
            stage0.commitments.clone(),
            zk_stages.clone(),
            fixture.padded_trace_length,
            ram_k,
        );

        let mut verifier_state = verify_until_stage1::<
            MockPcs,
            MockVectorCommitment,
            Blake2bTranscript<Fr>,
            (),
        >(&verifier_preprocessing, &public_io, &proof, None, true)
        .map_err(|error| error.to_string())?;
        let stage1_output = jolt_verifier::stages::stage1::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
        )
        .map_err(|error| error.to_string())?;

        assert!(matches!(stage1_output, Stage1Output::Zk(_)));
        let stage2_output = jolt_verifier::stages::stage2::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage2::inputs::deps(&stage1_output),
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage2_output, Stage2Output::Zk(_)));

        let mut stage3_prover_transcript = verifier_state.transcript.clone();
        let stage3 = prove_stage3_committed_boundary::<Fr, _, _, _, MockVectorCommitment>(
            Stage3ProverInput::new(
                Stage3ProverConfig::new(log_t),
                &verifier_state.checked,
                &clear_stage1_verified,
                &clear_stage2_verified,
                &witness,
            ),
            &mut backend,
            &mut stage3_prover_transcript,
            &vc_capacity,
        )
        .map_err(|error| error.to_string())?;
        assert_eq!(stage3.output_claim_values.len(), 13);
        zk_stages.stage3_sumcheck_proof = stage3.stage3_sumcheck_proof.clone();
        let proof = zk_frontier_proof(
            stage0.commitments.clone(),
            zk_stages.clone(),
            fixture.padded_trace_length,
            ram_k,
        );

        let mut verifier_state = verify_until_stage1::<
            MockPcs,
            MockVectorCommitment,
            Blake2bTranscript<Fr>,
            (),
        >(&verifier_preprocessing, &public_io, &proof, None, true)
        .map_err(|error| error.to_string())?;
        let stage1_output = jolt_verifier::stages::stage1::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
        )
        .map_err(|error| error.to_string())?;

        assert!(matches!(stage1_output, Stage1Output::Zk(_)));
        let stage2_output = jolt_verifier::stages::stage2::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage2::inputs::deps(&stage1_output),
        )
        .map_err(|error| error.to_string())?;
        assert!(matches!(stage2_output, Stage2Output::Zk(_)));
        let stage3_output = jolt_verifier::stages::stage3::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage3::inputs::deps(&stage1_output, &stage2_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let Stage3Output::Zk(stage3_output) = stage3_output else {
            return Err("field-inline ZK Stage 3 verifier did not return ZK output".to_owned());
        };
        assert_eq!(stage3_output.public, stage3.public);
        assert_eq!(
            stage3_output.batch_output_claims.shape.output_claim_count,
            13
        );
        assert_eq!(stage3_output.batch_output_claims.shape.row_count, 1);
        assert_eq!(
            verifier_state.transcript.state(),
            stage3_prover_transcript.state()
        );

        let mut stage4_prover_transcript = verifier_state.transcript.clone();
        let stage4 = prove_stage4_committed_boundary::<Fr, _, _, _, _, MockVectorCommitment>(
            Stage4ProverInput::new(
                Stage4ProverConfig::new(
                    log_t,
                    log_k,
                    jolt_claims::protocols::jolt::JoltReadWriteConfig {
                        ram_rw_phase1_num_rounds: 1,
                        ram_rw_phase2_num_rounds: 1,
                        registers_rw_phase1_num_rounds: 1,
                        registers_rw_phase2_num_rounds: 1,
                    },
                ),
                &verifier_state.checked,
                &clear_stage2_verified,
                &clear_stage3.verifier_output,
                ram_val_check_init.clone(),
                &witness,
                &field_witness,
            ),
            &mut backend,
            &mut stage4_prover_transcript,
            &vc_capacity,
        )
        .map_err(|error| error.to_string())?;
        assert_eq!(stage4.output_claim_values.len(), 12);
        zk_stages.stage4_sumcheck_proof = stage4.stage4_sumcheck_proof.clone();
        let proof = zk_frontier_proof(
            stage0.commitments.clone(),
            zk_stages.clone(),
            fixture.padded_trace_length,
            ram_k,
        );
        let mut verifier_state = verify_until_stage1::<
            MockPcs,
            MockVectorCommitment,
            Blake2bTranscript<Fr>,
            (),
        >(&verifier_preprocessing, &public_io, &proof, None, true)
        .map_err(|error| error.to_string())?;
        let stage1_output = jolt_verifier::stages::stage1::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let stage2_output = jolt_verifier::stages::stage2::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage2::inputs::deps(&stage1_output),
        )
        .map_err(|error| error.to_string())?;
        let stage3_output = jolt_verifier::stages::stage3::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage3::inputs::deps(&stage1_output, &stage2_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage4_output = jolt_verifier::stages::stage4::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage4::inputs::deps(&stage2_output, &stage3_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let Stage4Output::Zk(stage4_output) = stage4_output else {
            return Err("field-inline ZK Stage 4 verifier did not return ZK output".to_owned());
        };
        assert_eq!(stage4_output.public, stage4.public);
        assert_eq!(
            stage4_output.batch_output_claims.shape.output_claim_count,
            12
        );
        assert_eq!(stage4_output.batch_output_claims.shape.row_count, 1);
        assert_eq!(
            verifier_state.transcript.state(),
            stage4_prover_transcript.state()
        );

        let mut stage5_prover_transcript = verifier_state.transcript.clone();
        let stage5 = prove_stage5_committed_boundary::<Fr, _, _, _, _, MockVectorCommitment>(
            Stage5ProverInput::new(
                Stage5ProverConfig::new(log_t, log_k, dimensions.instruction_read_raf),
                &verifier_state.checked,
                &clear_stage2_verified,
                &clear_stage4.verifier_output,
                &witness,
                &field_witness,
            ),
            &mut backend,
            &mut stage5_prover_transcript,
            &vc_capacity,
        )
        .map_err(|error| error.to_string())?;
        let stage5_output_claim_count = stage5.output_claim_values.len();
        assert!(stage5_output_claim_count > 6);
        zk_stages.stage5_sumcheck_proof = stage5.stage5_sumcheck_proof.clone();
        let proof = zk_frontier_proof(
            stage0.commitments.clone(),
            zk_stages.clone(),
            fixture.padded_trace_length,
            ram_k,
        );
        let mut verifier_state = verify_until_stage1::<
            MockPcs,
            MockVectorCommitment,
            Blake2bTranscript<Fr>,
            (),
        >(&verifier_preprocessing, &public_io, &proof, None, true)
        .map_err(|error| error.to_string())?;
        let stage1_output = jolt_verifier::stages::stage1::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let stage2_output = jolt_verifier::stages::stage2::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage2::inputs::deps(&stage1_output),
        )
        .map_err(|error| error.to_string())?;
        let stage3_output = jolt_verifier::stages::stage3::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage3::inputs::deps(&stage1_output, &stage2_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage4_output = jolt_verifier::stages::stage4::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage4::inputs::deps(&stage2_output, &stage3_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage5_output = jolt_verifier::stages::stage5::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage5::inputs::deps(&stage2_output, &stage4_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let Stage5Output::Zk(stage5_output) = stage5_output else {
            return Err("field-inline ZK Stage 5 verifier did not return ZK output".to_owned());
        };
        assert_eq!(stage5_output.public, stage5.public);
        assert_eq!(
            stage5_output.batch_output_claims.shape.output_claim_count,
            stage5_output_claim_count
        );
        assert_eq!(stage5_output.batch_output_claims.shape.row_count, 1);
        assert_eq!(
            verifier_state.transcript.state(),
            stage5_prover_transcript.state()
        );

        let mut stage6_prover_transcript = verifier_state.transcript.clone();
        let stage6 = prove_stage6_committed_boundary::<Fr, _, _, _, _, MockVectorCommitment>(
            Stage6ProverInput::new(
                &stage6_config,
                &verifier_state.checked,
                &clear_stage1_verified,
                &clear_stage2_verified,
                &clear_stage3.verifier_output,
                &clear_stage4.verifier_output,
                &stage5.verifier_output,
                &witness,
                &field_witness,
            ),
            &mut backend,
            &mut stage6_prover_transcript,
            &vc_capacity,
        )
        .map_err(|error| error.to_string())?;
        let stage6_output_claim_count = stage6.output_claim_values.len();
        assert!(stage6_output_claim_count > stage5_output_claim_count);
        zk_stages.stage6_sumcheck_proof = stage6.stage6_sumcheck_proof.clone();
        let proof = zk_frontier_proof(
            stage0.commitments.clone(),
            zk_stages.clone(),
            fixture.padded_trace_length,
            ram_k,
        );
        let mut verifier_state = verify_until_stage1::<
            MockPcs,
            MockVectorCommitment,
            Blake2bTranscript<Fr>,
            (),
        >(&verifier_preprocessing, &public_io, &proof, None, true)
        .map_err(|error| error.to_string())?;
        let stage1_output = jolt_verifier::stages::stage1::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
        )
        .map_err(|error| error.to_string())?;
        let stage2_output = jolt_verifier::stages::stage2::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage2::inputs::deps(&stage1_output),
        )
        .map_err(|error| error.to_string())?;
        let stage3_output = jolt_verifier::stages::stage3::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage3::inputs::deps(&stage1_output, &stage2_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage4_output = jolt_verifier::stages::stage4::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage4::inputs::deps(&stage2_output, &stage3_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage5_output = jolt_verifier::stages::stage5::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage5::inputs::deps(&stage2_output, &stage4_output)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let stage6_output = jolt_verifier::stages::stage6::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage6::inputs::deps(
                &stage1_output,
                &stage2_output,
                &stage3_output,
                &stage4_output,
                &stage5_output,
            )
            .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let Stage6Output::Zk(stage6_output) = stage6_output else {
            return Err("field-inline ZK Stage 6 verifier did not return ZK output".to_owned());
        };
        assert_eq!(stage6_output.public, stage6.public);
        assert_eq!(
            stage6_output.batch_output_claims.shape.output_claim_count,
            stage6_output_claim_count
        );
        assert_eq!(stage6_output.batch_output_claims.shape.row_count, 1);
        assert_eq!(
            verifier_state.transcript.state(),
            stage6_prover_transcript.state()
        );

        let mut stage7_prover_transcript = verifier_state.transcript.clone();
        let stage7 = prove_stage7_committed_boundary::<Fr, _, _, _, MockVectorCommitment>(
            Stage7ProverInput::new(
                &stage7_config,
                &verifier_state.checked,
                &clear_stage4.verifier_output,
                &stage6.verifier_output,
                &witness,
            ),
            &mut backend,
            &mut stage7_prover_transcript,
            &vc_capacity,
        )
        .map_err(|error| error.to_string())?;
        let stage7_output_claim_count = stage7.output_claim_values.len();
        assert!(stage7_output_claim_count > 0);
        zk_stages.stage7_sumcheck_proof = stage7.stage7_sumcheck_proof.clone();
        let proof = zk_frontier_proof(
            stage0.commitments.clone(),
            zk_stages.clone(),
            fixture.padded_trace_length,
            ram_k,
        );
        let stage6_output_enum = Stage6Output::Zk(stage6_output.clone());
        let stage7_output = jolt_verifier::stages::stage7::verify(
            &verifier_state.checked,
            &verifier_preprocessing,
            &proof,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage7::inputs::deps(&stage4_output, &stage6_output_enum)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let Stage7Output::Zk(stage7_output) = stage7_output else {
            return Err("field-inline ZK Stage 7 verifier did not return ZK output".to_owned());
        };
        assert_eq!(stage7_output.public, stage7.public);
        assert_eq!(
            stage7_output.batch_output_claims.shape.output_claim_count,
            stage7_output_claim_count
        );
        assert_eq!(stage7_output.batch_output_claims.shape.row_count, 1);
        assert_eq!(
            verifier_state.transcript.state(),
            stage7_prover_transcript.state()
        );

        let mut commitments_ordered = vec![
            proof.commitments.ram_inc.clone(),
            proof.commitments.rd_inc.clone(),
            proof
                .commitments
                .field_inline
                .field_registers
                .rd_inc
                .clone(),
        ];
        commitments_ordered.extend(proof.commitments.ra.instruction.iter().cloned());
        commitments_ordered.extend(proof.commitments.ra.bytecode.iter().cloned());
        commitments_ordered.extend(proof.commitments.ra.ram.iter().cloned());
        let hints_ordered = vec![(); commitments_ordered.len()];
        let mut stage8_prover_transcript = verifier_state.transcript.clone();
        let stage8 = prove_stage8_zk::<Fr, Stage8RoundHidingMockPcs, _, _>(
            &stage8_config,
            &stage6.verifier_output,
            &stage7.verifier_output,
            &witness,
            &field_witness,
            &commitments_ordered,
            hints_ordered,
            &(),
            &mut stage8_prover_transcript,
        )
        .map_err(|error| error.to_string())?;
        let stage8_verifier_preprocessing =
            JoltVerifierPreprocessing::<Stage8RoundHidingMockPcs, MockVectorCommitment>::new(
                fixture.preprocessing.clone(),
                [7; 32],
                (),
                Some(vc_capacity),
            )
            .with_field_inline_bytecode(verifier_bytecode);
        let stage8_proof = zk_stage8_round_hiding_proof(
            proof.commitments.clone(),
            zk_stages,
            stage8.joint_opening_proof.clone(),
            fixture.padded_trace_length,
            ram_k,
        );
        let stage8_output = jolt_verifier::stages::stage8::verify(
            &verifier_state.checked,
            &stage8_verifier_preprocessing,
            &stage8_proof,
            None,
            &mut verifier_state.transcript,
            jolt_verifier::stages::stage8::inputs::Deps::Zk {
                stage6: &stage6_output,
                stage7: &stage7_output,
            },
        )
        .map_err(|error| error.to_string())?;
        let jolt_verifier::stages::stage8::Stage8Output::Zk(stage8_output) = stage8_output else {
            return Err("field-inline ZK Stage 8 verifier did not return ZK output".to_owned());
        };
        assert_eq!(stage8_output.opening_ids, stage8.structure.opening_ids);
        assert_eq!(
            stage8_output.constraint_coefficients,
            stage8.structure.constraint_coefficients
        );
        assert_eq!(
            stage8_output.pcs_opening_point,
            stage8.structure.pcs_opening_point
        );
        assert_eq!(stage8_output.joint_commitment, stage8.joint_commitment);
        assert_eq!(
            stage8_output.hiding_evaluation_commitment,
            stage8.hiding_evaluation_commitment
        );
        assert_eq!(
            verifier_state.transcript.state(),
            stage8_prover_transcript.state()
        );

        Ok(())
    }

    #[cfg(feature = "zk")]
    fn with_zk_field_inline_stack(
        name: &'static str,
        test: fn() -> Result<(), String>,
    ) -> Result<(), String> {
        let pool = rayon::ThreadPoolBuilder::new()
            .thread_name(move |index| format!("{name}-rayon-{index}"))
            .stack_size(128 * 1024 * 1024)
            .build()
            .map_err(|error| format!("build {name} Rayon pool: {error}"))?;
        let handle = std::thread::Builder::new()
            .name(name.to_owned())
            .stack_size(128 * 1024 * 1024)
            .spawn(move || pool.install(test))
            .map_err(|error| format!("spawn {name}: {error}"))?;
        match handle.join() {
            Ok(result) => result,
            Err(payload) => {
                if let Some(message) = payload.downcast_ref::<&str>() {
                    Err(format!("{name} panicked: {message}"))
                } else if let Some(message) = payload.downcast_ref::<String>() {
                    Err(format!("{name} panicked: {message}"))
                } else {
                    Err(format!("{name} panicked with non-string payload"))
                }
            }
        }
    }

    fn frontier_proof(
        commitments: jolt_verifier::proof::JoltCommitments<<MockPcs as Commitment>::Output>,
        stages: JoltStageProofs<Fr, MockVectorCommitment>,
        claims: jolt_verifier::proof::ClearProofClaims<Fr>,
        trace_length: usize,
        ram_k: usize,
    ) -> JoltProof<MockPcs, MockVectorCommitment> {
        JoltProof::<MockPcs, MockVectorCommitment>::new(
            commitments,
            stages,
            mock_joint_opening_proof(),
            None,
            JoltProofClaims::Clear(claims),
            trace_length,
            ram_k,
            jolt_claims::protocols::jolt::JoltReadWriteConfig {
                ram_rw_phase1_num_rounds: 1,
                ram_rw_phase2_num_rounds: 1,
                registers_rw_phase1_num_rounds: 1,
                registers_rw_phase2_num_rounds: 1,
            },
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
            jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
        )
    }

    fn stage8_verifier_proof(
        proof: &JoltProof<MockPcs, MockVectorCommitment>,
        joint_opening_proof: <MockPcs as CommitmentScheme>::Proof,
    ) -> Result<JoltProof<MockPcs, MockHidingVectorCommitment>, String> {
        let JoltProofClaims::Clear(claims) = &proof.claims else {
            return Err("Stage 8 verifier replay expects a clear proof shell".to_owned());
        };
        Ok(JoltProof::<MockPcs, MockHidingVectorCommitment>::new(
            proof.commitments.clone(),
            empty_hiding_stage_proofs(),
            joint_opening_proof,
            proof.untrusted_advice_commitment.clone(),
            JoltProofClaims::Clear(claims.clone()),
            proof.trace_length,
            proof.ram_K,
            proof.rw_config,
            proof.one_hot_config,
            proof.trace_polynomial_order,
        ))
    }

    fn empty_hiding_stage_proofs() -> JoltStageProofs<Fr, MockHidingVectorCommitment> {
        let empty = SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof::default()));
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: empty.clone(),
            stage1_sumcheck_proof: empty.clone(),
            stage2_uni_skip_first_round_proof: empty.clone(),
            stage2_sumcheck_proof: empty.clone(),
            stage3_sumcheck_proof: empty.clone(),
            stage4_sumcheck_proof: empty.clone(),
            stage5_sumcheck_proof: empty.clone(),
            stage6_sumcheck_proof: empty.clone(),
            stage7_sumcheck_proof: empty,
        }
    }

    #[cfg(feature = "zk")]
    fn zk_frontier_proof(
        commitments: jolt_verifier::proof::JoltCommitments<<MockPcs as Commitment>::Output>,
        stages: JoltStageProofs<Fr, MockVectorCommitment>,
        trace_length: usize,
        ram_k: usize,
    ) -> JoltProof<MockPcs, MockVectorCommitment, ()> {
        JoltProof::<MockPcs, MockVectorCommitment, ()>::new(
            commitments,
            stages,
            mock_joint_opening_proof(),
            None,
            JoltProofClaims::Zk {
                blindfold_proof: (),
            },
            trace_length,
            ram_k,
            jolt_claims::protocols::jolt::JoltReadWriteConfig {
                ram_rw_phase1_num_rounds: 1,
                ram_rw_phase2_num_rounds: 1,
                registers_rw_phase1_num_rounds: 1,
                registers_rw_phase2_num_rounds: 1,
            },
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
            jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
        )
    }

    #[cfg(feature = "zk")]
    fn zk_stage8_round_hiding_proof(
        commitments: jolt_verifier::proof::JoltCommitments<
            <Stage8RoundHidingMockPcs as Commitment>::Output,
        >,
        stages: JoltStageProofs<Fr, MockVectorCommitment>,
        joint_opening_proof: <Stage8RoundHidingMockPcs as CommitmentScheme>::Proof,
        trace_length: usize,
        ram_k: usize,
    ) -> JoltProof<Stage8RoundHidingMockPcs, MockVectorCommitment, ()> {
        JoltProof::<Stage8RoundHidingMockPcs, MockVectorCommitment, ()>::new(
            commitments,
            stages,
            joint_opening_proof,
            None,
            JoltProofClaims::Zk {
                blindfold_proof: (),
            },
            trace_length,
            ram_k,
            jolt_claims::protocols::jolt::JoltReadWriteConfig {
                ram_rw_phase1_num_rounds: 1,
                ram_rw_phase2_num_rounds: 1,
                registers_rw_phase1_num_rounds: 1,
                registers_rw_phase2_num_rounds: 1,
            },
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
            jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
        )
    }

    fn empty_stage_proofs() -> JoltStageProofs<Fr, MockVectorCommitment> {
        let empty = empty_sumcheck_proof();
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: empty.clone(),
            stage1_sumcheck_proof: empty.clone(),
            stage2_uni_skip_first_round_proof: empty.clone(),
            stage2_sumcheck_proof: empty.clone(),
            stage3_sumcheck_proof: empty.clone(),
            stage4_sumcheck_proof: empty.clone(),
            stage5_sumcheck_proof: empty.clone(),
            stage6_sumcheck_proof: empty.clone(),
            stage7_sumcheck_proof: empty,
        }
    }

    #[cfg(feature = "zk")]
    fn zk_stage1_stage2_boundary_proofs(
        log_t: usize,
        log_k: usize,
    ) -> JoltStageProofs<Fr, MockVectorCommitment> {
        let empty = empty_committed_sumcheck_proof();
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: committed_sumcheck_proof(
                1,
                SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
                1,
            ),
            stage1_sumcheck_proof: committed_sumcheck_proof(
                log_t + 1,
                SPARTAN_OUTER_REMAINDER_DEGREE,
                1,
            ),
            stage2_uni_skip_first_round_proof: committed_sumcheck_proof(
                1,
                SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
                1,
            ),
            stage2_sumcheck_proof: committed_sumcheck_proof(log_t + log_k, 3, 1),
            stage3_sumcheck_proof: committed_sumcheck_proof(log_t, 3, 1),
            stage4_sumcheck_proof: empty.clone(),
            stage5_sumcheck_proof: empty.clone(),
            stage6_sumcheck_proof: empty.clone(),
            stage7_sumcheck_proof: empty,
        }
    }

    fn stage_proofs(
        stage1: Stage1SumcheckOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
    ) -> JoltStageProofs<Fr, MockVectorCommitment> {
        let empty = empty_sumcheck_proof();
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: stage1.uniskip_proof,
            stage1_sumcheck_proof: stage1.remainder_proof,
            stage2_uni_skip_first_round_proof: empty.clone(),
            stage2_sumcheck_proof: empty.clone(),
            stage3_sumcheck_proof: empty.clone(),
            stage4_sumcheck_proof: empty.clone(),
            stage5_sumcheck_proof: empty.clone(),
            stage6_sumcheck_proof: empty.clone(),
            stage7_sumcheck_proof: empty,
        }
    }

    fn stage1_stage2_proofs(
        stage1: Stage1SumcheckOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage2: Stage2ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
    ) -> JoltStageProofs<Fr, MockVectorCommitment> {
        let empty = empty_sumcheck_proof();
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: stage1.uniskip_proof,
            stage1_sumcheck_proof: stage1.remainder_proof,
            stage2_uni_skip_first_round_proof: stage2.product_uniskip_proof,
            stage2_sumcheck_proof: stage2.regular_batch_proof,
            stage3_sumcheck_proof: empty.clone(),
            stage4_sumcheck_proof: empty.clone(),
            stage5_sumcheck_proof: empty.clone(),
            stage6_sumcheck_proof: empty.clone(),
            stage7_sumcheck_proof: empty,
        }
    }

    fn stage1_stage2_stage3_proofs(
        stage1: Stage1SumcheckOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage2: Stage2ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage3: Stage3ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
    ) -> JoltStageProofs<Fr, MockVectorCommitment> {
        let empty = empty_sumcheck_proof();
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: stage1.uniskip_proof,
            stage1_sumcheck_proof: stage1.remainder_proof,
            stage2_uni_skip_first_round_proof: stage2.product_uniskip_proof,
            stage2_sumcheck_proof: stage2.regular_batch_proof,
            stage3_sumcheck_proof: stage3.stage3_sumcheck_proof,
            stage4_sumcheck_proof: empty.clone(),
            stage5_sumcheck_proof: empty.clone(),
            stage6_sumcheck_proof: empty.clone(),
            stage7_sumcheck_proof: empty,
        }
    }

    fn stage1_stage2_stage3_stage4_proofs(
        stage1: Stage1SumcheckOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage2: Stage2ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage3: Stage3ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage4: Stage4ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
    ) -> JoltStageProofs<Fr, MockVectorCommitment> {
        let empty = empty_sumcheck_proof();
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: stage1.uniskip_proof,
            stage1_sumcheck_proof: stage1.remainder_proof,
            stage2_uni_skip_first_round_proof: stage2.product_uniskip_proof,
            stage2_sumcheck_proof: stage2.regular_batch_proof,
            stage3_sumcheck_proof: stage3.stage3_sumcheck_proof,
            stage4_sumcheck_proof: stage4.stage4_sumcheck_proof,
            stage5_sumcheck_proof: empty.clone(),
            stage6_sumcheck_proof: empty.clone(),
            stage7_sumcheck_proof: empty,
        }
    }

    fn stage1_stage2_stage3_stage4_stage5_proofs(
        stage1: Stage1SumcheckOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage2: Stage2ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage3: Stage3ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage4: Stage4ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage5: Stage5ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
    ) -> JoltStageProofs<Fr, MockVectorCommitment> {
        let empty = empty_sumcheck_proof();
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: stage1.uniskip_proof,
            stage1_sumcheck_proof: stage1.remainder_proof,
            stage2_uni_skip_first_round_proof: stage2.product_uniskip_proof,
            stage2_sumcheck_proof: stage2.regular_batch_proof,
            stage3_sumcheck_proof: stage3.stage3_sumcheck_proof,
            stage4_sumcheck_proof: stage4.stage4_sumcheck_proof,
            stage5_sumcheck_proof: stage5.stage5_sumcheck_proof,
            stage6_sumcheck_proof: empty.clone(),
            stage7_sumcheck_proof: empty,
        }
    }

    fn stage1_stage2_stage3_stage4_stage5_stage6_proofs(
        stage1: Stage1SumcheckOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage2: Stage2ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage3: Stage3ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage4: Stage4ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage5: Stage5ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage6: Stage6ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
    ) -> JoltStageProofs<Fr, MockVectorCommitment> {
        let empty = empty_sumcheck_proof();
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: stage1.uniskip_proof,
            stage1_sumcheck_proof: stage1.remainder_proof,
            stage2_uni_skip_first_round_proof: stage2.product_uniskip_proof,
            stage2_sumcheck_proof: stage2.regular_batch_proof,
            stage3_sumcheck_proof: stage3.stage3_sumcheck_proof,
            stage4_sumcheck_proof: stage4.stage4_sumcheck_proof,
            stage5_sumcheck_proof: stage5.stage5_sumcheck_proof,
            stage6_sumcheck_proof: stage6.stage6_sumcheck_proof,
            stage7_sumcheck_proof: empty,
        }
    }

    fn stage1_stage2_stage3_stage4_stage5_stage6_stage7_proofs(
        stage1: Stage1SumcheckOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage2: Stage2ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage3: Stage3ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage4: Stage4ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage5: Stage5ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage6: Stage6ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
        stage7: Stage7ProverOutput<Fr, SumcheckProof<Fr, MockRoundCommitment>>,
    ) -> JoltStageProofs<Fr, MockVectorCommitment> {
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: stage1.uniskip_proof,
            stage1_sumcheck_proof: stage1.remainder_proof,
            stage2_uni_skip_first_round_proof: stage2.product_uniskip_proof,
            stage2_sumcheck_proof: stage2.regular_batch_proof,
            stage3_sumcheck_proof: stage3.stage3_sumcheck_proof,
            stage4_sumcheck_proof: stage4.stage4_sumcheck_proof,
            stage5_sumcheck_proof: stage5.stage5_sumcheck_proof,
            stage6_sumcheck_proof: stage6.stage6_sumcheck_proof,
            stage7_sumcheck_proof: stage7.stage7_sumcheck_proof,
        }
    }

    fn stage4_ram_val_check_initial_evaluation(
        preprocessing: &JoltProgramPreprocessing,
        public_io: &common::jolt_device::JoltDevice,
        log_k: usize,
        stage2: &jolt_verifier::stages::stage2::Stage2ClearOutput<Fr>,
    ) -> Result<Stage4RamValCheckInitialEvaluation<Fr>, String> {
        let (r_address, _) = stage2.batch.ram_read_write.opening_point.split_at(log_k);
        let public_initial_ram = PublicInitialRam::new(&preprocessing.ram, public_io)
            .map_err(|error| error.to_string())?;
        let public_eval = sparse_segments_mle_msb(
            public_initial_ram
                .segments
                .iter()
                .map(|segment| (segment.start_index, segment.words.as_slice())),
            r_address,
        );
        Ok(Stage4RamValCheckInitialEvaluation {
            public_eval,
            advice_contributions: Vec::new(),
            full_eval: public_eval,
        })
    }

    fn empty_sumcheck_proof() -> SumcheckProof<Fr, MockRoundCommitment> {
        SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof::default()))
    }

    #[cfg(feature = "zk")]
    fn empty_committed_sumcheck_proof() -> SumcheckProof<Fr, MockRoundCommitment> {
        committed_sumcheck_proof(0, 0, 0)
    }

    #[cfg(feature = "zk")]
    fn committed_sumcheck_proof(
        rounds: usize,
        degree: usize,
        output_claim_commitments: usize,
    ) -> SumcheckProof<Fr, MockRoundCommitment> {
        SumcheckProof::Committed(CommittedSumcheckProof {
            rounds: (0..rounds)
                .map(|round| CommittedRound {
                    commitment: MockRoundCommitment((round + 1) as u64),
                    degree,
                })
                .collect(),
            output_claims: CommittedOutputClaims {
                commitments: (0..output_claim_commitments)
                    .map(|row| MockRoundCommitment((rounds + row + 1) as u64))
                    .collect(),
            },
        })
    }

    fn mock_joint_opening_proof() -> <MockPcs as CommitmentScheme>::Proof {
        let poly = Polynomial::new(vec![Fr::from_u64(0)]);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"sdk-field-inline-opening");
        MockPcs::open(&poly, &[], Fr::from_u64(0), &(), None, &mut transcript)
    }

    fn field_rd_inc_stage8_constituent(
        field_witness: &TraceBackedFieldInlineWitness<
            '_,
            '_,
            impl Clone + jolt_program::execution::TraceSource,
        >,
        committed_chunk_bits: usize,
        log_t: usize,
        trace_polynomial_order: jolt_verifier::proof::TracePolynomialOrder,
    ) -> Result<Polynomial<Fr>, String> {
        let rows = 1usize << log_t;
        let addresses = 1usize << committed_chunk_bits;
        let full = 1usize << (committed_chunk_bits + log_t);
        let mut evals = vec![Fr::from_u64(0); full];
        let mut stream = <TraceBackedFieldInlineWitness<'_, '_, _> as WitnessProvider<
            Fr,
            FieldInlineNamespace,
        >>::committed_stream(
            field_witness,
            FieldInlineCommittedPolynomial::FieldRdInc,
            1024,
        )
        .map_err(|error| error.to_string())?;
        let mut index = 0usize;
        while let Some(chunk) = stream.next_chunk().map_err(|error| error.to_string())? {
            let jolt_witness::PolynomialChunk::Dense(values) = chunk else {
                return Err("expected dense field-inline rd_inc stream".to_owned());
            };
            for value in values {
                if index >= rows {
                    return Err("field-inline rd_inc stream exceeded trace length".to_owned());
                }
                let flat = trace_polynomial_order.address_cycle_to_index(0, index, addresses, rows);
                evals[flat] = value;
                index += 1;
            }
        }
        if index != rows {
            return Err(format!(
                "field-inline rd_inc stream produced {index} rows, expected {rows}"
            ));
        }
        Ok(Polynomial::from(evals))
    }

    fn verifier_field_inline_bytecode_rows(
        preprocessing: &JoltProgramPreprocessing,
    ) -> Result<
        Vec<jolt_claims::protocols::field_inline::formulas::bytecode::FieldInlineBytecodeRow>,
        String,
    > {
        preprocessing
            .bytecode
            .bytecode
            .iter()
            .map(verifier_field_inline_bytecode_row)
            .collect()
    }

    fn verifier_field_inline_bytecode_row(
        instruction: &jolt_riscv::JoltInstructionRow,
    ) -> Result<
        jolt_claims::protocols::field_inline::formulas::bytecode::FieldInlineBytecodeRow,
        String,
    > {
        use jolt_claims::protocols::field_inline::formulas::bytecode::{
            FieldInlineBytecodeFlags, FieldInlineBytecodeOperands, FieldInlineBytecodeRow,
        };

        let operands = FieldInlineBytecodeOperands {
            rd: instruction.operands.rd,
            rs1: instruction.operands.rs1,
            rs2: instruction.operands.rs2,
        };
        let mut row = FieldInlineBytecodeRow::default();
        match instruction.instruction_kind {
            JoltInstructionKind::NoOp => {}
            JoltInstructionKind::FIELD_ADD => {
                row.operands = operands;
                row.flags = FieldInlineBytecodeFlags {
                    add: true,
                    ..Default::default()
                };
            }
            JoltInstructionKind::FIELD_SUB => {
                row.operands = operands;
                row.flags = FieldInlineBytecodeFlags {
                    sub: true,
                    ..Default::default()
                };
            }
            JoltInstructionKind::FIELD_MUL => {
                row.operands = operands;
                row.flags = FieldInlineBytecodeFlags {
                    mul: true,
                    ..Default::default()
                };
            }
            JoltInstructionKind::FIELD_INV => {
                row.operands = FieldInlineBytecodeOperands {
                    rd: instruction.operands.rd,
                    rs1: instruction.operands.rs1,
                    rs2: None,
                };
                row.flags = FieldInlineBytecodeFlags {
                    inv: true,
                    ..Default::default()
                };
            }
            JoltInstructionKind::FIELD_ASSERT_EQ => {
                row.operands = FieldInlineBytecodeOperands {
                    rd: None,
                    rs1: instruction.operands.rs1,
                    rs2: instruction.operands.rs2,
                };
                row.flags = FieldInlineBytecodeFlags {
                    assert_eq: true,
                    ..Default::default()
                };
            }
            JoltInstructionKind::FIELD_LOAD_FROM_X => {
                row.operands = FieldInlineBytecodeOperands {
                    rd: instruction.operands.rd,
                    rs1: None,
                    rs2: None,
                };
                row.flags = FieldInlineBytecodeFlags {
                    load_from_x: true,
                    ..Default::default()
                };
            }
            JoltInstructionKind::FIELD_STORE_TO_X => {
                row.operands = FieldInlineBytecodeOperands {
                    rd: None,
                    rs1: instruction.operands.rs1,
                    rs2: None,
                };
                row.flags = FieldInlineBytecodeFlags {
                    store_to_x: true,
                    ..Default::default()
                };
            }
            JoltInstructionKind::FIELD_LOAD_IMM => {
                row.operands = FieldInlineBytecodeOperands {
                    rd: instruction.operands.rd,
                    rs1: None,
                    rs2: None,
                };
                row.flags = FieldInlineBytecodeFlags {
                    load_imm: true,
                    ..Default::default()
                };
            }
            _ => {}
        }
        Ok(row)
    }
}
