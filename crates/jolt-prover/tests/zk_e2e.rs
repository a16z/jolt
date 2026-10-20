//! ZK end-to-end coverage for the modular prover and verifier: the
//! mode-specific checks (BlindFold tampering, the reference backend, the
//! unaligned SHA3 inline expansion, committed programs). Plain acceptance
//! across guests is `e2e_matrix.rs`.

#[cfg(all(
    feature = "prover-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
mod support;

#[cfg(all(
    feature = "prover-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests should fail loudly"
)]
mod zk {
    extern crate jolt_inlines_keccak256;

    use common::jolt_device::JoltDevice;
    use jolt_algebra::{Fr, Ring, Zero};
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::{DoryCommitment, DoryScheme};
    use jolt_prover::dory::DoryProverPreprocessing;
    use jolt_prover::zk_secrets::{self, ZkSecrets};
    use jolt_prover::{JoltBackend, JoltSharedPreprocessing, ProverConfig};
    use jolt_riscv::JoltTraceRow;
    use jolt_transcript::VerifierTranscript;
    use jolt_verifier::proof::JoltProof;
    use jolt_verifier::JoltProtocolConfig;
    use jolt_verifier::JoltSponge;
    use jolt_verifier::VerifierError;
    use jolt_verifier::{
        jolt_protocol_id, verify_stages, MaskedStages, VerifiedStages, JOLT_SESSION,
    };
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use jolt_zk::masking::{ResidualRelation, ResidualVar};

    use crate::support::{self, with_zk_stack, GuestCase, PreparedGuest};

    type Proof = JoltProof;

    struct ProvedGuest {
        preprocessing: DoryProverPreprocessing,
        public_io: JoltDevice,
        proof: Proof,
        trusted_advice_commitment: Option<DoryCommitment>,
    }

    fn muldiv_case() -> GuestCase {
        GuestCase {
            inputs: postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs"),
            ..GuestCase::new("muldiv-guest")
        }
    }

    fn derive_config(run: &PreparedGuest) -> ProverConfig {
        ProverConfig::derive_from_dimensions::<Fr>(
            run.trace.dimensions,
            &run.preprocessing.memory_layout,
            run.preprocessing.ram.min_bytecode_address,
            run.preprocessing.ram.bytecode_words.len(),
            run.preprocessing.max_padded_trace_length,
        )
        .expect("derive config")
    }

    fn prove_guest(
        case: GuestCase,
        backend: JoltBackend<Fr, DoryScheme>,
        inspect_trace: impl FnOnce(&[JoltTraceRow]),
    ) -> ProvedGuest {
        let run = support::prepare(&case);
        inspect_trace(run.trace.trace.proof_rows());
        let mut config = derive_config(&run);
        // Exercise inactive cycle rounds and RAF claim scaling through BlindFold.
        config.rw_config.ram_rw_phase1_num_rounds = 0;
        config.rw_config.registers_rw_phase1_num_rounds = 0;
        let shared = JoltSharedPreprocessing::new(run.preprocessing).expect("shared preprocessing");
        let preprocessing = jolt_prover::dory::from_shared(shared).expect("Dory preprocessing");
        assert!(preprocessing.verifier.vc_setup.is_some());
        let program_preprocessing = preprocessing
            .program_arc()
            .expect("full program preprocessing");
        let public_io = run.trace.device.clone();
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            )
            .include_trusted_advice(!case.trusted_advice.is_empty())
            .include_untrusted_advice(!case.untrusted_advice.is_empty()),
            JoltVmWitnessInputs::new(&run.program, &program_preprocessing, run.trace),
        );
        let trusted = (!case.trusted_advice.is_empty()).then(|| {
            jolt_prover::dory::commit_trusted_advice(&preprocessing, &case.trusted_advice)
                .expect("trusted advice commitment")
        });
        let proof = jolt_prover::dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
            backend.with_witness(&witness),
            &preprocessing,
            &config,
            trusted.as_ref(),
            &public_io,
        )
        .expect("modular ZK prove");
        ProvedGuest {
            trusted_advice_commitment: trusted.map(|entry| entry.commitment),
            preprocessing,
            public_io,
            proof,
        }
    }

    fn prove_muldiv(backend: JoltBackend<Fr, DoryScheme>) -> ProvedGuest {
        prove_guest(muldiv_case(), backend, |_| {})
    }

    fn verify(proved: &ProvedGuest) -> Result<(), VerifierError> {
        jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
            &proved.preprocessing.verifier,
            &proved.public_io,
            &proved.proof,
            proved.trusted_advice_commitment.as_ref(),
        )
    }

    /// The reference kernel tier under the ZK envelope; the optimized tier is
    /// covered by the guest matrix.
    #[test]
    fn zk_muldiv_reference_backend_proof_is_accepted() {
        with_zk_stack(|| {
            let proved = prove_muldiv(JoltBackend::reference());
            assert_eq!(proved.proof.protocol, JoltProtocolConfig::for_zk(true));
            verify(&proved).expect("modular ZK proof must verify");
        });
    }

    /// A proved muldiv ZK proof, its secrets, and the verifier's masked
    /// stage outputs replayed from its argument string.
    struct CheatingProver {
        masked: MaskedStages<Fr, Bn254G1>,
        secrets: ZkSecrets<Fr>,
        /// The honest masked auxiliary block and its relation: a shift that
        /// leaves the block unchanged leaves the relation unchanged.
        honest: (Vec<Fr>, ResidualRelation<Fr>),
    }

    impl CheatingProver {
        fn new() -> Self {
            let proved = prove_muldiv(JoltBackend::optimized());
            let secrets = zk_secrets::take::<Fr>().expect("a ZK prove records its secrets");
            let mut transcript = VerifierTranscript::<JoltSponge>::new(
                &jolt_protocol_id::<JoltSponge>(),
                JOLT_SESSION,
                &proved.proof.narg,
            );
            let VerifiedStages::Zk(masked) =
                verify_stages::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
                    &proved.preprocessing.verifier,
                    &proved.public_io,
                    proved.trusted_advice_commitment.as_ref(),
                    &mut transcript,
                )
                .expect("stage spine accepts the honest proof")
            else {
                panic!("a ZK proof yields masked stages");
            };
            let masked = *masked;
            let block = Self::masked_auxiliary(&masked, &secrets.pads);
            let relation = masked.relation(&block).expect("residual relation");
            let cheater = Self {
                masked,
                secrets,
                honest: (block, relation),
            };
            assert_eq!(cheater.violation(&cheater.secrets.pads), None);
            cheater
        }

        /// The auxiliary block a prover holding `pads` sends: the masked
        /// messages already sent decode against `pads`, and the products are
        /// recomputed from those decoded values to fit.
        fn masked_auxiliary(masked: &MaskedStages<Fr, Bn254G1>, pads: &[Fr]) -> Vec<Fr> {
            let auxiliary = masked
                .residual
                .auxiliary_values(|operand| Ok(operand.masked - pads[operand.pad.0]))
                .expect("auxiliary values");
            let start = masked.pads.layout().auxiliary.start;
            auxiliary
                .iter()
                .zip(&pads[start..])
                .map(|(value, pad)| *value + *pad)
                .collect()
        }

        /// The relation that prover's argument string yields.
        fn relation(&self, pads: &[Fr]) -> ResidualRelation<Fr> {
            self.masked
                .relation(&Self::masked_auxiliary(&self.masked, pads))
                .expect("residual relation")
        }

        fn violation(&self, pads: &[Fr]) -> Option<usize> {
            let witness = [self.secrets.joint_evaluation];
            let block = Self::masked_auxiliary(&self.masked, pads);
            let (honest_block, honest_relation) = &self.honest;
            let rebuilt;
            let relation = if block == *honest_block {
                honest_relation
            } else {
                rebuilt = self.masked.relation(&block).expect("residual relation");
                &rebuilt
            };
            relation
                .first_violation(pads, &witness)
                .expect("evaluate residual rows")
        }

        /// `pads` with `shifts` applied: lowering pad i by δ raises the value
        /// m̂_i − p_i the masked message m̂_i decodes to by δ.
        fn shifted(&self, shifts: &[(usize, Fr)]) -> Vec<Fr> {
            let mut pads = self.secrets.pads.clone();
            for (slot, delta) in shifts {
                pads[*slot] -= *delta;
            }
            pads
        }
    }

    /// Soundness probe: shifting any one decoded base value of the muldiv
    /// proof by δ, keeping its masked message, and recomputing the
    /// auxiliary block as a cheating prover would, violates a residual row.
    /// The challenges stay those of the honest transcript, which a
    /// cheater's masked messages also reproduce.
    #[test]
    fn zk_muldiv_shifting_any_decoded_value_breaks_a_row() {
        with_zk_stack(|| {
            let cheater = CheatingProver::new();
            let delta = Fr::from_u64(7);
            let base = cheater.masked.pads.layout().auxiliary.start;
            let undetected: Vec<usize> = (0..base)
                .filter(|&slot| {
                    cheater
                        .violation(&cheater.shifted(&[(slot, delta)]))
                        .is_none()
                })
                .collect();
            assert!(
                undetected.is_empty(),
                "decoded values no row catches: {undetected:?}"
            );
        });
    }

    /// The stage-6b runtime alias is sent masked with an equality row to its
    /// read-RAF source. Shifting the alias cell alone violates that equality
    /// row; shifting it together with its source keeps them equal, and the
    /// rows that consume the two copies still catch the shift.
    #[test]
    fn zk_muldiv_stage6b_alias_is_bound_to_its_source() {
        with_zk_stack(|| {
            let cheater = CheatingProver::new();
            let honest = cheater.relation(&cheater.secrets.pads);
            let (row, alias, source) = honest
                .affine
                .iter()
                .enumerate()
                .find_map(|(row, form)| match form.terms.as_slice() {
                    [(ResidualVar::Pad(a), ca), (ResidualVar::Pad(b), cb)]
                        if *ca + *cb == Fr::from_u64(0) =>
                    {
                        Some((row, a.0.max(b.0), a.0.min(b.0)))
                    }
                    _ => None,
                })
                .expect("muldiv's stage 6b aliases a bytecode RA opening");
            let delta = Fr::from_u64(7);

            let pads = cheater.shifted(&[(alias, delta)]);
            let relation = cheater.relation(&pads);
            let witness = [cheater.secrets.joint_evaluation];
            assert!(!relation.affine[row]
                .evaluate(&pads, &witness)
                .expect("evaluate the equality row")
                .is_zero());

            let pads = cheater.shifted(&[(alias, delta), (source, delta)]);
            let relation = cheater.relation(&pads);
            assert!(relation.affine[row]
                .evaluate(&pads, &witness)
                .expect("evaluate the equality row")
                .is_zero());
            assert!(cheater.violation(&pads).is_some());
        });
    }

    #[test]
    fn zk_muldiv_tampered_blindfold_is_rejected() {
        with_zk_stack(|| {
            let mut proved = prove_muldiv(JoltBackend::reference());
            // The argument string ends with BlindFold's openings.
            *proved.proof.narg.last_mut().expect("non-empty proof") ^= 1;
            assert!(verify(&proved).is_err());
        });
    }

    #[test]
    fn zk_advice_consumer_modular_proof_is_accepted() {
        with_zk_stack(|| {
            let proved = prove_guest(
                GuestCase {
                    inputs: postcard::to_stdvec(&12u64).expect("serialize input"),
                    untrusted_advice: postcard::to_stdvec(&5u64)
                        .expect("serialize untrusted advice"),
                    trusted_advice: postcard::to_stdvec(&7u64).expect("serialize trusted advice"),
                    ..GuestCase::new("advice-consumer-guest")
                },
                JoltBackend::reference(),
                |_| {},
            );
            assert!(!proved.public_io.untrusted_advice.is_empty());
            verify(&proved).expect("modular ZK advice proof must verify");
        });
    }

    #[test]
    fn zk_committed_muldiv_modular_proof_is_accepted() {
        with_zk_stack(|| {
            let run = support::prepare(&muldiv_case());
            let config = derive_config(&run);
            let preprocessing = jolt_prover::dory::preprocess_committed(run.preprocessing)
                .expect("committed preprocessing");
            let program_preprocessing = preprocessing.program_arc().expect("retained full program");
            let public_io = run.trace.device.clone();
            let witness = TraceBackend::new(
                JoltVmWitnessConfig::new(
                    config.trace_length.ilog2() as usize,
                    config.ram_K,
                    config.one_hot_config,
                ),
                JoltVmWitnessInputs::new(&run.program, &program_preprocessing, run.trace),
            );
            let proof = jolt_prover::dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
                JoltBackend::reference().with_witness(&witness),
                &preprocessing,
                &config,
                None,
                &public_io,
            )
            .expect("committed ZK prove");
            jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
                &preprocessing.verifier,
                &public_io,
                &proof,
                None,
            )
            .expect("committed ZK proof must verify");
        });
    }
}
