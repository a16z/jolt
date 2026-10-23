//! Whiteout's masked stage spine on the Akita build (W6d prerequisite):
//! stages 1–7 proved masked over uncommitted pads, replayed by the masked
//! verifier, and checked against the residual relation the verifier lowers
//! from the Akita relations' own formula owners. Whiteout's pad commitment,
//! masked opening and Σ tail are W6d, so a Whiteout build rejects every
//! proof until then.

#[cfg(all(
    feature = "prover-fixtures",
    feature = "akita",
    feature = "zk",
    not(feature = "field-inline")
))]
mod support;

#[cfg(all(
    feature = "prover-fixtures",
    feature = "akita",
    feature = "zk",
    not(feature = "field-inline")
))]
#[expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "integration tests should fail loudly"
)]
mod whiteout {
    use common::jolt_device::JoltDevice;
    use jolt_akita::{AkitaCommitment, AkitaField, AkitaScheduleArtifacts, AkitaScheme};
    use jolt_algebra::Ring;
    use jolt_prover::akita::preprocessing::{self, AkitaProverPreprocessing, AkitaVc};
    use jolt_prover::akita::{self, MaskedStagesProof};
    use jolt_prover::{JoltBackend, ProverConfig, ProverError};
    use jolt_transcript::VerifierTranscript;
    use jolt_verifier::proof::JoltProof;
    use jolt_verifier::stages::zk::residual::JoltResidual;
    use jolt_verifier::{
        jolt_protocol_id, seed_transcript, verify_masked_stages, JoltProtocolConfig, JoltSponge,
        VerifierError, ZkConfig, JOLT_SESSION,
    };
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use jolt_zk::masking::{CommittedPadsView, PadLayout, ResidualRelation, ResidualRows};

    use crate::support::{self, GuestCase, PreparedGuest};

    type F = AkitaField;

    /// Which program shape a case proves.
    #[derive(Clone, Copy)]
    enum Case {
        Muldiv,
        CommittedMuldiv,
        /// Untrusted and trusted advice.
        Advice,
    }

    struct Proved {
        preprocessing: AkitaProverPreprocessing,
        public_io: JoltDevice,
        trusted_advice_commitment: Option<AkitaCommitment>,
        stages: MaskedStagesProof<F>,
        /// The production entry's result on the same inputs.
        production: Result<JoltProof, ProverError<F>>,
    }

    fn prepare(case: Case) -> PreparedGuest {
        match case {
            Case::Muldiv | Case::CommittedMuldiv => support::prepare(&GuestCase {
                inputs: postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs"),
                ..GuestCase::new("muldiv-guest")
            }),
            Case::Advice => support::prepare(&GuestCase {
                inputs: postcard::to_stdvec(&12u64).expect("serialize inputs"),
                untrusted_advice: postcard::to_stdvec(&5u64).expect("serialize advice"),
                trusted_advice: postcard::to_stdvec(&7u64).expect("serialize advice"),
                ..GuestCase::new("advice-consumer-guest")
            }),
        }
    }

    fn prove(case: Case) -> Proved {
        let run = prepare(case);
        let config = ProverConfig::derive_from_dimensions::<F>(
            run.trace.dimensions,
            &run.preprocessing.memory_layout,
            run.preprocessing.ram.min_bytecode_address,
            run.preprocessing.ram.bytecode_words.len(),
            run.preprocessing.max_padded_trace_length,
        )
        .expect("derive config");
        let artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
        let advice = matches!(case, Case::Advice);
        let preprocessing = match case {
            Case::Muldiv | Case::Advice => preprocessing::preprocess_full_with_advice(
                &artifacts,
                run.preprocessing,
                &config,
                advice,
                advice,
            ),
            Case::CommittedMuldiv => {
                preprocessing::preprocess_committed(&artifacts, run.preprocessing, &config)
            }
        }
        .expect("Akita preprocessing");
        let trusted = advice.then(|| {
            preprocessing::commit_trusted_advice(
                &preprocessing,
                &postcard::to_stdvec(&7u64).expect("serialize advice"),
            )
            .expect("trusted advice commitment")
        });
        let program = preprocessing.program_arc().expect("full program");
        let public_io = run.trace.device.clone();
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            )
            .include_untrusted_advice(advice)
            .include_trusted_advice(advice),
            JoltVmWitnessInputs::new(&run.program, &program, run.trace),
        );
        let backend = JoltBackend::optimized();
        let production = akita::prove::<F, AkitaScheme, AkitaVc, JoltSponge>(
            backend.with_witness(&witness),
            &preprocessing,
            &config,
            trusted.as_ref(),
            &public_io,
        );
        let stages = akita::prove_masked_stages::<F, AkitaScheme, AkitaVc, JoltSponge>(
            backend.with_witness(&witness),
            &preprocessing,
            &config,
            trusted.as_ref(),
            &public_io,
        )
        .expect("masked stages 0-7");
        Proved {
            preprocessing,
            public_io,
            trusted_advice_commitment: trusted.map(|object| object.commitment),
            stages,
            production,
        }
    }

    /// The masked verifier's replay of the stage spine: its residual and the
    /// layout its counting view and lowering derive.
    fn replay(proved: &Proved) -> (JoltResidual<F>, PadLayout) {
        let mut transcript = VerifierTranscript::<JoltSponge>::new(
            &jolt_protocol_id::<JoltSponge>(),
            JOLT_SESSION,
            &proved.stages.narg,
        );
        let seeded = seed_transcript::<AkitaScheme, AkitaVc, JoltSponge>(
            &proved.preprocessing.verifier,
            &proved.public_io,
            proved.trusted_advice_commitment.as_ref(),
            &mut transcript,
        )
        .expect("seeding messages");
        let mut view = CommittedPadsView::counting();
        let residual = verify_masked_stages::<F, AkitaScheme, AkitaVc, JoltSponge, ()>(
            &proved.preprocessing.verifier,
            &seeded,
            &mut transcript,
            &mut view,
        )
        .expect("masked stage spine");
        assert_eq!(transcript.remaining(), 0, "the replay reads every byte");
        let layout = residual.pad_layout(view.used()).expect("pad layout");
        (residual, layout)
    }

    /// The relation a prover holding `pads` yields: the masked messages
    /// decode against `pads`, and the auxiliary block is recomputed from the
    /// decoded values, as a consistent cheater would.
    fn relation(residual: &JoltResidual<F>, layout: &PadLayout, pads: &[F]) -> ResidualRelation<F> {
        let auxiliary = residual
            .auxiliary_values(|operand| Ok(operand.masked - pads[operand.pad.0]))
            .expect("auxiliary values");
        let masked: Vec<F> = auxiliary
            .iter()
            .zip(&pads[layout.auxiliary.clone()])
            .map(|(value, pad)| *value + *pad)
            .collect();
        let mut relation = ResidualRelation::new(layout.clone(), 0);
        residual
            .rows(&masked)
            .append_rows(&mut relation)
            .expect("residual rows");
        relation
    }

    /// Honest pads satisfy every row; the verifier-derived layout fills the
    /// prover's pads; and every base slot appears in some row: shifting any
    /// one decoded value, with the auxiliary block recomputed consistently,
    /// breaks a row. This does not show that each value is uniquely
    /// determined: slots read only by the final opening are bound by W6d's
    /// opening, whose tamper tests cover compensating shifts across them.
    fn check_every_slot_appears_in_a_row(case: Case) {
        let proved = prove(case);
        let (residual, layout) = replay(&proved);
        let pads = &proved.stages.pads;
        assert_eq!(layout.num_pads, pads.len());
        assert_eq!(
            relation(&residual, &layout, pads)
                .first_violation(pads, &[])
                .expect("evaluate rows"),
            None
        );
        let delta = F::from_u64(7);
        let unread: Vec<usize> = (0..layout.auxiliary.start)
            .filter(|&slot| {
                let mut shifted = pads.clone();
                shifted[slot] -= delta;
                relation(&residual, &layout, &shifted)
                    .first_violation(&shifted, &[])
                    .expect("evaluate rows")
                    .is_none()
            })
            .collect();
        assert!(unread.is_empty(), "slots no row reads: {unread:?}");
    }

    #[test]
    fn whiteout_muldiv_every_slot_appears_in_a_residual_row() {
        check_every_slot_appears_in_a_row(Case::Muldiv);
    }

    #[test]
    fn whiteout_committed_muldiv_every_slot_appears_in_a_residual_row() {
        check_every_slot_appears_in_a_row(Case::CommittedMuldiv);
    }

    #[test]
    fn whiteout_advice_every_slot_appears_in_a_residual_row() {
        check_every_slot_appears_in_a_row(Case::Advice);
    }

    /// A Whiteout build accepts nothing before W6d: the prover's production
    /// entry refuses, and the verifier rejects where the pad commitment would
    /// be read.
    #[test]
    fn whiteout_rejects_before_the_tail_exists() {
        let proved = prove(Case::Muldiv);
        assert!(matches!(
            proved.production,
            Err(ProverError::Unsupported { .. })
        ));
        let proof = JoltProof {
            protocol: JoltProtocolConfig::for_zk(ZkConfig::Whiteout),
            narg: proved.stages.narg.clone(),
        };
        assert!(matches!(
            jolt_verifier::verify::<F, AkitaScheme, AkitaVc, JoltSponge>(
                &proved.preprocessing.verifier,
                &proved.public_io,
                &proof,
                None,
            ),
            Err(VerifierError::ProtocolAxisUnimplemented { .. })
        ));
    }
}
