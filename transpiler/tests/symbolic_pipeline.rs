//! End-to-end execution of the symbolic transpilation pipeline (T5 + T1/T2/T3/T4
//! together) on a REAL non-ZK Poseidon proof — the path the unit tests and the
//! real-transcript parity test do NOT exercise: `MleAst` flowing through all of
//! stages 1–7 over `SymbolicVerifierFs` + `AstOpeningAccumulator`, producing an
//! `AstBundle`.
//!
//! Only meaningful under `transcript-poseidon` (the symbolic sponge models
//! Poseidon) and non-ZK (the transpiler refuses ZK). Run:
//!   cargo nextest run -p transpiler --features transcript-poseidon symbolic_pipeline
#![cfg(all(feature = "transcript-poseidon", not(feature = "zk")))]

use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::field::JoltField;
use jolt_core::transcript_msgs::{FsAbsorb, FsChallenge, VerifierFs};
use jolt_core::zkvm::verifier::JoltVerifierPreprocessing;
use jolt_transcript::{verifier_transcript, PoseidonSponge, VerificationResult, VerifierState};
use serial_test::serial;
use transpiler::pipeline::{bundle_needs_non_native, run_symbolic_pipeline};

/// Build a real non-ZK muldiv proof (Poseidon sponge under this feature set,
/// inputs `[9, 5, 3]`) plus the io device and verifier preprocessing the
/// pipeline consumes, mirroring jolt-core's `muldiv_e2e_dory` / parity test.
fn build_muldiv_proof() -> (
    jolt_core::zkvm::RV64IMACProof,
    common::jolt_device::JoltDevice,
    JoltVerifierPreprocessing<
        Fr,
        jolt_core::curve::Bn254Curve,
        jolt_core::poly::commitment::dory::DoryCommitmentScheme,
    >,
) {
    use jolt_core::host;
    use jolt_core::zkvm::program::ProgramPreprocessing;
    use jolt_core::zkvm::prover::JoltProverPreprocessing;
    use jolt_core::zkvm::verifier::JoltSharedPreprocessing;
    use jolt_core::zkvm::RV64IMACProver;

    let mut program = host::Program::new("muldiv-guest");
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let program_pp =
        ProgramPreprocessing::preprocess(bytecode, init_memory_state, e_entry).expect("preprocess");
    let shared = JoltSharedPreprocessing::new(program_pp, io_device.memory_layout.clone(), 1 << 16);
    let prover_pp = JoltProverPreprocessing::new(shared);
    let elf = program.get_elf_contents().expect("elf contents");
    let prover =
        RV64IMACProver::gen_from_elf(&prover_pp, &elf, &inputs, &[], &[], None, None, None);
    let io_device = prover.program_io.clone();
    let (proof, _debug) = prover.prove();
    assert!(!proof.zk_mode, "test fixture must be a non-ZK proof");

    let verifier_pp = JoltVerifierPreprocessing::from(&prover_pp);
    (proof, io_device, verifier_pp)
}

/// `VerifierFs<Fr>` wrapper that delegates every call to the REAL Poseidon
/// verifier transcript and records each squeezed challenge — the native half of
/// the dispatch-layer challenge differential below.
struct RecordingFs<'a> {
    inner: VerifierState<'a, PoseidonSponge>,
    challenges: Vec<Fr>,
}

impl FsChallenge<Fr> for RecordingFs<'_> {
    fn challenge_field(&mut self) -> Fr {
        let c: Fr = self.inner.challenge_field();
        self.challenges.push(c);
        c
    }
    fn challenge_optimized(&mut self) -> <Fr as JoltField>::Challenge {
        let c = FsChallenge::<Fr>::challenge_optimized(&mut self.inner);
        self.challenges.push(c.into());
        c
    }
}

impl FsAbsorb for RecordingFs<'_> {
    fn absorb<T: CanonicalSerialize>(&mut self, value: &T) {
        self.inner.absorb(value);
    }
    fn absorb_slice<T: CanonicalSerialize>(&mut self, values: &[T]) {
        self.inner.absorb_slice(values);
    }
    fn absorb_bytes(&mut self, bytes: &[u8]) {
        self.inner.absorb_bytes(bytes);
    }
    fn absorb_scalar<T: CanonicalSerialize>(&mut self, value: &T) {
        self.inner.absorb_scalar(value);
    }
    fn absorb_scalars<T: CanonicalSerialize + Clone>(&mut self, values: &[T]) {
        self.inner.absorb_scalars(values);
    }
    fn absorb_commitment<T: CanonicalSerialize>(&mut self, value: &T) {
        self.inner.absorb_commitment(value);
    }
    fn absorb_commitment_bytes(&mut self, bytes: &[u8]) {
        self.inner.absorb_commitment_bytes(bytes);
    }
}

impl VerifierFs<Fr> for RecordingFs<'_> {
    fn read_slice<T: CanonicalDeserialize>(&mut self) -> VerificationResult<Vec<T>> {
        VerifierFs::<Fr>::read_slice(&mut self.inner)
    }
    fn read_scalars(&mut self) -> VerificationResult<Vec<Fr>> {
        VerifierFs::<Fr>::read_scalars(&mut self.inner)
    }
    fn read_commitments<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
    ) -> VerificationResult<Vec<T>> {
        VerifierFs::<Fr>::read_commitments(&mut self.inner)
    }
}

#[test]
#[serial]
fn symbolic_pipeline_runs_on_real_muldiv_poseidon_proof() {
    // 1. Build a real muldiv proof (Poseidon sponge, under this feature set),
    //    mirroring jolt-core's `muldiv_e2e_dory` / parity test.
    let (proof, io_device, verifier_pp) = build_muldiv_proof();

    // 2. Run the ACTUAL symbolic pipeline (the binary's path) on the real proof.
    let out = run_symbolic_pipeline(&proof, io_device.clone(), &verifier_pp, None)
        .expect("symbolic pipeline must complete on a real proof");

    // 3. Structural assertions: the full stage-1..7 symbolic replay produced a
    //    non-trivial circuit. (Value-level agreement with the native verifier is
    //    the field-aligned differential gates' job — poseidon_model + the
    //    FieldAlignedLayout test in verifier_fs.rs.)
    assert!(out.num_frames > 0, "NARG had no frames to replay");
    assert!(
        out.num_assertions > 0,
        "symbolic replay recorded zero constraints — verifier asserted nothing"
    );
    assert!(
        !out.bundle.inputs.is_empty(),
        "no witness inputs allocated during replay"
    );
    assert!(
        !out.bundle.constraints.is_empty(),
        "bundle has no constraints"
    );
    assert!(
        !bundle_needs_non_native(&out.bundle),
        "stages 1-7 must be native-field only (no Fq witness)"
    );
    // T-O5: the pipeline uses the field-aligned sponge layout, so the bundle's
    // Fiat-Shamir challenges match the native verifier (proven value-exact by the
    // `poseidon_model` + `field_aligned_layout_matches_native_sponge` differential
    // gates).
    assert!(
        out.sponge_faithful,
        "pipeline must use a value-faithful sponge layout (T-O5)"
    );

    println!(
        "symbolic pipeline OK: {} frames, {} assertions, {} inputs, {} constraints, {} arena nodes",
        out.num_frames,
        out.num_assertions,
        out.bundle.inputs.len(),
        out.bundle.constraints.len(),
        out.bundle.nodes.len(),
    );

    // 4. THE DISPATCH-LAYER CHALLENGE DIFFERENTIAL: evaluate every Fiat-Shamir
    //    challenge AST the symbolic replay squeezed against the recorded witness,
    //    replay the SAME proof through the native `TranspilableVerifier` over the
    //    REAL Poseidon transcript, and assert the two challenge sequences are
    //    element-wise equal. The layout differential (`poseidon_model` +
    //    `field_aligned_layout_matches_native_sponge`) proves the sponge layout is
    //    value-faithful for a hand-built schedule; this closes the remaining
    //    compositional gap — that the symbolic replay drives the layout with the
    //    EXACT absorb/squeeze schedule the native verifier executes.
    assert!(
        !out.squeezed_challenge_roots.is_empty(),
        "symbolic replay squeezed no challenges"
    );
    let witness = out.var_alloc.witness_fr_map();
    // One shared memo cache across roots (challenge k's AST embeds the whole
    // chain of challenges 1..k-1).
    let symbolic_challenges = transpiler::ast_evaluator::eval_roots(
        &out.bundle.nodes,
        &out.squeezed_challenge_roots,
        &witness,
    );

    let native_challenges = native_verifier_challenges(&proof, io_device, &verifier_pp);

    assert_eq!(
        symbolic_challenges.len(),
        native_challenges.len(),
        "symbolic and native replays squeezed different challenge COUNTS — \
         absorb/squeeze schedule mismatch in the dispatch layer"
    );
    assert_eq!(
        symbolic_challenges, native_challenges,
        "symbolic challenge ASTs evaluate differently from the native verifier's \
         Fiat-Shamir challenges on the same proof"
    );
    println!(
        "challenge differential OK: {} challenges match the native verifier",
        native_challenges.len()
    );
}

/// Replay `proof` through the native `TranspilableVerifier` over the real Poseidon
/// verifier transcript (built exactly like the `muldiv_transpilable_verifier_parity`
/// test in jolt-core), recording every squeezed challenge.
fn native_verifier_challenges(
    proof: &jolt_core::zkvm::RV64IMACProof,
    io_device: common::jolt_device::JoltDevice,
    verifier_pp: &JoltVerifierPreprocessing<
        Fr,
        jolt_core::curve::Bn254Curve,
        jolt_core::poly::commitment::dory::DoryCommitmentScheme,
    >,
) -> Vec<Fr> {
    use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
    use jolt_core::poly::opening_proof::VerifierOpeningAccumulator;
    use jolt_core::utils::math::Math;
    use jolt_core::zkvm::fiat_shamir_instance;
    use jolt_core::zkvm::transpilable_verifier::{TranspilableProofData, TranspilableVerifier};

    // Pre-seed the accumulator with the structural opening claims, exactly as
    // `JoltVerifier::new` does.
    let mut accumulator = VerifierOpeningAccumulator::<Fr>::new(proof.trace_length.log_2(), false);
    accumulator.preseed_structural_claims(&proof.opening_claims.0);

    let proof_data = TranspilableProofData::from_proof(proof);
    let mut verifier: TranspilableVerifier<
        '_,
        Fr,
        jolt_core::curve::Bn254Curve,
        DoryCommitmentScheme,
        VerifierOpeningAccumulator<Fr>,
    > = TranspilableVerifier::new(verifier_pp, proof_data, io_device, None, accumulator)
        .expect("native TranspilableVerifier construction failed");

    // Real transcript over the proof's NARG, mirroring `verify_inner` (instance
    // over the TRUNCATED program_io the verifier holds).
    let preprocessing_digest = verifier_pp.shared.digest();
    let instance = fiat_shamir_instance(
        &verifier.program_io,
        proof.ram_K,
        proof.trace_length,
        verifier_pp.shared.program_meta.entry_address,
        &proof.rw_config,
        &proof.one_hot_config,
        proof.dory_layout,
        &preprocessing_digest,
    );
    let mut recorder = RecordingFs {
        inner: verifier_transcript(b"Jolt", instance, PoseidonSponge::default(), &proof.narg),
        challenges: Vec::new(),
    };

    verifier
        .verify(&mut recorder)
        .expect("native TranspilableVerifier failed on the real proof");
    recorder
        .inner
        .check_eof()
        .expect("NARG not fully consumed after stage 7");
    recorder.challenges
}
