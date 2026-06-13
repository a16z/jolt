//! The symbolic transpilation pipeline as a callable function, so both the
//! `transpiler` binary and the integration test (`tests/symbolic_pipeline.rs`)
//! drive the *identical* path: symbolize the proof's structural parts, replay
//! the NARG frames through `TranspilableVerifier` stages 1–7 over the symbolic
//! `SymbolicVerifierFs` + `AstOpeningAccumulator`, and build the `AstBundle`.
//!
//! Non-ZK only (spec §16/§17): `symbolize_proof` refuses `zk_mode`.

use std::cell::RefCell;
use std::rc::Rc;

use ark_bn254::Fr;
use jolt_core::curve::Bn254Curve;
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryContext, DoryGlobals};
use jolt_core::zkvm::fiat_shamir_instance;
use jolt_core::zkvm::transpilable_verifier::TranspilableVerifier;
use jolt_core::zkvm::verifier::JoltVerifierPreprocessing;
use jolt_core::zkvm::RV64IMACProof;
use zklean_extractor::mle_ast::{
    disable_constraint_mode, enable_constraint_mode, take_constraints as take_assertions,
    AstBundle, MleAst, TargetField, WitnessType,
};
use zklean_extractor::AstCommitment;

use crate::symbolic_proof::{symbolize_proof, SymbolizedProof, VarAllocator};
use crate::symbolic_traits::{FrameLabel, SymbolicVerifierFs};
use crate::{AstCommitmentScheme, AstCurve, AstOpeningAccumulator};

type RealPreprocessing = JoltVerifierPreprocessing<Fr, Bn254Curve, DoryCommitmentScheme>;

#[derive(Debug)]
pub enum PipelineError {
    Symbolize(crate::narg_parser::NargParseError),
    VerifierConstruction(jolt_core::utils::errors::ProofVerifyError),
    Replay(jolt_core::utils::errors::ProofVerifyError),
    /// Replay finished but left NARG frames unread — a frame/read-order mismatch.
    UnconsumedFrames(usize),
    /// The witness allocator's `Rc` was still shared after replay (a symbolizer
    /// closure or `fs` clone outlived the replay) — an internal invariant violation.
    VarAllocatorStillShared,
    /// The build's `transcript-poseidon` feature is off, so jolt-core proves/verifies
    /// under a byte sponge while the symbolic mirror models the field-aligned Poseidon
    /// sponge — every circuit challenge would diverge from the native verifier's.
    WrongSpongeFeature,
    /// The single-threaded rayon pool the replay runs in could not be built.
    ThreadPool(rayon::ThreadPoolBuildError),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Symbolize(e) => write!(f, "symbolize_proof failed: {e}"),
            Self::VerifierConstruction(e) => write!(f, "verifier construction failed: {e:?}"),
            Self::Replay(e) => write!(f, "symbolic replay failed: {e:?}"),
            Self::UnconsumedFrames(n) => {
                write!(f, "{n} NARG frame(s) left unconsumed after stage 7")
            }
            Self::VarAllocatorStillShared => {
                write!(
                    f,
                    "var_alloc Rc still shared after replay (internal invariant)"
                )
            }
            Self::WrongSpongeFeature => {
                write!(
                    f,
                    "transpiler built without the `transcript-poseidon` feature: jolt-core's \
                     transcript is a byte sponge, but the symbolic mirror models the \
                     field-aligned Poseidon sponge — circuit challenges would diverge from \
                     the native verifier. Rebuild with `--features transcript-poseidon` \
                     (and a proof generated under it)."
                )
            }
            Self::ThreadPool(e) => {
                write!(f, "failed to build the single-threaded replay pool: {e}")
            }
        }
    }
}

impl std::error::Error for PipelineError {}

/// Result of a symbolic pipeline run.
pub struct PipelineOutput {
    pub bundle: AstBundle,
    pub var_alloc: VarAllocator,
    pub num_assertions: usize,
    pub num_frames: usize,
    /// Tier-2 canonicalization counters (`canonicalize_and_sweep`), printed by the binary.
    pub canon_stats: zklean_extractor::ast_bundle::CanonicalizeStats,
    /// `true` iff the symbolic sponge layout is value-faithful to the native
    /// transcript — the circuit's Fiat-Shamir challenges match the real verifier's.
    /// Driven by the layout's `SymbolicSpongeLayout::FAITHFUL` const; `true` with
    /// the field-aligned layout (T-O5), gated by the `poseidon_model` +
    /// `field_aligned_layout_matches_native_sponge` differential tests.
    /// (A1 guard — deviations-doc TDEV-6.)
    pub sponge_faithful: bool,
    /// Root NodeId (into `bundle.nodes`) of every Fiat-Shamir challenge the symbolic
    /// replay squeezed, in squeeze order (`SymbolicVerifierFs::squeezed_challenges`).
    /// Stored as post-`canonicalize_and_sweep` NodeIds — the pass remaps them along
    /// with the constraint roots, so they stay valid against the compacted arena.
    /// Consumed by the in-CI real-proof challenge differential in
    /// `tests/symbolic_pipeline.rs`, which evaluates each AST against the witness and
    /// asserts equality with the native verifier's challenge sequence on the same proof.
    pub squeezed_challenge_roots: Vec<zklean_extractor::mle_ast::NodeId>,
}

impl PipelineOutput {
    /// Human-readable reasons this bundle must NOT be deployed as an on-chain verifier
    /// (empty ⇒ no known blocker). The transpiler binary prints these prominently; a
    /// future deploy path MUST hard-refuse while any remain. Mirrors the spec's
    /// no-partial-artifact rule (§7.12 / §8 Phase 3).
    pub fn deployment_blockers(&self) -> Vec<&'static str> {
        let mut blockers = Vec::new();
        if !self.sponge_faithful {
            blockers.push(
                "Fiat-Shamir sponge layout is not value-faithful; circuit challenges \
                 diverge from the native verifier.",
            );
        }
        if !cfg!(feature = "transcript-poseidon") {
            // Unreachable through `run_symbolic_pipeline` (it errors with
            // `WrongSpongeFeature` first); kept so a `PipelineOutput` can never claim
            // deployability under a non-Poseidon build.
            blockers.push(
                "built without `transcript-poseidon`: jolt-core's transcript is a byte \
                 sponge, the symbolic mirror models Poseidon — challenges diverge.",
            );
        }
        // Structural: stages 1-7 only — the deferred stage-8 PCS check is what binds the
        // committed polynomials and final opening claims (spec §7.12 no-partial-artifact).
        blockers.push(
            "stage 8 (Dory PCS) is not transpiled; committed polynomials and final opening \
             claims are unbound (spec §7.12 / §8 Phase 3).",
        );
        blockers
    }
}

/// Run the full symbolic transpilation pipeline (stages 1–7) over a real non-ZK
/// proof, producing the `AstBundle` and witness allocator.
///
/// `real_trusted_advice` is the host-side `Option<PCS::Commitment>` (symbolized
/// internally if present), matching the binary's CLI-loaded commitment.
///
/// The replay drives jolt-core verifier code whose internal `par_iter` reductions
/// (e.g. `EqPolynomial::mle`) allocate AST nodes into the process-global arena;
/// under a multi-threaded rayon pool the allocation ORDER — and therefore every
/// NodeId, the `gcse[i]` slot assignment, and the emitted Go circuit — is
/// scheduling-dependent and permutes between runs (so would the Groth16 keys
/// compiled from it). The whole pipeline therefore runs inside a single-threaded
/// rayon pool, which makes node creation deterministic. Running the entire body
/// on the pool's one worker thread also keeps every thread-local hook pair
/// (read symbolizer, pending initial RAM, constraint mode + accumulated
/// assertions) set and consumed on the same thread.
pub fn run_symbolic_pipeline(
    real_proof: &RV64IMACProof,
    io_device: common::jolt_device::JoltDevice,
    real_preprocessing: &RealPreprocessing,
    real_trusted_advice: Option<<DoryCommitmentScheme as jolt_core::poly::commitment::commitment_scheme::CommitmentScheme>::Commitment>,
) -> Result<PipelineOutput, PipelineError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        // 64MiB worker stack: the replay previously ran on the ~8MiB main
        // stack, but rayon workers default to ~2MiB — and this codebase has
        // known stack-depth sensitivity (deep recursive verifier/AST paths;
        // cf. the RUST_MIN_STACK=1GiB requirement for the opt-0 ZK suite).
        .stack_size(64 * 1024 * 1024)
        .build()
        .map_err(PipelineError::ThreadPool)?;
    pool.install(|| {
        run_symbolic_pipeline_inner(
            real_proof,
            io_device,
            real_preprocessing,
            real_trusted_advice,
        )
    })
}

fn run_symbolic_pipeline_inner(
    real_proof: &RV64IMACProof,
    io_device: common::jolt_device::JoltDevice,
    real_preprocessing: &RealPreprocessing,
    real_trusted_advice: Option<<DoryCommitmentScheme as jolt_core::poly::commitment::commitment_scheme::CommitmentScheme>::Commitment>,
) -> Result<PipelineOutput, PipelineError> {
    // The symbolic sponge mirrors the field-aligned `U = Fr` Poseidon transcript; a
    // non-Poseidon jolt-core build proves/verifies under a byte sponge, so the mirror
    // (and any bundle built from it) would be challenge-divergent garbage. A runtime
    // error — not `compile_error!` — because the lib must still compile featureless
    // (workspace clippy builds the transpiler without transcript features).
    if !cfg!(feature = "transcript-poseidon") {
        return Err(PipelineError::WrongSpongeFeature);
    }
    let mut var_alloc = VarAllocator::new();
    let SymbolizedProof {
        parsed_narg,
        accumulator,
        proof_data,
    } = symbolize_proof(real_proof, &mut var_alloc).map_err(PipelineError::Symbolize)?;

    let symbolic_trusted_advice = real_trusted_advice.as_ref().map(|commitment| {
        let chunks = var_alloc.alloc_commitment(commitment, "trusted_advice_commitment");
        AstCommitment::new(chunks)
    });

    // Symbolize IO + set the initial-RAM override (consumed by the verifier math).
    let (eval_input_words, _eval_output_words) =
        crate::symbolize::symbolize_io_device(&io_device, &mut var_alloc);
    {
        use jolt_core::zkvm::ram::{set_pending_initial_ram, PendingInitialRamValues};
        let bytecode_words: Vec<MleAst> = if real_preprocessing.shared.program.is_full() {
            #[expect(clippy::unwrap_used)]
            real_preprocessing
                .shared
                .program
                .as_full()
                .unwrap()
                .ram
                .bytecode_words
                .iter()
                .map(|&w| MleAst::from_u64(w))
                .collect()
        } else {
            Vec::new()
        };
        set_pending_initial_ram(PendingInitialRamValues {
            bytecode_words,
            input_words: eval_input_words,
        });
    }

    let symbolic_preprocessing = build_symbolic_preprocessing(real_preprocessing, &mut var_alloc);

    let mut verifier =
        TranspilableVerifier::<MleAst, AstCurve, AstCommitmentScheme, AstOpeningAccumulator>::new(
            &symbolic_preprocessing,
            proof_data,
            io_device,
            symbolic_trusted_advice,
            accumulator,
        )
        .map_err(PipelineError::VerifierConstruction)?;

    let preprocessing_digest = real_preprocessing.shared.digest();
    let instance = fiat_shamir_instance(
        &verifier.program_io,
        real_proof.ram_K,
        real_proof.trace_length,
        real_preprocessing.shared.program_meta.entry_address,
        &real_proof.rw_config,
        &real_proof.one_hot_config,
        real_proof.dory_layout,
        &preprocessing_digest,
    );

    // T-O5 field-aligned sponge: the circuit's Fiat-Shamir challenges reproduce the
    // native `U = Fr` `PoseidonSponge` value-exactly (verified by the differential
    // gates: `poseidon_model::model_matches_real_field_aligned_poseidon_transcript`
    // level 1, `field_aligned_layout_matches_native_sponge` level 2). The A1 guard
    // (deviations-doc TDEV-6) is driven by the layout's `FAITHFUL` const, so swapping
    // the layout type is the single switch — and would auto-re-arm the warning if
    // a non-faithful layout were ever substituted.
    type Layout = crate::symbolic_traits::FieldAlignedLayout;
    let sponge_faithful = <Layout as crate::symbolic_traits::SymbolicSpongeLayout>::FAITHFUL;
    if !sponge_faithful {
        eprintln!(
            "WARNING: symbolic sponge layout is not value-faithful; the circuit's \
             Fiat-Shamir challenges DIVERGE from the native verifier — bundle must NOT \
             be deployed on-chain (oracle: poseidon_model.rs)."
        );
    }

    let var_alloc = Rc::new(RefCell::new(var_alloc));
    let layout = Layout::new(b"Jolt", &instance);
    let mut fs = SymbolicVerifierFs::new(layout, parsed_narg, Rc::clone(&var_alloc));
    let num_frames = fs.remaining_frames();

    // RAII so the process-global constraint mode is disabled on EVERY exit path —
    // including the `?` early returns below (replay error / unconsumed frames) — not
    // just the happy path. (code-review #10: the explicit disable leaked on error.)
    struct ConstraintModeGuard;
    impl Drop for ConstraintModeGuard {
        fn drop(&mut self) {
            disable_constraint_mode();
        }
    }
    enable_constraint_mode();
    let _constraint_mode = ConstraintModeGuard;

    let _dory_guard = DoryGlobals::initialize_context(
        1 << verifier.one_hot_params.log_k_chunk,
        verifier.proof_data.trace_length.next_power_of_two(),
        DoryContext::Main,
        Some(verifier.proof_data.dory_layout),
    );

    let replay = (|| -> Result<(), jolt_core::utils::errors::ProofVerifyError> {
        fs.set_label(FrameLabel::Prestage);
        verifier.read_commitment_frames(&mut fs)?;
        fs.set_label(FrameLabel::StageWithUniskip(1));
        verifier.verify_stage1(&mut fs)?;
        fs.set_label(FrameLabel::StageWithUniskip(2));
        verifier.verify_stage2(&mut fs)?;
        fs.set_label(FrameLabel::Rounds("stage3_sumcheck".into()));
        verifier.verify_stage3(&mut fs)?;
        fs.set_label(FrameLabel::Rounds("stage4_sumcheck".into()));
        verifier.verify_stage4(&mut fs)?;
        fs.set_label(FrameLabel::Rounds("stage5_sumcheck".into()));
        verifier.verify_stage5(&mut fs)?;
        let (bytecode_read_raf_params, booleanity_params) = {
            fs.set_label(FrameLabel::Rounds("stage6a_sumcheck".into()));
            verifier.verify_stage6a(&mut fs)?
        };
        fs.set_label(FrameLabel::Rounds("stage6b_sumcheck".into()));
        verifier.verify_stage6b(&mut fs, bytecode_read_raf_params, booleanity_params)?;
        fs.set_label(FrameLabel::Rounds("stage7_sumcheck".into()));
        verifier.verify_stage7(&mut fs)?;
        Ok(())
    })();
    replay.map_err(PipelineError::Replay)?;

    let remaining = fs.remaining_frames();
    if remaining != 0 {
        return Err(PipelineError::UnconsumedFrames(remaining));
    }
    let squeezed_challenges = std::mem::take(&mut fs.squeezed_challenges);
    drop(fs);

    let assertions = take_assertions();
    // (constraint mode is disabled by `_constraint_mode`'s Drop at scope exit.)
    let num_assertions = assertions.len();

    let mut bundle = AstBundle::new();
    bundle.snapshot_arena();
    let var_alloc = Rc::try_unwrap(var_alloc)
        .map_err(|_| PipelineError::VarAllocatorStillShared)?
        .into_inner();
    for (idx, name, target_field) in var_alloc.descriptions_with_fields() {
        bundle.add_input_with_field(*idx, name.clone(), input_visibility(name), *target_field);
    }
    for (i, assertion) in assertions.iter().enumerate() {
        bundle.add_constraint_eq_zero(format!("assertion_{i}"), assertion.root());
    }
    // Tier-2 algebra passes (spec §5.1/§5.3): structural hash-consing + dead-node
    // sweep, post-hoc on the snapshot, BEFORE the CSE passes (CSE bindings are
    // computed on the canonicalized arena, so there is no stale-binding hazard).
    // The squeezed-challenge roots ride along so the challenge-differential ASTs
    // stay valid across the NodeId compaction.
    let mut squeezed_challenge_roots: Vec<_> =
        squeezed_challenges.iter().map(|c| c.root()).collect();
    let canon_stats = bundle.canonicalize_and_sweep(&mut squeezed_challenge_roots);
    bundle.run_global_cse();
    bundle.run_cse();

    Ok(PipelineOutput {
        bundle,
        var_alloc,
        num_assertions,
        num_frames,
        canon_stats,
        sponge_faithful,
        squeezed_challenge_roots,
    })
}

/// Circuit visibility for a symbolic input, keyed on the frozen Era-2 witness names
/// (`verifier_fs::FrameLabel`, `symbolize`, `symbolic_proof`).
///
/// `PublicStatement` (emitted as a gnark public input) is reserved for the program IO
/// that forms the public statement and the stage-8 binding values — opening claims and
/// polynomial/advice/trusted commitments — that the deferred native PCS check must see.
/// Everything else (sumcheck round polynomials, uni-skip coefficients) is `ProofData`
/// (secret witness): the circuit re-derives every Fiat-Shamir challenge from those
/// bytes in-circuit, so they are self-binding and need not be public. This is what
/// keeps the on-chain public-input count (and thus gas) bounded.
fn input_visibility(name: &str) -> WitnessType {
    const PUBLIC_PREFIXES: [&str; 5] = [
        "io_",
        "claim_",
        "commitment_",
        "untrusted_advice_commitment",
        "trusted_",
    ];
    // Self-binding proof bytes: sumcheck round polynomials and uni-skip coefficients,
    // named from the `FrameLabel::{StageWithUniskip,Rounds}` prefixes (`stage{n}_…`) or a
    // `Prestage` overflow frame (`prestage_f{k}_…`).
    //
    // WARNING: the bare "stage" prefix shadows ANY future `stage8_*` witness name —
    // a stage-8 public binding value named through the `FrameLabel` scheme would be
    // silently classified as secret proof data, unbindable by the on-chain wrapper.
    // The stage-8 PR must add an explicit carve-out for its public binding values
    // BEFORE reusing the FrameLabel naming scheme (spec §6.2).
    const SECRET_PREFIXES: [&str; 2] = ["stage", "prestage"];
    if PUBLIC_PREFIXES.iter().any(|p| name.starts_with(p)) {
        WitnessType::PublicStatement
    } else if SECRET_PREFIXES.iter().any(|p| name.starts_with(p)) {
        WitnessType::ProofData
    } else {
        // Fail loud rather than silently flip a renamed input to the wrong visibility:
        // a public commitment/claim leaking to `secret` would be unbindable by the future
        // on-chain wrapper, and proof data promoted to `public` would blow up gas. A new
        // witness name must be classified here explicitly.
        panic!(
            "input_visibility: unrecognized witness name {name:?}; add it to PUBLIC_PREFIXES \
             or SECRET_PREFIXES (did the frozen Era-2 naming scheme change?)"
        )
    }
}

/// True iff the bundle needs non-native (Fq) arithmetic, which codegen does not
/// yet support.
pub fn bundle_needs_non_native(bundle: &AstBundle) -> bool {
    bundle.has_inputs_for_field(TargetField::Fq)
}

/// Build the symbolic `JoltVerifierPreprocessing` (Dory generators → stub),
/// symbolizing trusted commitments in committed mode.
fn build_symbolic_preprocessing(
    real_preprocessing: &RealPreprocessing,
    var_alloc: &mut VarAllocator,
) -> JoltVerifierPreprocessing<MleAst, AstCurve, AstCommitmentScheme> {
    use jolt_core::zkvm::bytecode::TrustedBytecodeCommitments;
    use jolt_core::zkvm::program::{
        CommittedProgramPreprocessing, ProgramPreprocessing, TrustedProgramCommitments,
    };

    let symbolic_program: ProgramPreprocessing<AstCommitmentScheme> = match &real_preprocessing
        .shared
        .program
    {
        ProgramPreprocessing::Full(full) => ProgramPreprocessing::Full(full.clone()),
        ProgramPreprocessing::Committed(committed) => {
            let bytecode_commitments = TrustedBytecodeCommitments {
                commitments: committed
                    .bytecode_commitments
                    .commitments
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let chunks =
                            var_alloc.alloc_commitment(c, &format!("trusted_bytecode_{i}"));
                        AstCommitment::new(chunks)
                    })
                    .collect(),
                num_columns: committed.bytecode_commitments.num_columns,
                log_k_chunk: committed.bytecode_commitments.log_k_chunk,
                bytecode_chunk_count: committed.bytecode_commitments.bytecode_chunk_count,
                bytecode_len: committed.bytecode_commitments.bytecode_len,
                bytecode_T: committed.bytecode_commitments.bytecode_T,
            };
            let program_commitments = TrustedProgramCommitments {
                program_image_commitment: {
                    let chunks = var_alloc.alloc_commitment(
                        &committed.program_commitments.program_image_commitment,
                        "trusted_program_image",
                    );
                    AstCommitment::new(chunks)
                },
                program_image_num_columns: committed.program_commitments.program_image_num_columns,
                program_image_num_words: committed.program_commitments.program_image_num_words,
            };
            ProgramPreprocessing::Committed(CommittedProgramPreprocessing {
                meta: committed.meta.clone(),
                bytecode_commitments,
                program_commitments,
            })
        }
    };

    let symbolic_shared = jolt_core::zkvm::verifier::JoltSharedPreprocessing::<AstCommitmentScheme> {
        program_meta: symbolic_program.meta(),
        program: symbolic_program,
        memory_layout: real_preprocessing.shared.memory_layout.clone(),
        max_padded_trace_length: real_preprocessing.shared.max_padded_trace_length,
        bytecode_chunk_count: real_preprocessing.shared.bytecode_chunk_count,
    };
    JoltVerifierPreprocessing::new(
        symbolic_shared,
        crate::symbolic_traits::ast_commitment_scheme::AstVerifierSetup,
        None,
    )
}
