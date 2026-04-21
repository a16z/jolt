//! Phase 1a: standalone prove→verify harness with honest + adversarial cases.
//!
//! This file is the seam that future Phase 1b/1c work plugs into. It validates
//! the four-step pipeline (`compile → link → prove → verify`) on a minimal
//! protocol with a custom witness, then exercises the adversarial invariant:
//! mutating the committed witness before `prove()` must cause `verify()` to
//! return `Err`.
//!
//! Phase 1a intentionally uses the toy `Σ a·b = 0` zero-check protocol rather
//! than a full FieldReg Twist Module. The goal here is to lock down the
//! honest-accepts / adversarial-rejects harness and validate the data-source
//! plumbing (Polynomials + ProverData) before taking on the hand-rolled
//! two-phase Twist in Phase 1b.

#![allow(unused_results)]

use std::borrow::Cow;
use std::collections::BTreeMap;

use jolt_compiler::builder::ModuleBuilder;
use jolt_compiler::formula::{BindingOrder, Factor, Formula, ProductTerm};
use jolt_compiler::ir::PolyKind;
use jolt_compiler::kernel_spec::{GruenHint, Iteration, LinComboQ};
use jolt_compiler::module::{
    BatchedInstance, BatchedSumcheckDef, ChallengeIdx, ChallengeSource, ClaimFactor, ClaimFormula,
    ClaimTerm, DomainSeparator, Evaluation, InputBinding, InstanceIdx, InstancePhase, KernelDef,
    Module, Op, PointNormalization, ScalarCapture, SegmentedConfig, SumcheckInstance, VerifierOp,
    VerifierStageIndex,
};
use jolt_compiler::module::EvalMode;
use jolt_compiler::{
    compile, CompileParams, Density, Expr, KernelSpec, Objective, Protocol, SolverConfig,
};
use jolt_compute::{link, BufferProvider, LookupTraceData};
use jolt_cpu::CpuBackend;
use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_r1cs::{ConstraintMatrices, R1csKey, R1csSource};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::{
    verify, JoltVerifyingKey, OneHotConfig, ProverConfig, ReadWriteConfig, TRANSCRIPT_LABEL,
};
use jolt_witness::derived::DerivedSource;
use jolt_witness::preprocessed::PreprocessedSource;
use jolt_witness::provider::ProverData;
use jolt_witness::{CycleInput, PolynomialConfig, PolynomialId, Polynomials};
use jolt_zkvm::prove::prove;
use num_traits::{One, Zero};

type MockPCS = MockCommitmentScheme<Fr>;

const LOG_N: usize = 2;
const SIZE: usize = 1 << LOG_N;

/// Toy zero-check protocol: two committed polys `a`, `b`; single sumcheck
/// proving `Σ_x a(x)·b(x) = 0`.
///
/// This is the same shape as `crates/jolt-zkvm/tests/e2e.rs:build_protocol`.
/// Phase 1b will replace this with a hand-rolled `Module` for the two-phase
/// FieldReg Twist.
fn build_protocol() -> Protocol {
    let mut p = Protocol::new();
    let log_n = p.dim("log_n");
    let a = p.poly("a", &[log_n], PolyKind::Committed);
    let b = p.poly("b", &[log_n], PolyKind::Committed);

    let claims = p.sumcheck(
        Expr::from(a) * Expr::from(b),
        Expr::from(0i64),
        &[log_n],
        Density::Dense,
    );
    p.evaluate(a, claims[0]);
    p.evaluate(b, claims[1]);

    p
}

/// Minimal R1CS for routing completeness (the zero-check doesn't exercise it).
fn dummy_r1cs(size: usize) -> (R1csKey<Fr>, Vec<Fr>) {
    let matrices = ConstraintMatrices::new(
        1,
        2,
        vec![vec![(0, Fr::from_u64(1))]],
        vec![vec![(0, Fr::from_u64(1))]],
        vec![vec![(1, Fr::from_u64(1))]],
    );
    let key = R1csKey::new(matrices, size);
    let witness = vec![Fr::zero(); size * key.num_vars_padded];
    (key, witness)
}

/// Common config used by both honest and adversarial tests.
fn prover_config() -> ProverConfig {
    let ram_k = SIZE;
    let ram_log_k = ram_k.trailing_zeros() as usize;
    ProverConfig {
        trace_length: SIZE,
        ram_k,
        bytecode_k: SIZE,
        one_hot_config: OneHotConfig::new(LOG_N),
        rw_config: ReadWriteConfig::new(LOG_N, ram_log_k),
        memory_start: 0x8000_0000,
        memory_end: 0x8001_0000,
        entry_address: 0x8000_0000,
        io_hash: [0u8; 32],
        max_input_size: 0,
        max_output_size: 0,
        heap_size: 0,
        inputs: Vec::new(),
        outputs: Vec::new(),
        panic: false,
        ram_lowest_address: 0x7FFF_0000,
        input_word_offset: 0,
        output_word_offset: 0,
        panic_word_offset: 0,
        termination_word_offset: 0,
    }
}

/// Honest witness: all-PADDING (all-zero dense, no one-hot). Σ a·b = 0
/// trivially. This matches `crates/jolt-zkvm/tests/e2e.rs` — the known-good
/// baseline for the compile/link/prove/verify pipeline.
///
/// Note: RamInc/RdInc are *semantically Jolt-typed* polynomial IDs. Feeding
/// them nontrivial values here trips implicit RAM/register constraints inside
/// the compiled Module (a CommitmentMismatch surfaces during opening).
/// Phase 1b's hand-rolled Module will declare its own FieldReg* IDs without
/// this baggage and feed them arbitrary synthetic witness.
fn honest_cycles() -> [CycleInput; SIZE] {
    [CycleInput::PADDING; SIZE]
}

/// Drive the full pipeline with a caller-supplied Polynomials.
///
/// Returns the proof, the module the executable was linked from (cloned out
/// of the executable so VK construction uses the same Module the prover saw),
/// and the r1cs_key.
fn run_prover(
    polys: &mut Polynomials<Fr>,
) -> (
    jolt_verifier::JoltProof<Fr, MockPCS>,
    Module,
    R1csKey<Fr>,
) {
    let protocol = build_protocol();
    let params = CompileParams {
        dim_sizes: vec![LOG_N as u64],
        field_size_bytes: 32,
        pcs_proof_size: 1,
    };
    let config = SolverConfig {
        proof_size: Objective::Minimize,
        peak_memory: Objective::Ignore,
        prover_time: Objective::Ignore,
    };
    let poly_ids = [PolynomialId::RdInc, PolynomialId::RamInc];
    let module =
        compile(&protocol, &params, &config, &poly_ids).expect("compilation should succeed");

    let backend = CpuBackend;
    let executable = link(module, &backend);

    let (r1cs_key, r1cs_witness) = dummy_r1cs(SIZE);
    let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
    let derived = DerivedSource::new(&r1cs_witness, SIZE, r1cs_key.num_vars_padded);
    let preprocessed = PreprocessedSource::new();
    let mut provider = ProverData::new(polys, r1cs, derived, preprocessed);

    let mut transcript = Blake2bTranscript::<Fr>::new(TRANSCRIPT_LABEL);
    let proof = prove::<_, _, _, MockPCS>(
        &executable,
        &mut provider,
        &backend,
        &(),
        &mut transcript,
        prover_config(),
    );
    (proof, executable.module.clone(), r1cs_key)
}

/// Honest path: witness satisfies Σ a·b = 0 → verify accepts.
#[test]
fn honest_accepts() {
    let poly_config = PolynomialConfig::new(4, 8, 4, 4);
    let mut polys = Polynomials::<Fr>::new(poly_config);
    polys.push(&honest_cycles());
    polys.finish();

    let (proof, module, r1cs_key) = run_prover(&mut polys);
    let vk = JoltVerifyingKey::<Fr, MockPCS>::new(&module, (), r1cs_key);
    verify(&vk, &proof, &[0u8; 32]).expect("honest proof should verify");
}

/// Buffer-override adversarial: mutate the committed witness buffer between
/// `finish()` and `prove()` — the prover commits to the mutated buffer and
/// the pipeline must not silently produce `Ok`.
///
/// Current witness is all-PADDING so the zero-check `Σ a·b = 0` is
/// *algebraically* still satisfied after mutating `RamInc` (since `RdInc = 0`
/// makes every product zero regardless of `b`). The rejection we observe here
/// comes from implicit RAM-specific constraints inside the compiled Module
/// that the nontrivial `RamInc` value violates. This is still a useful
/// harness signal — `Polynomials::insert` DOES reach the prover — but it is
/// not a clean soundness test of the zero-check itself.
///
/// Phase 1b's hand-rolled Module will use FieldReg*-family PolynomialIds with
/// no Jolt-semantic constraints, allowing a direct `Σ a·b ≠ 0` soundness
/// adversarial test.
#[test]
fn adversarial_inc_mutation_rejects() {
    let poly_config = PolynomialConfig::new(4, 8, 4, 4);
    let mut polys = Polynomials::<Fr>::new(poly_config);
    polys.push(&honest_cycles());
    polys.finish();

    polys.insert(
        PolynomialId::RamInc,
        vec![
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::zero(),
            Fr::zero(),
        ],
    );

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (proof, module, r1cs_key) = run_prover(&mut polys);
        let vk = JoltVerifyingKey::<Fr, MockPCS>::new(&module, (), r1cs_key);
        verify(&vk, &proof, &[0u8; 32])
    }));

    match outcome {
        Err(_) | Ok(Err(_)) => {
            // Either panic or verify(Err) — both signal the mutation was detected.
        }
        Ok(Ok(())) => {
            panic!(
                "witness buffer override between finish() and prove() produced \
                 a proof that verified — BufferProvider plumbing is not wired."
            );
        }
    }
}

/// Proof-tampering adversarial: run honest prove → mutate a commitment in the
/// proof → verify must reject. Tests Fiat-Shamir binding of commitments: any
/// post-prove change to a commitment invalidates the challenge derivation and
/// the sumcheck identity closes to a wrong value.
///
/// This is a clean adversarial test — independent of Jolt-semantic constraints
/// on `RdInc`/`RamInc` — and validates that the mock PCS + transcript pipeline
/// are doing their job.
#[test]
fn tampered_commitment_rejects() {
    use jolt_openings::CommitmentScheme;

    let poly_config = PolynomialConfig::new(4, 8, 4, 4);
    let mut polys = Polynomials::<Fr>::new(poly_config);
    polys.push(&honest_cycles());
    polys.finish();

    let (mut proof, module, r1cs_key) = run_prover(&mut polys);

    // Replace the first present commitment with one made from an all-ones
    // poly — a different commitment. Fiat-Shamir challenges derived by the
    // verifier from this tampered commitment diverge from the prover's, so
    // the sumcheck identity fails to close.
    let one_poly = [Fr::from_u64(1); SIZE];
    let (fake_commitment, ()) = MockPCS::commit(&one_poly[..], &());
    let tampered_slot = proof
        .commitments
        .iter_mut()
        .find_map(|c| c.as_mut())
        .expect("at least one commitment should be present in the proof");
    *tampered_slot = fake_commitment;

    let vk = JoltVerifyingKey::<Fr, MockPCS>::new(&module, (), r1cs_key);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        verify(&vk, &proof, &[0u8; 32])
    }));

    match outcome {
        Err(_) | Ok(Err(_)) => {}
        Ok(Ok(())) => {
            panic!(
                "tampered commitment verified — Fiat-Shamir binding is broken \
                 in the standalone pipeline."
            );
        }
    }
}

// ------------------------------------------------------------------
// Phase 1b: custom BufferProvider seam.
//
// `OverrideProvider` wraps `ProverData` and exposes a per-PolynomialId override
// map. When the prover calls `materialize(id)`:
//   * if `id` has an entry in `overrides`, the override buffer is served;
//   * otherwise the call falls through to the inner `ProverData`.
//
// This is the architectural seam Phase 1c will plug its hand-rolled Twist
// witness into. Phase 1c will author a `Module` whose polynomial IDs are
// `FieldReg*` variants and will preload the provider's `overrides` map with
// synthetic Inc/Val/Wa/Ra/ReadValue/WriteValue buffers — no `Polynomials`,
// no `CycleInput`, no RISC-V-shaped data. Phase 1b validates this layer in
// isolation against the same zero-check protocol used by Phase 1a.
// ------------------------------------------------------------------

struct OverrideProvider<'a, F: Field> {
    overrides: BTreeMap<PolynomialId, Vec<F>>,
    fallback: ProverData<'a, F>,
}

impl<'a, F: Field> OverrideProvider<'a, F> {
    fn new(fallback: ProverData<'a, F>) -> Self {
        Self {
            overrides: BTreeMap::new(),
            fallback,
        }
    }

    fn insert(&mut self, id: PolynomialId, buffer: Vec<F>) {
        let _ = self.overrides.insert(id, buffer);
    }
}

impl<F: Field> BufferProvider<F> for OverrideProvider<'_, F> {
    fn materialize(&self, poly_id: PolynomialId) -> Cow<'_, [F]> {
        if let Some(buf) = self.overrides.get(&poly_id) {
            return Cow::Borrowed(buf);
        }
        self.fallback.materialize(poly_id)
    }

    fn release(&mut self, poly_id: PolynomialId) {
        let _ = self.overrides.remove(&poly_id);
        self.fallback.release(poly_id);
    }

    fn lookup_trace(&self) -> Option<&LookupTraceData> {
        self.fallback.lookup_trace()
    }
}

/// Prover entry-point variant that takes a closure to construct the provider.
/// Returns the proof, the linked Module (for VK construction), and the
/// r1cs_key. Parallels `run_prover` but gives the caller control over the
/// BufferProvider wrapping.
fn run_prover_with<F>(
    polys: &mut Polynomials<Fr>,
    wrap: F,
) -> (
    jolt_verifier::JoltProof<Fr, MockPCS>,
    Module,
    R1csKey<Fr>,
)
where
    F: for<'a> FnOnce(ProverData<'a, Fr>) -> OverrideProvider<'a, Fr>,
{
    let protocol = build_protocol();
    let params = CompileParams {
        dim_sizes: vec![LOG_N as u64],
        field_size_bytes: 32,
        pcs_proof_size: 1,
    };
    let config = SolverConfig {
        proof_size: Objective::Minimize,
        peak_memory: Objective::Ignore,
        prover_time: Objective::Ignore,
    };
    let poly_ids = [PolynomialId::RdInc, PolynomialId::RamInc];
    let module =
        compile(&protocol, &params, &config, &poly_ids).expect("compilation should succeed");

    let backend = CpuBackend;
    let executable = link(module, &backend);

    let (r1cs_key, r1cs_witness) = dummy_r1cs(SIZE);
    let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
    let derived = DerivedSource::new(&r1cs_witness, SIZE, r1cs_key.num_vars_padded);
    let preprocessed = PreprocessedSource::new();
    let inner = ProverData::new(polys, r1cs, derived, preprocessed);
    let mut provider = wrap(inner);

    let mut transcript = Blake2bTranscript::<Fr>::new(TRANSCRIPT_LABEL);
    let proof = prove::<_, _, _, MockPCS>(
        &executable,
        &mut provider,
        &backend,
        &(),
        &mut transcript,
        prover_config(),
    );
    (proof, executable.module.clone(), r1cs_key)
}

/// Phase 1b: an `OverrideProvider` with an empty overrides map must be
/// observationally identical to plain `ProverData` — the full pipeline runs
/// and verify accepts. Proves the wrapping layer is transparent.
#[test]
fn override_provider_empty_is_transparent() {
    let poly_config = PolynomialConfig::new(4, 8, 4, 4);
    let mut polys = Polynomials::<Fr>::new(poly_config);
    polys.push(&honest_cycles());
    polys.finish();

    // Closure form (rather than `OverrideProvider::new`) is required so the
    // inner `ProverData`'s lifetime unifies under the HRTB of `run_prover_with`.
    #[expect(clippy::redundant_closure)]
    let (proof, module, r1cs_key) =
        run_prover_with(&mut polys, |inner| OverrideProvider::new(inner));
    let vk = JoltVerifyingKey::<Fr, MockPCS>::new(&module, (), r1cs_key);
    verify(&vk, &proof, &[0u8; 32])
        .expect("empty-overrides pipeline should verify identically to ProverData");
}

/// Phase 1b: the `OverrideProvider` correctly intercepts `materialize` —
/// overriding `RamInc` causes the prover to commit to the override value
/// instead of the `Polynomials`-resident value. The override must reach the
/// prover; verify rejects the resulting proof (same signal as Phase 1a's
/// `adversarial_inc_mutation_rejects`, but routed through the new seam).
#[test]
fn override_provider_intercepts_materialize() {
    let poly_config = PolynomialConfig::new(4, 8, 4, 4);
    let mut polys = Polynomials::<Fr>::new(poly_config);
    polys.push(&honest_cycles());
    polys.finish();

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (proof, module, r1cs_key) = run_prover_with(&mut polys, |inner| {
            let mut p = OverrideProvider::new(inner);
            p.insert(
                PolynomialId::RamInc,
                vec![
                    Fr::from_u64(7),
                    Fr::from_u64(11),
                    Fr::zero(),
                    Fr::zero(),
                ],
            );
            p
        });
        let vk = JoltVerifyingKey::<Fr, MockPCS>::new(&module, (), r1cs_key);
        verify(&vk, &proof, &[0u8; 32])
    }));

    match outcome {
        Err(_) | Ok(Err(_)) => {
            // Override flowed through; the mutated witness was detected.
        }
        Ok(Ok(())) => {
            panic!(
                "OverrideProvider did not intercept materialize — override for \
                 RamInc was ignored by the prover."
            );
        }
    }
}

// ------------------------------------------------------------------
// Phase 1c: hand-rolled 2-phase FieldReg Twist Module.
//
// Standalone prove/verify for a single read/write-checking Twist instance over
// K=16 slots × 2^LOG_T cycles. Structurally a copy of the RAM Twist from
// `crates/jolt-compiler/examples/jolt_core_module.rs` L2383-2486, scoped down
// to just the RW checking sumcheck (no ValEvaluation, no ClaimReduction).
//
// Uses dedicated `PolynomialId::FieldReg*` variants
// (Inc/Val/Ra/ReadValue/WriteValue/EqCycle) so the FR Twist coexists cleanly
// with the RAM Twist under Phase 2a/2b.
// ------------------------------------------------------------------

const LOG_T_TWIST: usize = 4;
const LOG_K_TWIST: usize = 4;
const T_TWIST: usize = 1 << LOG_T_TWIST;
const K_TWIST: usize = 1 << LOG_K_TWIST;

/// Build the standalone FieldReg-style 2-phase Twist Module.
///
/// Returns the Module plus:
///  - `challenges`: key indices the caller needs to reference — the
///    challenge slots for verifier-phase construction (unused externally).
///  - no verifying-side state is needed beyond the Module itself.
fn build_twist_module() -> Module {
    let mut b = ModuleBuilder::new();

    // Plan A wiring: all 5 witness polys Committed. Matches the RAM Twist
    // pattern in `crates/jolt-compiler/examples/jolt_core_module.rs`:
    //   Inc: num_vars=LOG_T, committed_num_vars=LOG_K+LOG_T (zero-padded).
    //     The FinalBind scalar is Inc_MLE(r_cycle) (cycle-only) but the PCS
    //     evaluates the padded commitment at the full point, giving
    //     Π(1-r_addr_i) · Inc_MLE(r_cycle). `Op::ScaleEval` pre-multiplies the
    //     claim by Π(1-r_addr) to reconcile.
    //   Ra, Val: num_vars=LOG_K+LOG_T, no padding. FinalBind scalar equals the
    //     full K×T MLE directly; no ScaleEval needed.
    //   ReadValue, WriteValue: num_vars=LOG_T, opened at rw_cycle (pre-sumcheck
    //     bind point, no padding).
    // Opening point for Inc/Ra/Val: [r_addr_BE ∥ r_cycle_BE] — BE-reversed per
    // segment. Raw LowToHigh round_slots DOES NOT work (RAM Twist precedent).
    let inc = b.add_padded_poly(
        PolynomialId::FieldRegInc,
        "Inc",
        PolyKind::Committed,
        LOG_T_TWIST,
        LOG_K_TWIST + LOG_T_TWIST,
    );
    let ra = b.add_poly(
        PolynomialId::FieldRegRa,
        "Ra",
        PolyKind::Committed,
        LOG_K_TWIST + LOG_T_TWIST,
    );
    let val = b.add_poly(
        PolynomialId::FieldRegVal,
        "Val",
        PolyKind::Committed,
        LOG_K_TWIST + LOG_T_TWIST,
    );
    let read_value = b.add_poly(
        PolynomialId::FieldRegReadValue,
        "ReadValue",
        PolyKind::Committed,
        LOG_T_TWIST,
    );
    let write_value = b.add_poly(
        PolynomialId::FieldRegWriteValue,
        "WriteValue",
        PolyKind::Committed,
        LOG_T_TWIST,
    );
    let eq_cycle = b.add_poly(
        PolynomialId::FieldRegEqCycle,
        "EqCycle",
        PolyKind::Virtual,
        LOG_T_TWIST,
    );

    // -------- Stage 0: commit all 5 polys.
    // Inc at full K×T grid (padded); Ra/Val at native K×T; RV/WV at T.
    b.preamble();
    b.begin_stage();
    b.commit(
        &[inc, ra, val],
        DomainSeparator::Commitment,
        LOG_K_TWIST + LOG_T_TWIST,
    );
    b.commit(&[read_value], DomainSeparator::Commitment, LOG_T_TWIST);
    b.commit(&[write_value], DomainSeparator::Commitment, LOG_T_TWIST);

    // -------- Stage 1: sumcheck --------
    b.begin_stage();

    // γ_rw — batching coefficient between read-check and write-check.
    let ch_gamma = b.squeeze("gamma_rw", ChallengeSource::FiatShamir { after_stage: 0 });

    // Per-cycle eq challenges (log_T of them).
    let rw_cycle_challenges: Vec<ChallengeIdx> = b.squeeze_fiat_shamir("r_cycle", LOG_T_TWIST, 0);

    // Bind ReadValue / WriteValue at rw_cycle to publish their scalar evals to
    // the prover's `evals` table. The AbsorbInputClaim below evaluates the real
    // ReadValue + γ·WriteValue formula against those scalars. The verifier
    // receives the scalars via RecordEvals (from the proof) and independently
    // verifies them against their committed polys via CollectOpeningClaimAt at
    // rw_cycle.
    b.push_op(Op::Materialize {
        binding: InputBinding::Provided { poly: read_value },
    });
    b.push_op(Op::Materialize {
        binding: InputBinding::Provided { poly: write_value },
    });
    b.bind_at_challenges(
        &[read_value, write_value],
        &rw_cycle_challenges,
        BindingOrder::LowToHigh,
    );
    b.evaluate(read_value, EvalMode::FullyBound);
    b.evaluate(write_value, EvalMode::FullyBound);
    b.record_evals(&[read_value, write_value]);
    b.absorb_evals(&[read_value, write_value], DomainSeparator::OpeningClaim);

    // Scalar-capture slots filled at phase 1→2 boundary.
    let ch_eq_bound = b.external_challenge("rw_eq_cycle_bound");
    let ch_inc_bound = b.external_challenge("rw_inc_bound");

    // Real input claim: ReadValue(r_cycle) + γ · WriteValue(r_cycle), read
    // from the evals table populated by the preceding bind+evaluate sequence.
    let input_claim = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Eval(read_value)],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma),
                    ClaimFactor::Eval(write_value),
                ],
            },
        ],
    };

    // Phase 1 kernel — cycle binding, Gruen-hinted.
    // Formula: eq·ra·val + γ·eq·ra·val + γ·eq·ra·inc  (degree 3)
    let phase1 = b.add_kernel(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(2)],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_gamma.0 as u32),
                        Factor::Input(0),
                        Factor::Input(1),
                        Factor::Input(2),
                    ],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_gamma.0 as u32),
                        Factor::Input(0),
                        Factor::Input(1),
                        Factor::Input(3),
                    ],
                },
            ]),
            num_evals: 4, // degree + 1
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: Some(GruenHint {
                eq_input: 0,
                eq_challenges: rw_cycle_challenges.clone(),
                q_lincombo: LinComboQ {
                    a_input: 1,
                    b_input: 2,
                    c_input: 3,
                    gamma_challenge: ch_gamma,
                },
            }),
        },
        inputs: vec![
            InputBinding::EqTable {
                poly: eq_cycle,
                challenges: rw_cycle_challenges.clone(),
            },
            InputBinding::Provided { poly: ra },
            InputBinding::Provided { poly: val },
            InputBinding::Provided { poly: inc },
        ],
        num_rounds: LOG_T_TWIST,
        instance_config: None,
    });

    // Phase 2 kernel — address binding, no Gruen (post-scalar-capture).
    // Formula: ch_eq·ra·val + γ·ch_eq·ra·val + γ·ch_eq·ch_inc·ra  (degree 2)
    let phase2 = b.add_kernel(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_eq_bound.0 as u32),
                        Factor::Input(0),
                        Factor::Input(1),
                    ],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_gamma.0 as u32),
                        Factor::Challenge(ch_eq_bound.0 as u32),
                        Factor::Input(0),
                        Factor::Input(1),
                    ],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_gamma.0 as u32),
                        Factor::Challenge(ch_eq_bound.0 as u32),
                        Factor::Challenge(ch_inc_bound.0 as u32),
                        Factor::Input(0),
                    ],
                },
            ]),
            num_evals: 3, // degree + 1
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::Provided { poly: ra },
            InputBinding::Provided { poly: val },
        ],
        num_rounds: LOG_K_TWIST,
        instance_config: None,
    });

    // Batching coefficient. Even for a single-instance sumcheck we squeeze one
    // FS challenge AFTER AbsorbInputClaim — this aligns the prover's transcript
    // with the verifier's `VerifySumcheck` batched-path (which absorbs the claim
    // and squeezes per-instance coefficients). Without this, the verifier
    // unbatched path skips the absorb and transcripts diverge.
    let ch_batch = b.add_challenge("batch_coeff", ChallengeSource::FiatShamir { after_stage: 0 });

    let batch = b.add_batched_sumcheck(BatchedSumcheckDef {
        instances: vec![BatchedInstance {
            phases: vec![
                InstancePhase {
                    kernel: phase1,
                    num_rounds: LOG_T_TWIST,
                    scalar_captures: vec![],
                    segmented: Some(SegmentedConfig {
                        inner_num_vars: LOG_T_TWIST,
                        outer_num_vars: LOG_K_TWIST,
                        inner_only: vec![true, false, false, true], // eq, ra, val, inc
                        outer_eq_challenges: vec![],
                    }),
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                },
                InstancePhase {
                    kernel: phase2,
                    num_rounds: LOG_K_TWIST,
                    scalar_captures: vec![
                        ScalarCapture {
                            poly: eq_cycle,
                            challenge: ch_eq_bound,
                        },
                        ScalarCapture {
                            poly: inc,
                            challenge: ch_inc_bound,
                        },
                    ],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                },
            ],
            batch_coeff: ch_batch,
            first_active_round: 0,
        }],
        input_claims: vec![input_claim.clone()],
        max_rounds: LOG_T_TWIST + LOG_K_TWIST,
        max_degree: 3,
    });

    b.absorb_input_claim(input_claim.clone(), batch, InstanceIdx(0), 0);
    // Squeeze the batch coefficient AFTER absorb (matches VerifySumcheck batched
    // path: absorbs claim, then squeezes per-instance coefficients).
    b.push_op(Op::Squeeze { challenge: ch_batch });

    let round_slots = b.unrolled_batched_sumcheck_rounds(
        batch,
        LOG_T_TWIST + LOG_K_TWIST,
        4, // num_coeffs = max_degree + 1
        VerifierStageIndex(1),
        "s1_r",
        0,
    );

    // Post-sumcheck: evaluate ra, val, inc at FinalBind, record, absorb.
    let eval_polys = [ra, val, inc];
    b.flush_evals(&eval_polys, DomainSeparator::OpeningClaim);

    // Opening point (matches RAM Twist precedent, jolt_core_module.rs L5997-6002):
    //   opening_point = [r_addr_BE ∥ r_cycle_BE]
    // where r_addr_BE = phase-2 address binds reversed (big-endian)
    //       r_cycle_BE = phase-1 cycle binds reversed (big-endian)
    //
    // In round_slots (LowToHigh raw order): first LOG_T entries are phase-1
    // cycle binds, next LOG_K are phase-2 address binds. Reverse each segment
    // and concatenate with address portion FIRST.
    let r_cycle_slots: Vec<ChallengeIdx> =
        round_slots[..LOG_T_TWIST].iter().rev().copied().collect();
    let r_addr_slots: Vec<ChallengeIdx> = round_slots
        [LOG_T_TWIST..LOG_T_TWIST + LOG_K_TWIST]
        .iter()
        .rev()
        .copied()
        .collect();
    let opening_point: Vec<ChallengeIdx> = r_addr_slots
        .iter()
        .chain(r_cycle_slots.iter())
        .copied()
        .collect();

    // Inc is dense cycle-only, padded to K×T grid. Scale its eval by
    // ∏(1-r_addr_i) so the recorded claim matches the zero-padded MLE that
    // the PCS computes at the full opening_point.
    b.push_op(Op::ScaleEval {
        poly: inc,
        factor_challenges: r_addr_slots.clone(),
    });

    // Opening claims for all 5 committed polys.
    b.push_op(Op::CollectOpeningClaimAt {
        poly: inc,
        point_challenges: opening_point.clone(),
        committed_num_vars: Some(LOG_K_TWIST + LOG_T_TWIST),
    });
    b.push_op(Op::CollectOpeningClaimAt {
        poly: ra,
        point_challenges: opening_point.clone(),
        committed_num_vars: Some(LOG_K_TWIST + LOG_T_TWIST),
    });
    b.push_op(Op::CollectOpeningClaimAt {
        poly: val,
        point_challenges: opening_point.clone(),
        committed_num_vars: Some(LOG_K_TWIST + LOG_T_TWIST),
    });
    b.push_op(Op::CollectOpeningClaimAt {
        poly: read_value,
        point_challenges: rw_cycle_challenges.clone(),
        committed_num_vars: Some(LOG_T_TWIST),
    });
    b.push_op(Op::CollectOpeningClaimAt {
        poly: write_value,
        point_challenges: rw_cycle_challenges.clone(),
        committed_num_vars: Some(LOG_T_TWIST),
    });
    b.push_op(Op::ReduceOpenings);
    b.push_op(Op::Open);

    // -------- Verifier schedule --------
    b.push_verifier_op(VerifierOp::Preamble);
    b.push_verifier_op(VerifierOp::BeginStage);
    // Commit order must mirror the prover: [Inc, Ra, Val] in the K×T barrier,
    // then ReadValue (T), then WriteValue (T).
    b.push_verifier_op(VerifierOp::AbsorbCommitment {
        poly: inc,
        tag: DomainSeparator::Commitment,
    });
    b.push_verifier_op(VerifierOp::AbsorbCommitment {
        poly: ra,
        tag: DomainSeparator::Commitment,
    });
    b.push_verifier_op(VerifierOp::AbsorbCommitment {
        poly: val,
        tag: DomainSeparator::Commitment,
    });
    b.push_verifier_op(VerifierOp::AbsorbCommitment {
        poly: read_value,
        tag: DomainSeparator::Commitment,
    });
    b.push_verifier_op(VerifierOp::AbsorbCommitment {
        poly: write_value,
        tag: DomainSeparator::Commitment,
    });

    // Stage 1: sumcheck.
    b.push_verifier_op(VerifierOp::BeginStage);
    b.push_verifier_op(VerifierOp::Squeeze { challenge: ch_gamma });
    for &c in &rw_cycle_challenges {
        b.push_verifier_op(VerifierOp::Squeeze { challenge: c });
    }
    // Read ReadValue / WriteValue scalars from the proof (prover wrote them via
    // RecordEvals), absorb into transcript to match prover's absorb order.
    let rv_wv_evals = vec![
        Evaluation {
            poly: read_value,
            at_stage: VerifierStageIndex(1),
        },
        Evaluation {
            poly: write_value,
            at_stage: VerifierStageIndex(1),
        },
    ];
    b.push_verifier_op(VerifierOp::RecordEvals {
        evals: rv_wv_evals,
    });
    b.push_verifier_op(VerifierOp::AbsorbEvals {
        polys: vec![read_value, write_value],
        tag: DomainSeparator::OpeningClaim,
    });

    // Build output_check. After normalization with
    // Segments { sizes: [log_T, log_K], output_order: [1, 0] }:
    //   normalized[0..log_K]            = big-endian addresses
    //   normalized[log_K..log_K+log_T]  = big-endian cycle
    // eq_slice for the cycle portion:
    let rw_eq = ClaimFactor::EqEvalSlice {
        challenges: rw_cycle_challenges.clone(),
        at_stage: VerifierStageIndex(1),
        offset: LOG_K_TWIST,
    };
    // StageEval indices match the order polys were recorded: [ra, val, inc].
    let se_ra = ClaimFactor::StageEval(0);
    let se_val = ClaimFactor::StageEval(1);
    let se_inc = ClaimFactor::StageEval(2);
    let output_check = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![rw_eq.clone(), se_ra.clone(), se_val.clone()],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma),
                    rw_eq.clone(),
                    se_ra.clone(),
                    se_val.clone(),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma),
                    rw_eq,
                    se_ra,
                    se_inc,
                ],
            },
        ],
    };

    let instance = SumcheckInstance {
        input_claim,
        output_check,
        num_rounds: LOG_T_TWIST + LOG_K_TWIST,
        degree: 3,
        normalize: Some(PointNormalization::Segments {
            sizes: vec![LOG_T_TWIST, LOG_K_TWIST],
            output_order: vec![1, 0],
        }),
    };

    b.push_verifier_op(VerifierOp::VerifySumcheck {
        instances: vec![instance.clone()],
        stage: 1,
        // Batched path: absorbs claim, squeezes batch coeff. Matches the
        // prover's AbsorbInputClaim + Squeeze sequence.
        batch_challenges: vec![ch_batch],
        claim_tag: Some(DomainSeparator::SumcheckClaim),
        sumcheck_challenge_slots: round_slots.clone(),
    });

    let evaluations: Vec<_> = eval_polys
        .iter()
        .map(|&poly| Evaluation {
            poly,
            at_stage: VerifierStageIndex(1),
        })
        .collect();
    b.push_verifier_op(VerifierOp::RecordEvals { evals: evaluations });
    b.push_verifier_op(VerifierOp::AbsorbEvals {
        polys: eval_polys.to_vec(),
        tag: DomainSeparator::OpeningClaim,
    });
    b.push_verifier_op(VerifierOp::CheckOutput {
        instances: vec![instance],
        stage: 1,
        batch_challenges: vec![ch_batch],
    });

    let _ = round_slots;
    // Mirror the prover's ScaleEval on Inc so verifier's evaluations[inc]
    // matches the prover's post-scale value.
    b.push_verifier_op(VerifierOp::ScaleEval {
        poly: inc,
        factor_challenges: r_addr_slots.clone(),
    });
    b.push_verifier_op(VerifierOp::CollectOpeningClaimAt {
        poly: inc,
        point_challenges: opening_point.clone(),
    });
    b.push_verifier_op(VerifierOp::CollectOpeningClaimAt {
        poly: ra,
        point_challenges: opening_point.clone(),
    });
    b.push_verifier_op(VerifierOp::CollectOpeningClaimAt {
        poly: val,
        point_challenges: opening_point,
    });
    b.push_verifier_op(VerifierOp::CollectOpeningClaimAt {
        poly: read_value,
        point_challenges: rw_cycle_challenges.clone(),
    });
    b.push_verifier_op(VerifierOp::CollectOpeningClaimAt {
        poly: write_value,
        point_challenges: rw_cycle_challenges,
    });
    b.push_verifier_op(VerifierOp::VerifyOpenings);

    b.build()
}

/// Type alias — events live in `jolt_witness::derived` now (Plan C: proper
/// witness infrastructure, not test-harness injection).
use jolt_witness::derived::{FieldRegConfig, FieldRegEvent};

fn honest_twist_events() -> Vec<FieldRegEvent> {
    // One no-op write (0 → 0). Ra has a nonzero entry at (slot 0, cycle 0),
    // Val stays zero everywhere, Inc stays zero at every cycle. The sumcheck
    // sum Σ eq·ra·[(1+γ)·val + γ·inc] = 0 (each term killed by val=0 and inc=0).
    vec![FieldRegEvent {
        cycle: 0,
        slot: 0,
        old: [0; 4],
        new: [0; 4],
    }]
}

fn prover_config_twist() -> ProverConfig {
    ProverConfig {
        trace_length: T_TWIST,
        ram_k: K_TWIST,
        bytecode_k: T_TWIST,
        one_hot_config: OneHotConfig::new(LOG_T_TWIST),
        rw_config: ReadWriteConfig::new(LOG_T_TWIST, LOG_K_TWIST),
        memory_start: 0x8000_0000,
        memory_end: 0x8001_0000,
        entry_address: 0x8000_0000,
        io_hash: [0u8; 32],
        max_input_size: 0,
        max_output_size: 0,
        heap_size: 0,
        inputs: Vec::new(),
        outputs: Vec::new(),
        panic: false,
        ram_lowest_address: 0x7FFF_0000,
        input_word_offset: 0,
        output_word_offset: 0,
        panic_word_offset: 0,
        termination_word_offset: 0,
    }
}

/// Drive the full pipeline using proper `ProverData` + `FieldRegConfig`.
/// Plan C: witness flows through `DerivedSource::with_field_reg()` (the same
/// infrastructure RAM/Registers polys use) — NOT test-harness injection.
/// Adversarial tests wrap the `ProverData` in an `OverrideProvider` to perturb
/// specific polys.
fn run_twist_prover_with<F>(
    events: &[FieldRegEvent],
    wrap: F,
) -> (
    jolt_verifier::JoltProof<Fr, MockPCS>,
    Module,
    R1csKey<Fr>,
)
where
    F: for<'a> FnOnce(ProverData<'a, Fr>) -> OverrideProvider<'a, Fr>,
{
    let module = build_twist_module();
    let backend = CpuBackend;
    let executable = link(module, &backend);

    let field_reg_config = FieldRegConfig {
        k: K_TWIST,
        initial_state: vec![[0u64; 4]; K_TWIST],
        events: events.to_vec(),
    };
    // ReadValue / WriteValue are `PolySource::Witness` — they must be inserted
    // into `Polynomials` directly (the same pattern `SpartanWitness` /
    // `TrustedAdvice` follow). `FieldRegConfig` provides the canonical
    // computation so there's a single source of truth.
    let rv = field_reg_config.compute_read_value::<Fr>(T_TWIST);
    let wv = field_reg_config.compute_write_value::<Fr>(T_TWIST);

    let mut polys = Polynomials::<Fr>::new(PolynomialConfig::new(4, 8, 4, 4));
    polys.push(&[CycleInput::PADDING; T_TWIST]);
    polys.finish();
    let _ = polys.insert(PolynomialId::FieldRegReadValue, rv);
    let _ = polys.insert(PolynomialId::FieldRegWriteValue, wv);

    let (r1cs_key, r1cs_witness) = dummy_r1cs(T_TWIST);
    let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
    let derived = DerivedSource::new(&r1cs_witness, T_TWIST, r1cs_key.num_vars_padded)
        .with_field_reg(field_reg_config);
    let preprocessed = PreprocessedSource::new();
    let inner = ProverData::new(&mut polys, r1cs, derived, preprocessed);
    let mut provider = wrap(inner);

    let mut transcript = Blake2bTranscript::<Fr>::new(TRANSCRIPT_LABEL);
    let proof = prove::<_, _, _, MockPCS>(
        &executable,
        &mut provider,
        &backend,
        &(),
        &mut transcript,
        prover_config_twist(),
    );
    (proof, executable.module.clone(), r1cs_key)
}

fn twist_vk(module: &Module) -> (JoltVerifyingKey<Fr, MockPCS>, R1csKey<Fr>) {
    let (r1cs_key, _w) = dummy_r1cs(T_TWIST);
    let r1cs_key_for_vk = r1cs_key.clone();
    (
        JoltVerifyingKey::<Fr, MockPCS>::new(module, (), r1cs_key),
        r1cs_key_for_vk,
    )
}

#[test]
fn twist_honest_accepts() {
    let events = honest_twist_events();
    // Closure (not `OverrideProvider::new` directly) is required for HRTB
    // inference on `run_twist_prover_with`'s FnOnce bound.
    #[allow(clippy::redundant_closure)]
    let (proof, module, _r1cs_key) =
        run_twist_prover_with(&events, |inner| OverrideProvider::new(inner));
    let (vk, _) = twist_vk(&module);
    verify(&vk, &proof, &[0u8; 32]).expect("honest FieldReg Twist proof should verify");
}

/// Phase 1e adversarial: perturb `ReadValue` (a committed poly). The prover's
/// input_claim scalar becomes `perturbed_RV + γ·honest_WV` ≠ 0, but the actual
/// Σ over the honest inner witness (all-zero ra/val/inc) is 0. Sumcheck closure
/// fails at round 0 → verify rejects.
///
/// Unlike Phase 1d's Ra/Val/Inc perturbations (which were vacuous — those
/// polys are uncommitted, so a malicious prover is free to choose any
/// consistent stage_evals), `ReadValue` IS committed, so its value is fixed at
/// commit time. The input_claim derived from it CANNOT be forged.
#[test]
fn twist_adversarial_read_value_mutation_rejects() {
    twist_adversarial_rejects(|provider| {
        let mut tampered = provider
            .fallback
            .materialize(PolynomialId::FieldRegReadValue)
            .into_owned();
        tampered[0] += Fr::one();
        provider.insert(PolynomialId::FieldRegReadValue, tampered);
    });
}

/// Phase 1e adversarial: perturb `WriteValue`. Same story — committed, so the
/// perturbation flows into input_claim and breaks sumcheck closure.
#[test]
fn twist_adversarial_write_value_mutation_rejects() {
    twist_adversarial_rejects(|provider| {
        let mut tampered = provider
            .fallback
            .materialize(PolynomialId::FieldRegWriteValue)
            .into_owned();
        tampered[0] += Fr::one();
        provider.insert(PolynomialId::FieldRegWriteValue, tampered);
    });
}

/// Full-commit adversarial: perturb committed `Inc`. With all 5 polys now
/// committed, the commitment + opening proof binds Inc's values. The
/// OverrideProvider intercepts materialize calls and returns a tampered
/// buffer — prover commits to the tampered data, verifier rejects.
#[test]
fn twist_adversarial_inc_commit_mutation_rejects() {
    twist_adversarial_rejects(|provider| {
        let mut tampered = provider
            .fallback
            .materialize(PolynomialId::FieldRegInc)
            .into_owned();
        tampered[0] += Fr::one();
        provider.insert(PolynomialId::FieldRegInc, tampered);
    });
}

/// Full-commit adversarial: perturb committed `Val`.
#[test]
fn twist_adversarial_val_commit_mutation_rejects() {
    twist_adversarial_rejects(|provider| {
        let mut tampered = provider
            .fallback
            .materialize(PolynomialId::FieldRegVal)
            .into_owned();
        tampered[0] += Fr::one();
        provider.insert(PolynomialId::FieldRegVal, tampered);
    });
}

/// Full-commit adversarial: paired Ra + Val mutation. A lone Ra mutation at a
/// position where val/inc are zero doesn't break Σ eq·ra·[(1+γ)·val + γ·inc]
/// — the sum stays 0, the prover is self-consistent, and the PCS opening
/// succeeds. Pairing with a Val mutation at the SAME (addr, cycle) makes
/// ra·val nonzero there, breaking the sumcheck identity.
#[test]
fn twist_adversarial_ra_val_paired_mutation_rejects() {
    twist_adversarial_rejects(|provider| {
        let t = T_TWIST;
        let mut ra = provider
            .fallback
            .materialize(PolynomialId::FieldRegRa)
            .into_owned();
        let mut val = provider
            .fallback
            .materialize(PolynomialId::FieldRegVal)
            .into_owned();
        // Inject (addr=1, cycle=0) with ra=1, val=1.
        ra[t] = Fr::one();
        val[t] = Fr::one();
        provider.insert(PolynomialId::FieldRegRa, ra);
        provider.insert(PolynomialId::FieldRegVal, val);
    });
}

/// Shared body: honest witness + `OverrideProvider` with adversarial mutations
/// injected via `tamper`. Either prover panic or `verify(Err)` is a valid
/// soundness signal; silent `Ok` is a test failure.
fn twist_adversarial_rejects<F>(tamper: F)
where
    F: for<'a> FnOnce(&mut OverrideProvider<'a, Fr>),
{
    let events = honest_twist_events();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (proof, module, _r1cs_key) = run_twist_prover_with(&events, |inner| {
            let mut p = OverrideProvider::new(inner);
            tamper(&mut p);
            p
        });
        let (vk, _) = twist_vk(&module);
        verify(&vk, &proof, &[0u8; 32])
    }));

    match outcome {
        Err(_) | Ok(Err(_)) => {}
        Ok(Ok(())) => {
            panic!(
                "adversarial mutation verified — the 2-phase Twist is not \
                 binding the committed witness."
            );
        }
    }
}
