//! Cross-system proof conversion.
//!
//! `modular_to_core` substitutes a modular prover's outputs (round
//! polynomials, commitments, opening proof) into a jolt-core proof
//! scaffold. The remaining structural fields (uniskip first-round
//! proofs, opening_claims, dory_layout, etc.) come from a reference
//! jolt-core proof produced from the same workload — they are
//! identical under transcript parity, so substitution is sound for
//! the cross-verifier soundness fixture.
//!
//! Note: this conversion is intentionally lossy on the modular eval
//! list. Modular today emits `Op::RecordEvals` only in stages 1–2;
//! stages 3–7 carry empty `evals` until the verifier-parity work
//! lands. Tampers against eval slots in unwired stages return
//! `TamperOutcome::Vacuous` from `apply_tamper`.

#![allow(non_snake_case)]

use ark_bn254::Fr as ArkFr;
use jolt_core::curve::Bn254Curve;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryLayout};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::subprotocols::univariate_skip::UniSkipFirstRoundProofVariant;
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::config::{OneHotConfig as CoreOneHotConfig, ReadWriteConfig as CoreRwConfig};
use jolt_core::zkvm::proof_serialization::{Claims, JoltProof as CoreJoltProof};
use jolt_dory::DoryScheme;
use jolt_field::Fr as NewFr;
use jolt_poly::UnivariatePoly;
use jolt_verifier::JoltProof as ModularJoltProof;

type CoreCommitment = <DoryCommitmentScheme as CommitmentScheme>::Commitment;

/// Reference scaffolding fields the modular proof does not produce
/// directly: structural pieces of a jolt-core proof that are identical
/// across both pipelines under transcript parity.
pub struct CoreScaffold {
    pub stage1_uni_skip_first_round_proof:
        UniSkipFirstRoundProofVariant<ArkFr, Bn254Curve, Blake2bTranscript>,
    pub stage2_uni_skip_first_round_proof:
        UniSkipFirstRoundProofVariant<ArkFr, Bn254Curve, Blake2bTranscript>,
    pub untrusted_advice_commitment: Option<CoreCommitment>,
    pub opening_claims: Claims<ArkFr>,
    pub trace_length: usize,
    pub ram_K: usize,
    pub rw_config: CoreRwConfig,
    pub one_hot_config: CoreOneHotConfig,
    pub dory_layout: DoryLayout,
    /// Number of commitments the core proof carries (filtered count
    /// from modular's `Vec<Option<PCS::Output>>`). Cached from a
    /// reference core proof.
    pub num_commitments: usize,
}

impl CoreScaffold {
    /// Borrow the relevant scaffolding fields from a reference core
    /// proof. The fields are cloned so the source proof can outlive the
    /// scaffold (or vice versa).
    pub fn from_core_proof(
        proof: &CoreJoltProof<ArkFr, Bn254Curve, DoryCommitmentScheme, Blake2bTranscript>,
    ) -> Self {
        Self {
            stage1_uni_skip_first_round_proof: proof.stage1_uni_skip_first_round_proof.clone(),
            stage2_uni_skip_first_round_proof: proof.stage2_uni_skip_first_round_proof.clone(),
            untrusted_advice_commitment: proof.untrusted_advice_commitment,
            opening_claims: clone_claims(&proof.opening_claims),
            trace_length: proof.trace_length,
            ram_K: proof.ram_K,
            rw_config: proof.rw_config.clone(),
            one_hot_config: proof.one_hot_config.clone(),
            dory_layout: proof.dory_layout,
            num_commitments: proof.commitments.len(),
        }
    }
}

fn clone_claims(claims: &Claims<ArkFr>) -> Claims<ArkFr> {
    Claims(claims.0.clone())
}

/// Convert `NewFr` → `ArkFr`. Both are the BN254 scalar field; this is
/// a representation-only cast.
pub fn to_ark(f: NewFr) -> ArkFr {
    f.into()
}

/// Convert a modular `UnivariatePoly<NewFr>` into a jolt-core
/// `CompressedUniPoly<ArkFr>`. `CompressedUniPoly` stores `[c0, c2, c3,
/// ...]` (linear term `c1` omitted; verifier reconstructs from `s(0) + s(1)`).
pub fn to_compressed_uni_poly(
    poly: &UnivariatePoly<NewFr>,
) -> jolt_core::poly::unipoly::CompressedUniPoly<ArkFr> {
    let coeffs = poly.coefficients();
    assert!(
        coeffs.len() >= 2,
        "round poly must have at least 2 coefficients"
    );
    let mut compressed = Vec::with_capacity(coeffs.len() - 1);
    compressed.push(to_ark(coeffs[0]));
    for c in &coeffs[2..] {
        compressed.push(to_ark(*c));
    }
    jolt_core::poly::unipoly::CompressedUniPoly {
        coeffs_except_linear_term: compressed,
    }
}

/// Convert a sequence of modular round polynomials into a jolt-core
/// `SumcheckInstanceProof::Clear` (non-zk).
pub fn to_core_sumcheck_proof(
    round_polys: &[UnivariatePoly<NewFr>],
) -> SumcheckInstanceProof<ArkFr, Bn254Curve, Blake2bTranscript> {
    let compressed: Vec<_> = round_polys.iter().map(to_compressed_uni_poly).collect();
    SumcheckInstanceProof::Clear(jolt_core::subprotocols::sumcheck::ClearSumcheckProof::new(
        compressed,
    ))
}

/// Convert a jolt-dory `DoryCommitment` to jolt-core's `ArkGT` shape.
///
/// Both are repr(transparent) wrappers over the same `Fq12` type.
pub fn commitment_to_ark(c: &jolt_dory::types::DoryCommitment) -> CoreCommitment {
    // SAFETY: Bn254GT and ArkGT are both repr(transparent) over Fq12.
    unsafe { std::mem::transmute_copy(&c.0) }
}

/// Substitute a modular proof's prover-produced fields into a jolt-core
/// proof scaffold. Stages 1-2 strip the first round polynomial (the
/// uniskip is represented separately on the core side); stages 3-7 use
/// the full round polynomial sequence.
pub fn modular_to_core(
    modular: &ModularJoltProof<NewFr, DoryScheme>,
    scaffold: &CoreScaffold,
) -> CoreJoltProof<ArkFr, Bn254Curve, DoryCommitmentScheme, Blake2bTranscript> {
    assert_eq!(
        modular.stage_proofs.len(),
        8,
        "expected 8 stage proofs (7 sumcheck + 1 PCS opening) — got {}",
        modular.stage_proofs.len()
    );
    let stage1_sc =
        to_core_sumcheck_proof(&modular.stage_proofs[0].round_polys.round_polynomials[1..]);
    let stage2_sc =
        to_core_sumcheck_proof(&modular.stage_proofs[1].round_polys.round_polynomials[1..]);
    let stage3_sc = to_core_sumcheck_proof(&modular.stage_proofs[2].round_polys.round_polynomials);
    let stage4_sc = to_core_sumcheck_proof(&modular.stage_proofs[3].round_polys.round_polynomials);
    let stage5_sc = to_core_sumcheck_proof(&modular.stage_proofs[4].round_polys.round_polynomials);
    let stage6_sc = to_core_sumcheck_proof(&modular.stage_proofs[5].round_polys.round_polynomials);
    let stage7_sc = to_core_sumcheck_proof(&modular.stage_proofs[6].round_polys.round_polynomials);

    // Filter out None slots (prover skipped them for all-zero advice);
    // truncate to the core proof's commitment count (excludes advice
    // commitments which jolt-core handles via separate fields).
    let commitments: Vec<_> = modular
        .commitments
        .iter()
        .filter_map(|c| c.as_ref())
        .take(scaffold.num_commitments)
        .map(commitment_to_ark)
        .collect();

    assert_eq!(
        modular.opening_proofs.len(),
        1,
        "expected exactly 1 joint opening proof"
    );
    let joint_opening_proof = modular.opening_proofs[0].0.clone();

    CoreJoltProof {
        commitments,
        stage1_sumcheck_proof: stage1_sc,
        stage2_sumcheck_proof: stage2_sc,
        stage3_sumcheck_proof: stage3_sc,
        stage4_sumcheck_proof: stage4_sc,
        stage5_sumcheck_proof: stage5_sc,
        stage6_sumcheck_proof: stage6_sc,
        stage7_sumcheck_proof: stage7_sc,
        joint_opening_proof,
        stage1_uni_skip_first_round_proof: scaffold.stage1_uni_skip_first_round_proof.clone(),
        stage2_uni_skip_first_round_proof: scaffold.stage2_uni_skip_first_round_proof.clone(),
        untrusted_advice_commitment: scaffold.untrusted_advice_commitment,
        opening_claims: clone_claims(&scaffold.opening_claims),
        trace_length: scaffold.trace_length,
        ram_K: scaffold.ram_K,
        rw_config: scaffold.rw_config.clone(),
        one_hot_config: scaffold.one_hot_config.clone(),
        dory_layout: scaffold.dory_layout,
    }
}
