//! Symbolic implementation of jolt-core's spongefish verifier surface
//! (`FsChallenge` / `FsAbsorb` / `VerifierFs`) for `F = MleAst`.
//!
//! Mirrors the **field-aligned** Poseidon transcript (spec §4, T-O5): the
//! native `PoseidonSponge` is a `U = Fr` compression chain and every absorb is
//! a tagged unit message (see `crates/jolt-transcript/src/codec.rs`), so the
//! symbolic sponge absorbs witness variables as native units and a challenge
//! is ONE `TranscriptHash` node — proof scalars are never byte-decomposed
//! in-circuit.
//!
//! Structure:
//!
//! 1. **Challenges return directly.** `FsChallenge`'s methods return
//!    `F = MleAst`; a squeeze is a single `MleAst::poseidon(state, 0, 0)` node.
//! 2. **Frame reads are typed.** `read_scalars`/`read_commitments` pop the next
//!    pre-parsed NARG frame (`narg_parser::ParsedNarg`), symbolize the real
//!    proof bytes into fresh witness variables via the `set_read_symbolizer`
//!    hook, and absorb them through the matching typed layout hook — exactly
//!    the kinds jolt-core's `VerifierFs` reads on the non-ZK stages-1–7 path.
//! 3. **The sponge layout is a test seam.** [`SymbolicSpongeLayout`] hides how
//!    absorbs/squeezes update the in-circuit sponge; [`FieldAlignedLayout`] is
//!    the only production implementation, gated by the differential tests
//!    below plus the `poseidon_model` oracle. The trait exists so recording
//!    test doubles can observe absorb routing (C7) and so `FAITHFUL` acts as
//!    a faithfulness tripwire (A1 deployment-blocker guard).

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::field::JoltField;
use jolt_core::transcript_msgs::{FsAbsorb, FsChallenge, VerifierFs};
use jolt_transcript::{
    poseidon_domain_separator_msgs, push_byte_rule_units, push_commitments_frame_header,
    push_field_frame_units, VerificationError, VerificationResult,
};
use zklean_extractor::mle_ast::{
    clear_read_symbolizer, set_read_symbolizer, take_pending_append,
    take_pending_commitment_chunks, MleAst,
};
use zklean_extractor::{COMMITMENT_BYTES, COMMITMENT_CHUNKS};

use crate::narg_parser::ParsedNarg;
use crate::symbolic_proof::VarAllocator;

/// How one absorb/squeeze updates the symbolic sponge, in the typed message
/// vocabulary of spec §4.2. [`FieldAlignedLayout`] is the only production
/// implementation; the trait is a test seam that lets recording test doubles
/// observe absorb routing (the C7 absorb-routing regression test) and ties the
/// pipeline's faithfulness flag to a compile-time constant.
pub trait SymbolicSpongeLayout {
    /// `true` iff this layout reproduces the native `PoseidonSponge`
    /// value-exactly, so the circuit's Fiat-Shamir challenges match the real
    /// verifier's. Drives the pipeline's `sponge_faithful` flag (A1
    /// deployment-blocker guard); a non-faithful test layout must say `false`.
    const FAITHFUL: bool;

    /// One `absorb_scalar` — the count-led single-element field frame
    /// `[Fr(3), value]` (`FieldFrameMsg` of one element).
    fn absorb_scalar_message(&mut self, value: &MleAst);

    /// One field frame of `k` elements — `[Fr(2k+1), e₁, …, e_k]`
    /// (`FieldFrameMsg`; `write_scalars`/`read_scalars`, `absorb_scalars`).
    fn absorb_field_frame(&mut self, elements: &[MleAst]);

    /// One commitments frame (`CommitmentsMsg`): the frame count unit
    /// `Fr(2k+1)` zero-padded to a whole permute pair, then one 14-unit
    /// byte-rule group per commitment — `[Fr(2·384), 13 chunks]`, the chunk
    /// units being the witness variables directly. An empty frame is the
    /// count-led `[Fr(1), 0]` pair.
    fn absorb_commitments_frame(&mut self, groups: &[Vec<MleAst>]);

    /// One lone commitment absorb (`absorb_commitment`, e.g. trusted
    /// commitments): the byte rule over ONE canonical serialization —
    /// `[Fr(2·384), 13 chunks]` with NO frame count unit, mirroring
    /// `messages.rs`'s Poseidon `absorb_commitment` (= `RawBytesMsg` over the
    /// serialization).
    fn absorb_commitment_message(&mut self, chunks: &[MleAst]);

    /// One byte-rule message over concrete bytes both sides know
    /// (`RawBytesMsg`: `[Fr(2L), ceil(L/31) 31-byte-LE chunks]`), hashed into
    /// the circuit as constants.
    fn absorb_byte_message(&mut self, bytes: &[u8]);

    /// Squeeze one native-unit field challenge: ONE permute, challenge = the
    /// new state.
    fn squeeze(&mut self) -> MleAst;
}

// The byte-rule chunk width exists in two crates (`zklean_extractor::BYTES_PER_CHUNK`
// drives the commitment re-chunking, `jolt_transcript::BYTE_RULE_CHUNK` drives the
// native sponge encoding). They MUST be the same 31 bytes or the symbolic chunk
// witness values silently diverge from the units the native sponge absorbs — tie the
// split brains together at compile time.
const _: () = assert!(
    zklean_extractor::BYTES_PER_CHUNK == jolt_transcript::BYTE_RULE_CHUNK,
    "zklean_extractor::BYTES_PER_CHUNK != jolt_transcript::BYTE_RULE_CHUNK: the \
     commitment re-chunking no longer matches the native byte-rule encoding"
);

/// An `Fr` constant as an `MleAst` scalar node (Montgomery-free canonical
/// limbs).
fn fr_const(value: Fr) -> MleAst {
    MleAst::from(value.into_bigint().0)
}

/// PRODUCTION sponge layout (spec §4.5): mirrors the field-aligned
/// `PoseidonSponge` (`U = Fr` compression chain) over `MleAst::poseidon`
/// nodes, absorbing exactly the codec's tagged unit streams. Every constant
/// unit (message tags, frame counts, even-padding) is produced by the
/// IMPORTED `jolt_transcript` encoders ([`push_byte_rule_units`],
/// [`push_field_frame_units`], [`push_commitments_frame_header`]) and lifted
/// via [`fr_const`]; only the payload positions are symbolic values.
pub struct FieldAlignedLayout {
    state: MleAst,
}

impl FieldAlignedLayout {
    /// Seed with the spongefish domain separator: the same three byte strings
    /// the native factories absorb ([`poseidon_domain_separator_msgs`]:
    /// `PROTOCOL_ID` ‖ session-`BytesMsg` ‖ 32-byte instance digest), each
    /// under the byte rule, all as baked constants — the digest value is baked
    /// into the circuit like every other native tag.
    pub fn new(session: &[u8], instance: &[u8; 32]) -> Self {
        let mut layout = Self {
            state: MleAst::from_u64(0),
        };
        for msg in poseidon_domain_separator_msgs(session, *instance) {
            layout.absorb_byte_message(&msg.0);
        }
        layout
    }

    /// One sponge `absorb(units)`: unit pairs through the permutation,
    /// zero-padding an odd tail — `state' = poseidon(state, a, b)`. Every
    /// message the layout absorbs is a complete tagged group (already padded
    /// to whole pairs by the codec rules), so message boundaries bind.
    fn absorb_units(&mut self, units: &[MleAst]) {
        let zero = MleAst::from_u64(0);
        for pair in units.chunks(2) {
            let a = pair[0];
            let b = pair.get(1).copied().unwrap_or(zero);
            self.state = MleAst::poseidon(&self.state, &a, &b);
        }
    }

    /// The 14-unit per-commitment byte-rule group `[Fr(2·384), 13 chunks]`
    /// (one Dory GT = 384 canonical bytes ↦ 12×31B + 1×12B chunks). The tag
    /// is a constant because the schedule fixes every commitment's byte
    /// length; the chunk units are the witness variables directly. The tag is
    /// taken from the IMPORTED [`push_byte_rule_units`] over a 384-byte
    /// placeholder, so the encoding cannot drift from the native sponge's.
    fn push_commitment_group(units: &mut Vec<MleAst>, chunks: &[MleAst]) {
        assert_eq!(
            chunks.len(),
            COMMITMENT_CHUNKS,
            "commitment group must be the {COMMITMENT_CHUNKS}-chunk Dory GT re-chunking"
        );
        let mut fr_units: Vec<Fr> = Vec::with_capacity(1 + COMMITMENT_CHUNKS);
        push_byte_rule_units(&mut fr_units, &[0u8; COMMITMENT_BYTES]);
        // 1 tag + 13 chunks = 14 units (even), so the native encoder adds no pad;
        // every unit after the tag is payload (replaced by the witness chunk vars).
        assert_eq!(
            fr_units.len(),
            1 + COMMITMENT_CHUNKS,
            "native byte rule no longer encodes a Dory GT as [tag, {COMMITMENT_CHUNKS} chunks]"
        );
        units.push(fr_const(fr_units[0]));
        units.extend_from_slice(chunks);
    }
}

impl SymbolicSpongeLayout for FieldAlignedLayout {
    // Value-faithful to the native field-aligned `PoseidonSponge`: proven by
    // the differential gates — `poseidon_model` vs a real
    // `ProverState<PoseidonSponge>` (level 1), and
    // `field_aligned_layout_matches_native_sponge` below, which evaluates this
    // layout's challenge ASTs against a witness and asserts equality with the
    // model (level 2).
    const FAITHFUL: bool = true;

    fn absorb_scalar_message(&mut self, value: &MleAst) {
        self.absorb_field_frame(std::slice::from_ref(value));
    }

    fn absorb_field_frame(&mut self, elements: &[MleAst]) {
        // Encode a zero-placeholder frame of the same arity through the IMPORTED
        // native encoder, then splice the symbolic payload into the element
        // positions — the count tag and even-pad units (positions 0 and, when
        // present, the tail) come from `jolt_transcript` verbatim.
        let placeholder = vec![Fr::from(0u64); elements.len()];
        let mut fr_units: Vec<Fr> = Vec::with_capacity(2 + elements.len());
        push_field_frame_units(&mut fr_units, &placeholder);
        let units: Vec<MleAst> = fr_units
            .iter()
            .enumerate()
            .map(|(i, unit)| match i.checked_sub(1) {
                Some(k) if k < elements.len() => elements[k],
                _ => fr_const(*unit),
            })
            .collect();
        self.absorb_units(&units);
    }

    fn absorb_commitments_frame(&mut self, groups: &[Vec<MleAst>]) {
        let mut units = Vec::with_capacity(2 + (1 + COMMITMENT_CHUNKS) * groups.len());
        // Frame count unit padded to a whole permute pair (review fix F1), so
        // the per-GT groups stay pair-aligned — header units from the IMPORTED
        // native encoder.
        let mut header: Vec<Fr> = Vec::with_capacity(2);
        push_commitments_frame_header(&mut header, groups.len());
        units.extend(header.into_iter().map(fr_const));
        for chunks in groups {
            Self::push_commitment_group(&mut units, chunks);
        }
        self.absorb_units(&units);
    }

    fn absorb_commitment_message(&mut self, chunks: &[MleAst]) {
        let mut units = Vec::with_capacity(1 + COMMITMENT_CHUNKS);
        Self::push_commitment_group(&mut units, chunks);
        self.absorb_units(&units);
    }

    fn absorb_byte_message(&mut self, bytes: &[u8]) {
        let mut fr_units = Vec::with_capacity(2 + bytes.len() / 31);
        push_byte_rule_units(&mut fr_units, bytes);
        let units: Vec<MleAst> = fr_units.into_iter().map(fr_const).collect();
        self.absorb_units(&units);
    }

    fn squeeze(&mut self) -> MleAst {
        let zero = MleAst::from_u64(0);
        self.state = MleAst::poseidon(&self.state, &zero, &zero);
        self.state
    }
}

/// Witness-variable naming context, set by the transpiler driver between stages so
/// names follow the frozen Era-2 contract (spec §13.6): `commitment_{c}_{chunk}`,
/// `stage{n}_uni_skip_coeff_{i}`, `stage{n}_sumcheck_r{round}_{i}`.
#[derive(Clone, Debug)]
pub enum FrameLabel {
    /// Pre-stage frames: frame 0 = witness commitments, frame 1 = advice presence.
    /// Commitment elements are the 13 byte-rule chunks per Dory GT (spec §4.5
    /// re-chunking: 12×31B + 1×12B).
    Prestage,
    /// A stage that begins with a uni-skip first-round frame (stages 1–2):
    /// frame 0 → `stage{n}_uni_skip_coeff_{i}`, frame k → `stage{n}_sumcheck_r{k-1}_{i}`.
    StageWithUniskip(u8),
    /// Every frame is a sumcheck round: `{prefix}_r{frame}_{i}` (e.g.
    /// `Rounds("stage6a_sumcheck")`).
    Rounds(String),
}

impl FrameLabel {
    fn element_name(&self, frame_in_label: usize, element: usize) -> String {
        match self {
            Self::Prestage => match frame_in_label {
                0 => format!(
                    "commitment_{}_{}",
                    element / COMMITMENT_CHUNKS,
                    element % COMMITMENT_CHUNKS
                ),
                1 => {
                    // The advice presence frame carries AT MOST one commitment
                    // (`read_commitment_frames` rejects len > 1), so element ==
                    // chunk index. Assert rather than `% COMMITMENT_CHUNKS`: the
                    // chunk-only name is the frozen Go contract, and silently
                    // wrapping a second commitment onto the same names would
                    // corrupt the witness map.
                    assert!(
                        element < COMMITMENT_CHUNKS,
                        "untrusted-advice presence frame symbolized element {element} — more \
                         than one commitment in a frame the verifier caps at one"
                    );
                    format!("untrusted_advice_commitment_{element}")
                }
                k => format!("prestage_f{k}_{element}"),
            },
            Self::StageWithUniskip(n) => {
                if frame_in_label == 0 {
                    format!("stage{n}_uni_skip_coeff_{element}")
                } else {
                    format!("stage{n}_sumcheck_r{}_{element}", frame_in_label - 1)
                }
            }
            Self::Rounds(prefix) => format!("{prefix}_r{frame_in_label}_{element}"),
        }
    }
}

/// Symbolic verifier transcript: replays pre-parsed NARG frames as witness
/// variables, records absorbs/squeezes through the sponge layout, and returns
/// challenges as AST nodes. Drop-in for `&mut impl VerifierFs<MleAst>` in
/// `TranspilableVerifier`.
pub struct SymbolicVerifierFs<L: SymbolicSpongeLayout> {
    layout: L,
    frames: VecDeque<Vec<u8>>,
    var_alloc: Rc<RefCell<VarAllocator>>,
    label: FrameLabel,
    frames_read_in_label: usize,
    /// Squeezed challenge nodes, in order. Consumed by the in-CI real-proof
    /// challenge differential (`symbolic_pipeline_runs_on_real_muldiv_poseidon_proof`,
    /// `transpiler/tests/symbolic_pipeline.rs`): the pipeline surfaces this record
    /// through `PipelineOutput::squeezed_challenges`, the test evaluates each AST
    /// against the recorded witness and asserts element-wise equality with the
    /// challenges the NATIVE `TranspilableVerifier` + real Poseidon transcript
    /// squeeze on the same proof — closing the dispatch-layer schedule gap the
    /// layout differential (`FieldAlignedLayout` vs the `poseidon_model` oracle)
    /// cannot see.
    pub squeezed_challenges: Vec<MleAst>,
}

impl<L: SymbolicSpongeLayout> SymbolicVerifierFs<L> {
    pub fn new(layout: L, parsed: ParsedNarg, var_alloc: Rc<RefCell<VarAllocator>>) -> Self {
        Self {
            layout,
            frames: parsed.into_frames().into(),
            var_alloc,
            label: FrameLabel::Prestage,
            frames_read_in_label: 0,
            squeezed_challenges: Vec::new(),
        }
    }

    /// Set the naming context for subsequent frame reads (driver calls this
    /// between stages).
    pub fn set_label(&mut self, label: FrameLabel) {
        self.label = label;
        self.frames_read_in_label = 0;
    }

    /// Frames not yet consumed — must be 0 after stage 7 (the offline analogue of
    /// `check_eof`; in non-ZK mode stage 8 reads no frames).
    pub fn remaining_frames(&self) -> usize {
        self.frames.len()
    }

    /// Pop the next NARG frame and decode it with the read symbolizer
    /// installed: each element the decode consumes becomes a fresh named
    /// witness variable carrying its concrete value (the shared `nodes`
    /// collector returns them in read order — 1 per scalar, 13 chunk vars per
    /// commitment).
    fn pop_symbolized_frame<T: CanonicalDeserialize>(
        &mut self,
    ) -> VerificationResult<(Vec<T>, Vec<MleAst>)> {
        let frame = self.frames.pop_front().ok_or(VerificationError)?;
        let frame_in_label = self.frames_read_in_label;
        self.frames_read_in_label += 1;

        let label = self.label.clone();
        let alloc = Rc::clone(&self.var_alloc);
        let element_counter = Rc::new(Cell::new(0usize));
        let nodes: Rc<RefCell<Vec<MleAst>>> = Rc::new(RefCell::new(Vec::new()));
        let (counter_hook, nodes_hook) = (Rc::clone(&element_counter), Rc::clone(&nodes));
        set_read_symbolizer(Box::new(move |bytes: &[u8; 32]| {
            let witness = Fr::from_le_bytes_mod_order(bytes);
            let i = counter_hook.get();
            counter_hook.set(i + 1);
            let name = label.element_name(frame_in_label, i);
            let var = alloc.borrow_mut().alloc_with_value(&name, &witness);
            nodes_hook.borrow_mut().push(var);
            var
        }));

        // RAII so the thread-local symbolizer is cleared on EVERY exit path —
        // including the `?` early-return below and any panic inside
        // `deserialize_compressed` — so a stale hook can never leak into a later
        // `MleAst`/`AstCommitment` deserialize. (Code-review #6.)
        struct SymbolizerGuard;
        impl Drop for SymbolizerGuard {
            fn drop(&mut self) {
                clear_read_symbolizer();
            }
        }
        let symbolizer_guard = SymbolizerGuard;

        // Standard self-delimiting decode loop — mirrors jolt-core's `read_all`.
        let mut cursor = frame.as_slice();
        let mut out: Vec<T> = Vec::new();
        while !cursor.is_empty() {
            match T::deserialize_compressed(&mut cursor) {
                Ok(value) => out.push(value),
                Err(_) => return Err(VerificationError),
            }
        }

        // Clear the hook (dropping its `nodes` Rc clone) so the collector can
        // be unwrapped.
        drop(symbolizer_guard);
        #[expect(clippy::expect_used)] // sole owner: the hook's clone was just dropped
        let nodes = Rc::try_unwrap(nodes)
            .expect("read symbolizer still holds the node collector")
            .into_inner();
        Ok((out, nodes))
    }
}

impl<L: SymbolicSpongeLayout> FsChallenge<MleAst> for SymbolicVerifierFs<L> {
    fn challenge_field(&mut self) -> MleAst {
        let c = self.layout.squeeze();
        self.squeezed_challenges.push(c);
        c
    }

    // Poseidon semantics (transcript_msgs): full-field squeeze; `challenge_field`
    // and `challenge_optimized` return the SAME value (no 128-bit masking), and
    // `MleAst::Challenge = MleAst`.
    fn challenge_optimized(&mut self) -> <MleAst as JoltField>::Challenge {
        self.challenge_field()
    }
}

/// Drain the pending-value channels for one serialized `value`, mapping a
/// symbolic scalar to its node and a concrete 32-byte-multiple serialization
/// to constant nodes. A pending COMMITMENT here is a stage-logic bug: a
/// commitment reached a scalar-typed absorb.
fn scalar_nodes_for<T: CanonicalSerialize>(value: &T, nodes: &mut Vec<MleAst>) {
    // Both pending channels must be empty BEFORE serializing (same guard pattern as
    // `absorb`/`absorb_slice`): a node left behind by an earlier serialize would
    // otherwise be silently consumed as if it were THIS value when `value` is
    // concrete (channel-hygiene bug, not data).
    assert!(
        take_pending_commitment_chunks().is_none() && take_pending_append().is_none(),
        "absorb_scalar(s): stale pending symbolic value before serialization \
         (channel-hygiene bug)"
    );
    let mut buf = Vec::new();
    let _ = value.serialize_compressed(&mut buf);
    if take_pending_commitment_chunks().is_some() {
        panic!("absorb_scalar(s): a commitment reached a scalar-typed absorb (stage-logic bug)");
    }
    if let Some(node) = take_pending_append() {
        nodes.push(node);
        return;
    }
    // Concrete value: its canonical bytes must be a sequence of 32-byte field
    // elements (mirrors the native `parse_scalar_units` contract).
    assert!(
        buf.len().is_multiple_of(32),
        "absorb_scalar(s): concrete value is not a sequence of 32-byte field elements \
         ({} bytes)",
        buf.len()
    );
    for chunk in buf.chunks_exact(32) {
        #[expect(clippy::expect_used)] // caller-contract violation, not data
        let fr = Fr::deserialize_compressed(chunk).expect("non-canonical scalar absorbed");
        nodes.push(fr_const(fr));
    }
}

impl<L: SymbolicSpongeLayout> FsAbsorb for SymbolicVerifierFs<L> {
    fn absorb<T: CanonicalSerialize>(&mut self, value: &T) {
        // Untyped absorb = the byte rule over the serialization (native
        // Poseidon `absorb`). A symbolic value here is unmirrorable without
        // byte decomposition — the field-aligned protocol moved every
        // symbolic absorb to the typed methods, so this is a stage-logic bug.
        let mut buf = Vec::new();
        let _ = value.serialize_compressed(&mut buf);
        assert!(
            take_pending_commitment_chunks().is_none() && take_pending_append().is_none(),
            "untyped absorb of a symbolic value — use absorb_scalar/absorb_commitment \
             (field-aligned transcript, spec §4.4)"
        );
        self.layout.absorb_byte_message(&buf);
    }

    fn absorb_slice<T: CanonicalSerialize>(&mut self, values: &[T]) {
        // One byte-rule message of concatenated serializations (NOT N messages).
        let mut bytes = Vec::new();
        for value in values {
            let mut buf = Vec::new();
            let _ = value.serialize_compressed(&mut buf);
            assert!(
                take_pending_commitment_chunks().is_none() && take_pending_append().is_none(),
                "untyped absorb_slice of a symbolic value — use absorb_scalars \
                 (field-aligned transcript, spec §4.4)"
            );
            bytes.extend_from_slice(&buf);
        }
        self.layout.absorb_byte_message(&bytes);
    }

    fn absorb_bytes(&mut self, bytes: &[u8]) {
        self.layout.absorb_byte_message(bytes);
    }

    fn absorb_scalar<T: CanonicalSerialize>(&mut self, value: &T) {
        let mut nodes = Vec::with_capacity(1);
        scalar_nodes_for(value, &mut nodes);
        self.layout.absorb_field_frame(&nodes);
    }

    // CRITICAL (review C7): the inherited default `absorb_scalars(values)` =
    // `absorb(&values.to_vec())` serializes the whole Vec through the
    // single-slot `PENDING_APPEND` thread-local, silently DROPPING k-1
    // symbolic values. Route per-element instead, then absorb ONE count-led
    // field frame — matching the native Poseidon `absorb_scalars`.
    fn absorb_scalars<T: CanonicalSerialize + Clone>(&mut self, values: &[T]) {
        let mut nodes = Vec::with_capacity(values.len());
        for value in values {
            scalar_nodes_for(value, &mut nodes);
        }
        self.layout.absorb_field_frame(&nodes);
    }

    fn absorb_commitment<T: CanonicalSerialize>(&mut self, value: &T) {
        // Symbolic commitments route their chunk vars through the
        // pending-chunks channel; concrete ones fall back to the byte rule
        // over their real serialization (both match the native
        // `absorb_commitment` = byte rule over the compressed bytes).
        let mut buf = Vec::new();
        let _ = value.serialize_compressed(&mut buf);
        if let Some(chunks) = take_pending_commitment_chunks() {
            self.layout.absorb_commitment_message(&chunks);
        } else if take_pending_append().is_some() {
            panic!("absorb_commitment of a symbolic scalar — use absorb_scalar (spec §4.4)");
        } else {
            self.layout.absorb_byte_message(&buf);
        }
    }

    fn absorb_commitment_bytes(&mut self, bytes: &[u8]) {
        self.layout.absorb_byte_message(bytes);
    }
}

impl<L: SymbolicSpongeLayout> VerifierFs<MleAst> for SymbolicVerifierFs<L> {
    fn read_slice<T: CanonicalDeserialize>(&mut self) -> VerificationResult<Vec<T>> {
        // Untyped frame read: deliberately UNSUPPORTED. The only sound symbolic
        // mirror would bake the frame's exact proof bytes into the circuit as
        // CONSTANTS (the native Poseidon `read_slice` absorbs them as a
        // `RawBytesMsg`), producing a circuit valid for that ONE proof — a
        // proof-specific artifact that must never be generated silently. No
        // frame on the non-ZK stages-1–7 path reaches this (uni-skip and
        // sumcheck rounds are `read_scalars`; the pre-stage frames are
        // `read_commitments`); only concrete-valued ZK-era frames would, and
        // ZK proofs are refused up front (spec §16 guardrail 4 / §17).
        panic!(
            "SymbolicVerifierFs::read_slice: untyped NARG frame read reached the symbolic \
             replay — would bake proof bytes as circuit constants (proof-specific circuit). \
             Use read_scalars/read_commitments, or extend the typed layout vocabulary \
             (specs/transpiler-optimization-spec.md §4.4)."
        );
    }

    fn read_scalars(&mut self) -> VerificationResult<Vec<MleAst>> {
        let (values, nodes) = self.pop_symbolized_frame::<MleAst>()?;
        debug_assert_eq!(values.len(), nodes.len(), "scalar frame node mismatch");
        self.layout.absorb_field_frame(&nodes);
        Ok(values)
    }

    fn read_commitments<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
    ) -> VerificationResult<Vec<T>> {
        let (values, nodes) = self.pop_symbolized_frame::<T>()?;
        // Each commitment must have symbolized to exactly the 13 chunk vars of
        // the Dory GT re-chunking (`AstCommitment::deserialize_with_mode`).
        if nodes.len() != values.len() * COMMITMENT_CHUNKS {
            return Err(VerificationError);
        }
        let groups: Vec<Vec<MleAst>> = nodes
            .chunks(COMMITMENT_CHUNKS)
            .map(<[MleAst]>::to_vec)
            .collect();
        self.layout.absorb_commitments_frame(&groups);
        Ok(values)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod field_aligned_tests {
    use super::*;
    use crate::ast_evaluator::eval_root;
    use crate::poseidon_model::{model_challenges, HighLevelOp};
    use ark_ff::UniformRand;
    use ark_serialize::CanonicalSerialize;
    use std::collections::HashMap;
    use zklean_extractor::mle_ast::node_arena;

    /// THE LEVEL-2 GATE (spec §10.1): drive `FieldAlignedLayout` through a
    /// mixed absorb/challenge schedule symbolically, evaluate the resulting
    /// challenge AST nodes against a concrete witness, and assert they equal
    /// the native sponge's challenges (oracle = `poseidon_model`, itself
    /// verified against a real `ProverState<PoseidonSponge>`).
    #[test]
    fn field_aligned_layout_matches_native_sponge() {
        let mut rng = ark_std::test_rng();
        let session = b"Jolt";
        let instance = [0x5Cu8; 32];

        // Symbolic vars with known witness values.
        let mut witness: HashMap<u16, Fr> = HashMap::new();
        let mut next_idx = 0u16;
        let mut mk = |vals: &[Fr], witness: &mut HashMap<u16, Fr>| -> Vec<MleAst> {
            vals.iter()
                .map(|v| {
                    let idx = next_idx;
                    next_idx += 1;
                    witness.insert(idx, *v);
                    MleAst::from_var(idx)
                })
                .collect()
        };

        let claims: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();
        let frame: Vec<Fr> = (0..5).map(|_| Fr::rand(&mut rng)).collect();
        // Two Dory GT commitments: 384 canonical bytes each, witness = the 13
        // byte-rule chunk values (12×31B + 1×12B, each < 2^248 < r).
        let gts: Vec<Vec<u8>> = (0..2)
            .map(|_| {
                let gt = ark_bn254::Fq12::rand(&mut rng);
                let mut b = Vec::new();
                gt.serialize_compressed(&mut b).unwrap();
                assert_eq!(b.len(), COMMITMENT_BYTES);
                b
            })
            .collect();
        let chunk_vals = |bytes: &[u8]| -> Vec<Fr> { jolt_transcript::commitment_to_chunks(bytes) };

        let claim_asts = mk(&claims, &mut witness);
        let frame_asts = mk(&frame, &mut witness);
        let gt_groups: Vec<Vec<MleAst>> = gts
            .iter()
            .map(|b| mk(&chunk_vals(b), &mut witness))
            .collect();

        // Symbolic schedule mirroring the verifier's op kinds; the model gets
        // the identical high-level ops.
        let mut layout = FieldAlignedLayout::new(session, &instance);
        let mut ops: Vec<HighLevelOp> = Vec::new();
        let mut challenges: Vec<MleAst> = Vec::new();

        // Commitments frame, then the empty advice-presence frame.
        layout.absorb_commitments_frame(&gt_groups);
        ops.push(HighLevelOp::AbsorbCommitments(gts.clone()));
        layout.absorb_commitments_frame(&[]);
        ops.push(HighLevelOp::AbsorbCommitments(Vec::new()));
        challenges.push(layout.squeeze());
        ops.push(HighLevelOp::ChallengeFr);

        // Single-scalar absorbs (input claims / flushed claims).
        for (c_ast, c_val) in claim_asts.iter().zip(&claims) {
            layout.absorb_scalar_message(c_ast);
            ops.push(HighLevelOp::AbsorbScalars(vec![*c_val]));
        }
        challenges.push(layout.squeeze());
        ops.push(HighLevelOp::ChallengeFr);

        // A read_scalars frame, then two back-to-back challenges.
        layout.absorb_field_frame(&frame_asts);
        ops.push(HighLevelOp::AbsorbScalars(frame.clone()));
        challenges.push(layout.squeeze());
        ops.push(HighLevelOp::ChallengeFr);
        challenges.push(layout.squeeze());
        ops.push(HighLevelOp::ChallengeFr);

        // A lone trusted-commitment absorb and a raw byte message.
        layout.absorb_commitment_message(&gt_groups[0]);
        ops.push(HighLevelOp::AbsorbBytes(gts[0].clone()));
        layout.absorb_byte_message(b"jolt-layout-test");
        ops.push(HighLevelOp::AbsorbBytes(b"jolt-layout-test".to_vec()));
        challenges.push(layout.squeeze());
        ops.push(HighLevelOp::ChallengeFr);

        // Evaluate the symbolic challenge ASTs against the witness.
        let arena = node_arena().read().unwrap().clone();
        let got: Vec<Fr> = challenges
            .iter()
            .map(|c| eval_root(&arena, c.root(), &witness))
            .collect();

        let expected = model_challenges(session, &instance, &ops);
        assert_eq!(
            got, expected,
            "FieldAlignedLayout challenges diverge from the native field-aligned PoseidonSponge"
        );
    }

    /// Negative control: dropping the F1 frame-count unit (absorbing the two
    /// per-GT groups without the leading `[Fr(2k+1), 0]` pair) must diverge.
    #[test]
    fn field_aligned_layout_negative_control_missing_frame_count() {
        let mut rng = ark_std::test_rng();
        let session = b"Jolt";
        let instance = [0x77u8; 32];

        let gt = ark_bn254::Fq12::rand(&mut rng);
        let mut bytes = Vec::new();
        gt.serialize_compressed(&mut bytes).unwrap();
        let chunk_vals: Vec<Fr> = jolt_transcript::commitment_to_chunks(&bytes);
        let mut witness: HashMap<u16, Fr> = HashMap::new();
        let chunks: Vec<MleAst> = chunk_vals
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let idx = 5000 + i as u16;
                witness.insert(idx, *v);
                MleAst::from_var(idx)
            })
            .collect();

        // Wrong: lone-commitment encoding where a commitments FRAME is required.
        let mut layout = FieldAlignedLayout::new(session, &instance);
        layout.absorb_commitment_message(&chunks);
        let wrong = layout.squeeze();

        let arena = node_arena().read().unwrap().clone();
        let wrong_val = eval_root(&arena, wrong.root(), &witness);
        let expected = model_challenges(
            session,
            &instance,
            &[
                HighLevelOp::AbsorbCommitments(vec![bytes]),
                HighLevelOp::ChallengeFr,
            ],
        );
        assert_ne!(
            wrong_val, expected[0],
            "missing frame-count unit also matched — F1 binding would be vacuous"
        );
    }

    /// C7 probe: `SymbolicVerifierFs::absorb_scalars` of k symbolic values must
    /// reach the layout as ONE field frame of k elements — the inherited default
    /// (`absorb(&values.to_vec())`) would serialize through the single-slot
    /// `PENDING_APPEND` thread-local and silently drop k−1 values (or panic on
    /// the untyped-symbolic guard).
    #[test]
    fn absorb_scalars_routes_all_symbolic_elements_through_one_frame() {
        /// Records every layout call instead of hashing.
        #[derive(Default)]
        struct RecordingLayout {
            field_frames: Vec<Vec<MleAst>>,
            byte_messages: usize,
        }
        impl SymbolicSpongeLayout for RecordingLayout {
            const FAITHFUL: bool = false;
            fn absorb_scalar_message(&mut self, value: &MleAst) {
                self.field_frames.push(vec![*value]);
            }
            fn absorb_field_frame(&mut self, elements: &[MleAst]) {
                self.field_frames.push(elements.to_vec());
            }
            fn absorb_commitments_frame(&mut self, _groups: &[Vec<MleAst>]) {}
            fn absorb_commitment_message(&mut self, _chunks: &[MleAst]) {}
            fn absorb_byte_message(&mut self, _bytes: &[u8]) {
                self.byte_messages += 1;
            }
            fn squeeze(&mut self) -> MleAst {
                MleAst::from_u64(0)
            }
        }

        let parsed = crate::narg_parser::parse_narg(&[], false).unwrap();
        let alloc = Rc::new(RefCell::new(VarAllocator::new()));
        let mut fs = SymbolicVerifierFs::new(RecordingLayout::default(), parsed, alloc);

        let vars: Vec<MleAst> = (0u16..3).map(|i| MleAst::from_var(9100 + i)).collect();
        FsAbsorb::absorb_scalars(&mut fs, &vars);

        assert_eq!(
            fs.layout.field_frames.len(),
            1,
            "absorb_scalars must produce exactly one field frame"
        );
        assert_eq!(
            fs.layout.field_frames[0], vars,
            "all 3 symbolic elements must reach the layout, in order"
        );
        assert_eq!(fs.layout.byte_messages, 0, "no byte-rule fallback");
    }

    /// Spec guardrail §8.6: the symbolic read path must REJECT a non-canonical
    /// (>= r) 32-byte element exactly like the native `FieldFrameMsg` / ark
    /// `Validate::Yes` path — decode error → `VerificationError`, never a
    /// silently-reduced witness value.
    #[test]
    fn read_scalars_rejects_non_canonical_element() {
        // One canonical element, then 32 bytes of 0xFF (> r).
        let mut body = Vec::new();
        Fr::from(7u64).serialize_compressed(&mut body).unwrap();
        body.extend_from_slice(&[0xFF; 32]);
        let mut narg = (body.len() as u64).to_le_bytes().to_vec();
        narg.extend_from_slice(&body);

        let parsed = crate::narg_parser::parse_narg(&narg, false).unwrap();
        let alloc = Rc::new(RefCell::new(VarAllocator::new()));
        let mut fs =
            SymbolicVerifierFs::new(FieldAlignedLayout::new(b"t", &[0u8; 32]), parsed, alloc);
        assert!(
            VerifierFs::read_scalars(&mut fs).is_err(),
            "non-canonical (>= r) frame element must be rejected, matching the native verifier"
        );
    }
}
