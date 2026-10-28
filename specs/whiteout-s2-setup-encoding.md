# Whiteout S2: the verifier setup's encoding under `zk`

Debt S2 in `specs/whiteout-optimization-campaign.md`.

## Problem

Since W6b, a `zk` build's `AkitaVerifierSetup` ended with
`#[cfg(feature = "zk")] #[serde(default)] whiteout: Option<WhiteoutSetup>`.
Bincode is not self-describing and ignores `serde(default)`, so a clear
setup's bytes depended on the feature: once any crate unified `zk` on, setup
bytes written by a build without it no longer decoded (`UnexpectedEnd` in
`legacy_single_setup_keeps_its_transcript`), and a clear setup encoded under
`zk` carried a trailing `None` tag that a build without it rejects.

## Change

The Whiteout setup moved from a trailing optional field into a typed
schedule variant, appended last to `AkitaVerifierScheduleArtifacts`:

```rust
#[cfg(feature = "zk")]
OneHotWhiteout { one_hot: Vec<u8>, whiteout: WhiteoutSetup },
```

- **Clear bytes.** `AkitaVerifierSetup` has no feature-gated field. The
  variant is last, so `zk` leaves the clear variants' discriminants, and every
  clear setup's encoding, as they are without it (the convention for
  serialized enums in `CLAUDE.md`; the multi-chunk variants were appended the
  same way, `specs/akita-multi-chunk-witnesses.md`). The legacy-bytes test
  now passes in both builds.
- **Absence by type.** A setup is clear or Whiteout by its variant; there is
  no `Option` beside the schedule that could disagree with it.
  `AkitaVerifierSetup::whiteout()` reads the variant.
- **Owner.** `WhiteoutSetup` is the masked opening's persisted plans (catalog
  and response schedules), so it belongs beside the one-hot schedule it
  extends. Keeping it inside `AkitaVerifierSetup` keeps
  `MaskedGroupOpening::Setup = CommitmentScheme::VerifierSetup` and the public
  `jolt_verifier::verify` signature unchanged. A wrapper on Jolt's verifier
  preprocessing would have needed a second, PCS-specific setup type threaded
  through every Whiteout entry point.
- **Setup check.** Whiteout masks one single-chunk one-hot trace group.
  `AkitaScheme::setup` now refuses Whiteout on any other flavor or chunk
  profile at setup. Before, it built the setup and the first commitment
  failed. The K=16 check stays where it was (`opening_shape`).

`AkitaSetupParams` still carries `#[cfg(feature = "zk")] #[serde(default)]
whiteout: Option<WhiteoutSource>`. Setup parameters are regenerable inputs
whose bincode compatibility is not kept (`specs/akita-multi-chunk-witnesses.md`);
only the verifier setup is a transported artifact.

## Binding

Nothing absorbs a setup's serialized bytes. The transcript binds the setup
through derived values:

- the clear setup catalog (`absorb_setup_catalog`) absorbs the arities,
  `one_hot_k`, the chunk count (`akita_chunk_profile()`, `Single` for the new
  variant as before), the layout digest and the one-hot catalog digest of the
  `one_hot` bytes, which are the same bytes the setup held before;
- the Whiteout side absorbs the masked catalog's digest, the selected row
  digest and the response plan's catalog digest at PADS and at Stage 8
  (`WhiteoutTail::pad_statement`, `WhiteoutRow::absorb_selection`; D32). These
  are computed from the `WhiteoutSetup` value, wherever it is encoded.

The same values are absorbed in the same order, so nothing bound before is
unbound now. The Jolt preprocessing digest covers the program alone, before
and after. No paper delta is needed.

## Fixtures

The transcript does not change, so no proof or transcript digest changes,
clear or Whiteout. A Whiteout verifier preprocessing re-encodes differently:
the `Option` tag is gone and the variant index is 5. No checked-in fixture
pins those bytes. `whiteout_plans_persist_and_are_consumed` compares
encodings within one build, and it passes.
