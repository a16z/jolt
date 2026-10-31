//! Program-bound verifier material for recursive execution. The key and terminal
//! transform must come from trusted provisioning and be fixed by the guest image.

use akita_pcs::{AkitaSerialize, Compress, TrustedBytes};
use akita_types::AkitaExpandedSetup;
use akita_verifier::{build_riscv64_terminal_ntt_cache, TrustedTerminalCache};
use jolt_openings::OpeningsError;

use crate::adapters::{invalid_setup, AkitaBackendVerifierSetup};
use crate::one_hot_family::with_one_hot_family;
use crate::{AkitaBatchProof, AkitaField, AkitaVerifierSetup};

impl AkitaVerifierSetup {
    /// Export the seed-derived key and selected row's terminal transform for
    /// embedding in a recursion guest. The key body must start on a 16-byte
    /// boundary; its leading skew aligns the in-place matrix coefficients.
    pub fn prepare_recursion_material(
        &self,
        proof: &AkitaBatchProof,
    ) -> Result<(Vec<u8>, Vec<u8>), OpeningsError> {
        let selection = proof.selection();
        let verifier = self.one_hot_verifier()?;
        let (key, terminal) = with_one_hot_family!(verifier verifier, |verifier| {
            let key = verifier.setup();
            let row = verifier.schedules().resolve_selection(selection).map_err(invalid_setup)?;
            let terminal = build_riscv64_terminal_ntt_cache(key, row.schedule(), selection.row_digest).map_err(invalid_setup)?;
            let mut bytes = Vec::with_capacity(key.serialized_size(Compress::No));
            key.serialize_with_mode(&mut bytes, Compress::No).map_err(invalid_setup)?;
            Ok::<_, OpeningsError>((bytes, terminal))
        })?;
        let offset =
            AkitaExpandedSetup::<AkitaField>::coefficient_offset(&key).map_err(invalid_setup)?;
        let skew = 16 - offset % 16;
        let mut body = vec![0; skew];
        *body
            .first_mut()
            .ok_or_else(|| invalid_setup("empty key padding"))? =
            u8::try_from(skew).map_err(invalid_setup)?;
        body.extend_from_slice(&key);
        Ok((body, terminal))
    }

    /// Install material exported by `prepare_recursion_material`, fixed by the
    /// verifier program identity. `TrustedBytes` vouches for canonical field
    /// words; the caller must additionally authenticate the seed derivation of
    /// both payloads. Ordinary proof input does not meet that requirement.
    pub fn install_recursion_material(
        &self,
        proof: &AkitaBatchProof,
        key: TrustedBytes,
        terminal: TrustedBytes,
    ) -> Result<(), OpeningsError> {
        let skew = usize::from(
            *key.bytes()
                .first()
                .ok_or_else(|| invalid_setup("missing key padding"))?,
        );
        if !(1..=16).contains(&skew) {
            return Err(invalid_setup("invalid key padding"));
        }
        let key = key
            .skip(skew)
            .ok_or_else(|| invalid_setup("truncated key padding"))?;
        let setup = AkitaBackendVerifierSetup::borrow_trusted(key).map_err(invalid_setup)?;
        let verifier = self
            .one_hot_scheme()?
            .verifier_for_selection(
                setup,
                proof.selection(),
                TrustedTerminalCache::View(terminal.bytes()),
            )
            .map_err(invalid_setup)?;
        self.backend_cache.install_one_hot(verifier)
    }
}
