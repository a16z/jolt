//! Verifier state a trusted host prepares for one schedule row.
//!
//! An ordinary [`AkitaVerifierSetup`] carries shapes and JSON catalogs and
//! re-derives the backend key from the setup seed on first use. A verifier
//! running as a zkVM guest cannot afford that: seed expansion, catalog
//! parsing, and the terminal matrix NTT would dominate its cycles. A
//! [`PreparedVerifier`] carries their results for the single row a proof
//! selected.
//!
//! The payloads are trusted setup data, not proof data. The backend key is
//! taken as seed-derived without re-deriving it, and an attached terminal NTT
//! cache's residues are used without a range pass. The catalog view repeats
//! the catalog's semantic audit and reproduces the complete catalog digest
//! the transcript binds. Callers must bind the payload bytes to the verifier
//! program identity, for example by embedding them in the guest image.

use akita_pcs::{
    build_riscv64_terminal_ntt_cache, AkitaDeserialize, AkitaSerialize, Compress,
    TrustedTerminalCache, Validate,
};
use akita_types::{AkitaExpandedSetup, OpeningScheduleSelection, ScheduleRowDigest};
use jolt_openings::OpeningsError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_bytes::{ByteBuf, Bytes};

use crate::adapters::{
    invalid_setup, AkitaBackendFlavor, AkitaBackendVerifierSetup, AkitaField, AkitaVerifierSetup,
    SCHEDULE_SELECTION_BYTES,
};

/// One payload's bytes.
///
/// A host owns the bytes it prepared. For transport it may detach them into an
/// out-of-line body ([`AkitaVerifierSetup::detach_prepared_payloads`]); a
/// guest then attaches that body in place for the program's lifetime
/// ([`AkitaVerifierSetup::attach_prepared_payloads`]), so multi-megabyte
/// payloads are never copied out of the setup record.
///
/// Serialized as an optional byte string, absent when detached. Bincode would
/// otherwise decode a `Vec<u8>` element by element, at roughly 27 guest cycles
/// per byte.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PreparedBytes {
    Owned(Vec<u8>),
    Detached,
    Attached(&'static [u8]),
}

impl PreparedBytes {
    /// The payload bytes, or an error for a body that was never attached.
    pub(crate) fn bytes(&self) -> Result<&[u8], OpeningsError> {
        match self {
            Self::Owned(bytes) => Ok(bytes),
            Self::Attached(bytes) => Ok(bytes),
            Self::Detached => Err(invalid_setup(
                "prepared payload was detached and not attached",
            )),
        }
    }

    /// Move the bytes into a detached body.
    ///
    /// A body is self-aligning: its first byte is a skew `s` in `1..=8` and the
    /// payload starts at offset `s`, chosen so that, with the body on an
    /// 8-byte address, payload offset `aligned_offset` is 8-byte aligned.
    fn detach(&mut self, aligned_offset: usize) -> Result<Vec<u8>, OpeningsError> {
        let skew = 8 - aligned_offset % 8;
        let mut body = Vec::with_capacity(skew + self.bytes()?.len());
        body.push(u8::try_from(skew).map_err(invalid_setup)?);
        body.resize(skew, 0);
        body.extend_from_slice(self.bytes()?);
        *self = Self::Detached;
        Ok(body)
    }

    fn attach(body: &'static [u8]) -> Result<Self, OpeningsError> {
        let skew = usize::from(
            *body
                .first()
                .ok_or_else(|| invalid_setup("empty prepared payload body"))?,
        );
        if !(1..=8).contains(&skew) {
            return Err(invalid_setup("malformed prepared payload body skew"));
        }
        let payload = body
            .get(skew..)
            .ok_or_else(|| invalid_setup("truncated prepared payload body"))?;
        Ok(Self::Attached(payload))
    }
}

impl Serialize for PreparedBytes {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Owned(bytes) => Some(Bytes::new(bytes)).serialize(serializer),
            Self::Attached(bytes) => Some(Bytes::new(bytes)).serialize(serializer),
            Self::Detached => None::<&Bytes>.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for PreparedBytes {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Option::<ByteBuf>::deserialize(deserializer)?
            .map_or(Self::Detached, |bytes| Self::Owned(bytes.into_vec())))
    }
}

/// Backend verifier state for the schedule row `row_digest` of one flavor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreparedVerifier {
    pub(crate) flavor: AkitaBackendFlavor,
    pub(crate) row_digest: [u8; SCHEDULE_SELECTION_BYTES],
    /// The backend verifier key (public matrix and prefix registry),
    /// serialized uncompressed.
    pub(crate) key: PreparedBytes,
    /// A catalog verifier view carrying only `row_digest`'s parameters.
    pub(crate) catalog: PreparedBytes,
    /// The scalar Q128 terminal matrix artifact for `row_digest`.
    pub(crate) terminal_ntt: PreparedBytes,
}

impl PreparedVerifier {
    pub(crate) fn selection(&self) -> OpeningScheduleSelection {
        OpeningScheduleSelection {
            row_digest: ScheduleRowDigest::from_bytes(self.row_digest),
        }
    }

    /// The backend key: viewed in place when attached, else decoded without
    /// re-deriving the matrix from the seed.
    pub(crate) fn backend_key(&self) -> Result<AkitaBackendVerifierSetup, OpeningsError> {
        match &self.key {
            PreparedBytes::Attached(bytes) => {
                AkitaBackendVerifierSetup::borrow_trusted(bytes).map_err(invalid_setup)
            }
            key @ (PreparedBytes::Owned(_) | PreparedBytes::Detached) => {
                AkitaBackendVerifierSetup::deserialize_with_mode(
                    key.bytes()?,
                    Compress::No,
                    Validate::No,
                    &(),
                )
                .map_err(invalid_setup)
            }
        }
    }

    pub(crate) fn terminal_cache(&self) -> Result<TrustedTerminalCache<'_>, OpeningsError> {
        Ok(match &self.terminal_ntt {
            PreparedBytes::Attached(bytes) => TrustedTerminalCache::View(bytes),
            artifact @ (PreparedBytes::Owned(_) | PreparedBytes::Detached) => {
                TrustedTerminalCache::Decode(artifact.bytes()?)
            }
        })
    }
}

impl AkitaVerifierSetup {
    /// Prepare the backend verifier for the schedule row a proof selected
    /// ([`crate::AkitaBatchProof::schedule_row_digest`]) and carry it in the
    /// setup: the expanded key, a catalog view of that row, and its terminal
    /// matrix NTT. The setup then verifies only proofs for that row of its
    /// flavor. Returns the prepared byte count.
    ///
    /// # Errors
    ///
    /// Returns an error when no catalog of this setup contains the row, or
    /// building or encoding a payload fails.
    pub fn prepare_verifier(
        &mut self,
        row_digest: [u8; SCHEDULE_SELECTION_BYTES],
    ) -> Result<usize, OpeningsError> {
        let selection = OpeningScheduleSelection {
            row_digest: ScheduleRowDigest::from_bytes(row_digest),
        };
        let mut found = None;
        for flavor in [AkitaBackendFlavor::Dense, AkitaBackendFlavor::OneHot] {
            let Some(catalog) = self.schedule_catalog(flavor)? else {
                continue;
            };
            if let Ok(row) = catalog.resolve_selection(selection) {
                let view = catalog
                    .to_verifier_view(&[selection])
                    .map_err(invalid_setup)?;
                found = Some((flavor, row.schedule().clone(), view));
                break;
            }
        }
        let (flavor, schedule, catalog) =
            found.ok_or_else(|| invalid_setup("no catalog of this setup contains the row"))?;
        let key = self.derive_backend_key(flavor)?;
        let terminal_ntt = build_riscv64_terminal_ntt_cache(&key, &schedule, selection.row_digest)
            .map_err(invalid_setup)?;
        let mut key_bytes = Vec::with_capacity(key.serialized_size(Compress::No));
        key.serialize_with_mode(&mut key_bytes, Compress::No)
            .map_err(invalid_setup)?;
        let total = key_bytes.len() + catalog.len() + terminal_ntt.len();
        self.prepared = Some(PreparedVerifier {
            flavor,
            row_digest,
            key: PreparedBytes::Owned(key_bytes),
            catalog: PreparedBytes::Owned(catalog),
            terminal_ntt: PreparedBytes::Owned(terminal_ntt),
        });
        self.backend_cache = Default::default();
        Ok(total)
    }

    /// Move every payload out of the setup into out-of-line bodies, in the
    /// order [`Self::attach_prepared_payloads`] expects them back: the schedule
    /// artifacts, then the prepared key, catalog view, and terminal NTT cache.
    ///
    /// # Errors
    ///
    /// Returns an error when a payload is already detached or the key header
    /// cannot be parsed.
    pub fn detach_prepared_payloads(&mut self) -> Result<Vec<Vec<u8>>, OpeningsError> {
        let mut bodies = Vec::new();
        for slot in self.schedule_artifacts.slots() {
            bodies.push(slot.detach(0)?);
        }
        if let Some(prepared) = &mut self.prepared {
            // Aligning the key's coefficients lets the guest view them in place.
            let offset =
                AkitaExpandedSetup::<AkitaField>::coefficient_offset(prepared.key.bytes()?)
                    .map_err(invalid_setup)?;
            bodies.push(prepared.key.detach(offset)?);
            bodies.push(prepared.catalog.detach(0)?);
            bodies.push(prepared.terminal_ntt.detach(0)?);
        }
        Ok(bodies)
    }

    /// Attach detached payload bodies in place for the program's lifetime.
    ///
    /// # Errors
    ///
    /// Returns an error when the body count does not match the detached
    /// payloads or a body is malformed.
    pub fn attach_prepared_payloads(
        &mut self,
        bodies: &[&'static [u8]],
    ) -> Result<(), OpeningsError> {
        let mut bodies = bodies.iter();
        let slots =
            self.schedule_artifacts
                .slots()
                .into_iter()
                .chain(self.prepared.iter_mut().flat_map(|prepared| {
                    [
                        &mut prepared.key,
                        &mut prepared.catalog,
                        &mut prepared.terminal_ntt,
                    ]
                }));
        for slot in slots.filter(|slot| **slot == PreparedBytes::Detached) {
            let body = bodies.next().ok_or_else(|| {
                invalid_setup("fewer prepared payload bodies than detached slots")
            })?;
            *slot = PreparedBytes::attach(body)?;
        }
        if bodies.next().is_some() {
            return Err(invalid_setup(
                "more prepared payload bodies than detached slots",
            ));
        }
        self.backend_cache = Default::default();
        Ok(())
    }
}
