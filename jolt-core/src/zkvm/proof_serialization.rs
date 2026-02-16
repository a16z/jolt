use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use num::FromPrimitive;

use crate::{
    field::JoltField,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, dory::DoryLayout},
        opening_proof::{OpeningId, OpeningPoint, Openings, SumcheckId},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    zkvm::{
        config::{OneHotConfig, ReadWriteConfig},
        instruction::{CircuitFlags, InstructionFlags},
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use crate::{
    poly::opening_proof::PolynomialId, subprotocols::univariate_skip::UniSkipFirstRoundProof,
};

use crate::zkvm::transport;

/// Stream signature for `JoltProof` bytes (clean rewrite).
///
/// This is a short fixed header to fail fast on wrong-format inputs.
const PROOF_SIGNATURE: &[u8; 8] = b"JOLTPRF\0";

// Frame tags for proof sections. Decoding is strict: unknown tags are rejected.
const TAG_PARAMS: u8 = 1;
const TAG_COMMITMENTS: u8 = 2;
const TAG_OPENING_CLAIMS: u8 = 3;
const TAG_STAGE1: u8 = 10;
const TAG_STAGE2: u8 = 11;
const TAG_STAGE3: u8 = 12;
const TAG_STAGE4: u8 = 13;
const TAG_STAGE5: u8 = 14;
const TAG_STAGE6: u8 = 15;
const TAG_STAGE7: u8 = 16;
const TAG_JOINT_OPENING: u8 = 20;

// Per-section payload caps (DoS resistance). These can be tuned if legitimate proofs grow.
const MAX_PARAMS_LEN: u64 = 16 * 1024;
const MAX_COMMITMENTS_LEN: u64 = 256 * 1024 * 1024;
const MAX_OPENING_CLAIMS_LEN: u64 = 256 * 1024 * 1024;
const MAX_STAGE_LEN: u64 = 512 * 1024 * 1024;
const MAX_JOINT_OPENING_LEN: u64 = 512 * 1024 * 1024;
const MAX_ANY_SECTION_LEN: u64 = 512 * 1024 * 1024;

pub struct JoltProof<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> {
    pub opening_claims: Claims<F>,
    pub commitments: Vec<PCS::Commitment>,
    pub stage1_uni_skip_first_round_proof: UniSkipFirstRoundProof<F, FS>,
    pub stage1_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    pub stage2_uni_skip_first_round_proof: UniSkipFirstRoundProof<F, FS>,
    pub stage2_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    pub stage3_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    pub stage4_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    pub stage5_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    pub stage6_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    pub stage7_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    pub joint_opening_proof: PCS::Proof,
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    pub trace_length: usize,
    pub ram_K: usize,
    pub bytecode_K: usize,
    pub rw_config: ReadWriteConfig,
    pub one_hot_config: OneHotConfig,
    pub dory_layout: DoryLayout,
}

#[inline]
fn io_err(e: std::io::Error) -> SerializationError {
    SerializationError::IoError(e)
}

#[inline]
fn write_u8<W: Write>(w: &mut W, b: u8) -> Result<(), SerializationError> {
    w.write_all(&[b]).map_err(io_err)
}

#[inline]
fn write_varint_u64<W: Write>(w: &mut W, x: u64) -> Result<(), SerializationError> {
    transport::write_varint_u64(w, x).map_err(io_err)
}

#[inline]
fn read_varint_u64<R: Read>(r: &mut R) -> Result<u64, SerializationError> {
    transport::read_varint_u64(r).map_err(io_err)
}

#[inline]
fn write_frame_header<W: Write>(w: &mut W, tag: u8, len: u64) -> Result<(), SerializationError> {
    transport::write_frame_header(w, tag, len).map_err(io_err)
}

#[inline]
fn read_frame_header<R: Read>(
    r: &mut R,
    max_len: u64,
) -> Result<Option<(u8, u64)>, SerializationError> {
    transport::read_frame_header(r, max_len).map_err(io_err)
}

#[inline]
fn section_cap_for_tag(tag: u8) -> u64 {
    match tag {
        TAG_PARAMS => MAX_PARAMS_LEN,
        TAG_COMMITMENTS => MAX_COMMITMENTS_LEN,
        TAG_OPENING_CLAIMS => MAX_OPENING_CLAIMS_LEN,
        TAG_STAGE1 | TAG_STAGE2 | TAG_STAGE3 | TAG_STAGE4 | TAG_STAGE5 | TAG_STAGE6
        | TAG_STAGE7 => MAX_STAGE_LEN,
        TAG_JOINT_OPENING => MAX_JOINT_OPENING_LEN,
        _ => 0,
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> CanonicalSerialize
    for JoltProof<F, PCS, FS>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        transport::signature_write(&mut writer, PROOF_SIGNATURE).map_err(io_err)?;

        // ---------------- Params ----------------
        let params_len = (transport::varint_u64_len(self.trace_length as u64)
            + transport::varint_u64_len(self.ram_K as u64)
            + transport::varint_u64_len(self.bytecode_K as u64)) as u64
            + self.rw_config.serialized_size(compress) as u64
            + self.one_hot_config.serialized_size(compress) as u64
            + self.dory_layout.serialized_size(compress) as u64;
        write_frame_header(&mut writer, TAG_PARAMS, params_len)?;
        write_varint_u64(&mut writer, self.trace_length as u64)?;
        write_varint_u64(&mut writer, self.ram_K as u64)?;
        write_varint_u64(&mut writer, self.bytecode_K as u64)?;
        self.rw_config.serialize_with_mode(&mut writer, compress)?;
        self.one_hot_config
            .serialize_with_mode(&mut writer, compress)?;
        self.dory_layout
            .serialize_with_mode(&mut writer, compress)?;

        // ---------------- Commitments ----------------
        let commitments_count_len = transport::varint_u64_len(self.commitments.len() as u64) as u64;
        let commitments_items_len: u64 = self
            .commitments
            .iter()
            .map(|c| c.serialized_size(compress) as u64)
            .sum();
        let untrusted_commitment_len = self
            .untrusted_advice_commitment
            .as_ref()
            .map(|c| c.serialized_size(compress) as u64)
            .unwrap_or(0);
        let commitments_len =
            commitments_count_len + commitments_items_len + 1 + untrusted_commitment_len;
        write_frame_header(&mut writer, TAG_COMMITMENTS, commitments_len)?;
        write_varint_u64(&mut writer, self.commitments.len() as u64)?;
        for c in &self.commitments {
            c.serialize_with_mode(&mut writer, compress)?;
        }
        match &self.untrusted_advice_commitment {
            None => write_u8(&mut writer, 0)?,
            Some(c) => {
                write_u8(&mut writer, 1)?;
                c.serialize_with_mode(&mut writer, compress)?;
            }
        }

        // ---------------- Opening claims ----------------
        let claims_count_len = transport::varint_u64_len(self.opening_claims.0.len() as u64) as u64;
        let claims_items_len: u64 = self
            .opening_claims
            .0
            .iter()
            .map(|(k, (_p, claim))| {
                (k.serialized_size(compress) + claim.serialized_size(compress)) as u64
            })
            .sum();
        let claims_len = claims_count_len + claims_items_len;
        write_frame_header(&mut writer, TAG_OPENING_CLAIMS, claims_len)?;
        write_varint_u64(&mut writer, self.opening_claims.0.len() as u64)?;
        for (k, (_p, claim)) in self.opening_claims.0.iter() {
            k.serialize_with_mode(&mut writer, compress)?;
            claim.serialize_with_mode(&mut writer, compress)?;
        }

        // ---------------- Stages ----------------
        let stage1_len = self
            .stage1_uni_skip_first_round_proof
            .serialized_size(compress) as u64
            + self.stage1_sumcheck_proof.serialized_size(compress) as u64;
        write_frame_header(&mut writer, TAG_STAGE1, stage1_len)?;
        self.stage1_uni_skip_first_round_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage1_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;

        let stage2_len = self
            .stage2_uni_skip_first_round_proof
            .serialized_size(compress) as u64
            + self.stage2_sumcheck_proof.serialized_size(compress) as u64;
        write_frame_header(&mut writer, TAG_STAGE2, stage2_len)?;
        self.stage2_uni_skip_first_round_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;

        let stage3_len = self.stage3_sumcheck_proof.serialized_size(compress) as u64;
        write_frame_header(&mut writer, TAG_STAGE3, stage3_len)?;
        self.stage3_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;

        let stage4_len = self.stage4_sumcheck_proof.serialized_size(compress) as u64;
        write_frame_header(&mut writer, TAG_STAGE4, stage4_len)?;
        self.stage4_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;

        let stage5_len = self.stage5_sumcheck_proof.serialized_size(compress) as u64;
        write_frame_header(&mut writer, TAG_STAGE5, stage5_len)?;
        self.stage5_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;

        let stage6_len = self.stage6_sumcheck_proof.serialized_size(compress) as u64;
        write_frame_header(&mut writer, TAG_STAGE6, stage6_len)?;
        self.stage6_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;

        let stage7_len = self.stage7_sumcheck_proof.serialized_size(compress) as u64;
        write_frame_header(&mut writer, TAG_STAGE7, stage7_len)?;
        self.stage7_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;

        // ---------------- Joint opening proof ----------------
        let joint_len = self.joint_opening_proof.serialized_size(compress) as u64;
        write_frame_header(&mut writer, TAG_JOINT_OPENING, joint_len)?;
        self.joint_opening_proof
            .serialize_with_mode(&mut writer, compress)?;

        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut size = PROOF_SIGNATURE.len();

        let params_len = (transport::varint_u64_len(self.trace_length as u64)
            + transport::varint_u64_len(self.ram_K as u64)
            + transport::varint_u64_len(self.bytecode_K as u64)) as u64
            + self.rw_config.serialized_size(compress) as u64
            + self.one_hot_config.serialized_size(compress) as u64
            + self.dory_layout.serialized_size(compress) as u64;
        size += 1 + transport::varint_u64_len(params_len) + params_len as usize;

        let commitments_count_len = transport::varint_u64_len(self.commitments.len() as u64) as u64;
        let commitments_items_len: u64 = self
            .commitments
            .iter()
            .map(|c| c.serialized_size(compress) as u64)
            .sum();
        let untrusted_commitment_len = self
            .untrusted_advice_commitment
            .as_ref()
            .map(|c| c.serialized_size(compress) as u64)
            .unwrap_or(0);
        let commitments_len =
            commitments_count_len + commitments_items_len + 1 + untrusted_commitment_len;
        size += 1 + transport::varint_u64_len(commitments_len) + commitments_len as usize;

        let claims_count_len = transport::varint_u64_len(self.opening_claims.0.len() as u64) as u64;
        let claims_items_len: u64 = self
            .opening_claims
            .0
            .iter()
            .map(|(k, (_p, claim))| {
                (k.serialized_size(compress) + claim.serialized_size(compress)) as u64
            })
            .sum();
        let claims_len = claims_count_len + claims_items_len;
        size += 1 + transport::varint_u64_len(claims_len) + claims_len as usize;

        let stage1_len = self
            .stage1_uni_skip_first_round_proof
            .serialized_size(compress) as u64
            + self.stage1_sumcheck_proof.serialized_size(compress) as u64;
        size += 1 + transport::varint_u64_len(stage1_len) + stage1_len as usize;

        let stage2_len = self
            .stage2_uni_skip_first_round_proof
            .serialized_size(compress) as u64
            + self.stage2_sumcheck_proof.serialized_size(compress) as u64;
        size += 1 + transport::varint_u64_len(stage2_len) + stage2_len as usize;

        let stage3_len = self.stage3_sumcheck_proof.serialized_size(compress) as u64;
        size += 1 + transport::varint_u64_len(stage3_len) + stage3_len as usize;

        let stage4_len = self.stage4_sumcheck_proof.serialized_size(compress) as u64;
        size += 1 + transport::varint_u64_len(stage4_len) + stage4_len as usize;

        let stage5_len = self.stage5_sumcheck_proof.serialized_size(compress) as u64;
        size += 1 + transport::varint_u64_len(stage5_len) + stage5_len as usize;

        let stage6_len = self.stage6_sumcheck_proof.serialized_size(compress) as u64;
        size += 1 + transport::varint_u64_len(stage6_len) + stage6_len as usize;

        let stage7_len = self.stage7_sumcheck_proof.serialized_size(compress) as u64;
        size += 1 + transport::varint_u64_len(stage7_len) + stage7_len as usize;

        let joint_len = self.joint_opening_proof.serialized_size(compress) as u64;
        size += 1 + transport::varint_u64_len(joint_len) + joint_len as usize;

        size
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> Valid
    for JoltProof<F, PCS, FS>
{
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> CanonicalDeserialize
    for JoltProof<F, PCS, FS>
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        transport::signature_check(&mut reader, PROOF_SIGNATURE).map_err(io_err)?;

        let mut trace_length: Option<usize> = None;
        let mut ram_K: Option<usize> = None;
        let mut bytecode_K: Option<usize> = None;
        let mut rw_config: Option<ReadWriteConfig> = None;
        let mut one_hot_config: Option<OneHotConfig> = None;
        let mut dory_layout: Option<DoryLayout> = None;

        let mut commitments: Option<Vec<PCS::Commitment>> = None;
        let mut untrusted_advice_commitment: Option<Option<PCS::Commitment>> = None;
        let mut opening_claims: Option<Claims<F>> = None;

        let mut stage1_uni: Option<UniSkipFirstRoundProof<F, FS>> = None;
        let mut stage1_sumcheck: Option<SumcheckInstanceProof<F, FS>> = None;
        let mut stage2_uni: Option<UniSkipFirstRoundProof<F, FS>> = None;
        let mut stage2_sumcheck: Option<SumcheckInstanceProof<F, FS>> = None;
        let mut stage3_sumcheck: Option<SumcheckInstanceProof<F, FS>> = None;
        let mut stage4_sumcheck: Option<SumcheckInstanceProof<F, FS>> = None;
        let mut stage5_sumcheck: Option<SumcheckInstanceProof<F, FS>> = None;
        let mut stage6_sumcheck: Option<SumcheckInstanceProof<F, FS>> = None;
        let mut stage7_sumcheck: Option<SumcheckInstanceProof<F, FS>> = None;
        let mut joint_opening_proof: Option<PCS::Proof> = None;

        while let Some((tag, len)) = read_frame_header(&mut reader, MAX_ANY_SECTION_LEN)? {
            let cap = section_cap_for_tag(tag);
            if cap == 0 || len > cap {
                return Err(SerializationError::InvalidData);
            }
            let mut limited = (&mut reader).take(len);

            match tag {
                TAG_PARAMS => {
                    if trace_length.is_some() {
                        return Err(SerializationError::InvalidData);
                    }
                    let t = read_varint_u64(&mut limited)?;
                    let r = read_varint_u64(&mut limited)?;
                    let b = read_varint_u64(&mut limited)?;
                    trace_length =
                        Some(usize::try_from(t).map_err(|_| SerializationError::InvalidData)?);
                    ram_K = Some(usize::try_from(r).map_err(|_| SerializationError::InvalidData)?);
                    bytecode_K =
                        Some(usize::try_from(b).map_err(|_| SerializationError::InvalidData)?);
                    rw_config = Some(ReadWriteConfig::deserialize_with_mode(
                        &mut limited,
                        compress,
                        validate,
                    )?);
                    one_hot_config = Some(OneHotConfig::deserialize_with_mode(
                        &mut limited,
                        compress,
                        validate,
                    )?);
                    dory_layout = Some(DoryLayout::deserialize_with_mode(
                        &mut limited,
                        compress,
                        validate,
                    )?);
                }
                TAG_COMMITMENTS => {
                    if commitments.is_some() {
                        return Err(SerializationError::InvalidData);
                    }
                    let n = read_varint_u64(&mut limited)?;
                    let n_usize =
                        usize::try_from(n).map_err(|_| SerializationError::InvalidData)?;
                    if n_usize > 1_000_000 {
                        return Err(SerializationError::InvalidData);
                    }
                    let mut v = Vec::with_capacity(n_usize.min(1024));
                    for _ in 0..n_usize {
                        v.push(PCS::Commitment::deserialize_with_mode(
                            &mut limited,
                            compress,
                            validate,
                        )?);
                    }
                    let presence = u8::deserialize_with_mode(&mut limited, compress, validate)?;
                    let opt = match presence {
                        0 => None,
                        1 => Some(PCS::Commitment::deserialize_with_mode(
                            &mut limited,
                            compress,
                            validate,
                        )?),
                        _ => return Err(SerializationError::InvalidData),
                    };
                    commitments = Some(v);
                    untrusted_advice_commitment = Some(opt);
                }
                TAG_OPENING_CLAIMS => {
                    if opening_claims.is_some() {
                        return Err(SerializationError::InvalidData);
                    }
                    let n = read_varint_u64(&mut limited)?;
                    let n_usize =
                        usize::try_from(n).map_err(|_| SerializationError::InvalidData)?;
                    if n_usize > 10_000_000 {
                        return Err(SerializationError::InvalidData);
                    }
                    let mut claims = BTreeMap::new();
                    for _ in 0..n_usize {
                        let key =
                            OpeningId::deserialize_with_mode(&mut limited, compress, validate)?;
                        let claim = F::deserialize_with_mode(&mut limited, compress, validate)?;
                        claims.insert(key, (OpeningPoint::default(), claim));
                    }
                    opening_claims = Some(Claims(claims));
                }
                TAG_STAGE1 => {
                    if stage1_uni.is_some() {
                        return Err(SerializationError::InvalidData);
                    }
                    stage1_uni = Some(UniSkipFirstRoundProof::deserialize_with_mode(
                        &mut limited,
                        compress,
                        validate,
                    )?);
                    stage1_sumcheck = Some(SumcheckInstanceProof::deserialize_with_mode(
                        &mut limited,
                        compress,
                        validate,
                    )?);
                }
                TAG_STAGE2 => {
                    if stage2_uni.is_some() {
                        return Err(SerializationError::InvalidData);
                    }
                    stage2_uni = Some(UniSkipFirstRoundProof::deserialize_with_mode(
                        &mut limited,
                        compress,
                        validate,
                    )?);
                    stage2_sumcheck = Some(SumcheckInstanceProof::deserialize_with_mode(
                        &mut limited,
                        compress,
                        validate,
                    )?);
                }
                TAG_STAGE3 => {
                    if stage3_sumcheck.is_some() {
                        return Err(SerializationError::InvalidData);
                    }
                    stage3_sumcheck = Some(SumcheckInstanceProof::deserialize_with_mode(
                        &mut limited,
                        compress,
                        validate,
                    )?);
                }
                TAG_STAGE4 => {
                    if stage4_sumcheck.is_some() {
                        return Err(SerializationError::InvalidData);
                    }
                    stage4_sumcheck = Some(SumcheckInstanceProof::deserialize_with_mode(
                        &mut limited,
                        compress,
                        validate,
                    )?);
                }
                TAG_STAGE5 => {
                    if stage5_sumcheck.is_some() {
                        return Err(SerializationError::InvalidData);
                    }
                    stage5_sumcheck = Some(SumcheckInstanceProof::deserialize_with_mode(
                        &mut limited,
                        compress,
                        validate,
                    )?);
                }
                TAG_STAGE6 => {
                    if stage6_sumcheck.is_some() {
                        return Err(SerializationError::InvalidData);
                    }
                    stage6_sumcheck = Some(SumcheckInstanceProof::deserialize_with_mode(
                        &mut limited,
                        compress,
                        validate,
                    )?);
                }
                TAG_STAGE7 => {
                    if stage7_sumcheck.is_some() {
                        return Err(SerializationError::InvalidData);
                    }
                    stage7_sumcheck = Some(SumcheckInstanceProof::deserialize_with_mode(
                        &mut limited,
                        compress,
                        validate,
                    )?);
                }
                TAG_JOINT_OPENING => {
                    if joint_opening_proof.is_some() {
                        return Err(SerializationError::InvalidData);
                    }
                    joint_opening_proof = Some(PCS::Proof::deserialize_with_mode(
                        &mut limited,
                        compress,
                        validate,
                    )?);
                }
                _ => return Err(SerializationError::InvalidData),
            }

            if limited.limit() != 0 {
                return Err(SerializationError::InvalidData);
            }
        }

        Ok(Self {
            opening_claims: opening_claims.ok_or(SerializationError::InvalidData)?,
            commitments: commitments.ok_or(SerializationError::InvalidData)?,
            stage1_uni_skip_first_round_proof: stage1_uni.ok_or(SerializationError::InvalidData)?,
            stage1_sumcheck_proof: stage1_sumcheck.ok_or(SerializationError::InvalidData)?,
            stage2_uni_skip_first_round_proof: stage2_uni.ok_or(SerializationError::InvalidData)?,
            stage2_sumcheck_proof: stage2_sumcheck.ok_or(SerializationError::InvalidData)?,
            stage3_sumcheck_proof: stage3_sumcheck.ok_or(SerializationError::InvalidData)?,
            stage4_sumcheck_proof: stage4_sumcheck.ok_or(SerializationError::InvalidData)?,
            stage5_sumcheck_proof: stage5_sumcheck.ok_or(SerializationError::InvalidData)?,
            stage6_sumcheck_proof: stage6_sumcheck.ok_or(SerializationError::InvalidData)?,
            stage7_sumcheck_proof: stage7_sumcheck.ok_or(SerializationError::InvalidData)?,
            joint_opening_proof: joint_opening_proof.ok_or(SerializationError::InvalidData)?,
            untrusted_advice_commitment: untrusted_advice_commitment
                .ok_or(SerializationError::InvalidData)?,
            trace_length: trace_length.ok_or(SerializationError::InvalidData)?,
            ram_K: ram_K.ok_or(SerializationError::InvalidData)?,
            bytecode_K: bytecode_K.ok_or(SerializationError::InvalidData)?,
            rw_config: rw_config.ok_or(SerializationError::InvalidData)?,
            one_hot_config: one_hot_config.ok_or(SerializationError::InvalidData)?,
            dory_layout: dory_layout.ok_or(SerializationError::InvalidData)?,
        })
    }
}

impl CanonicalSerialize for DoryLayout {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        u8::from(*self).serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        u8::from(*self).serialized_size(compress)
    }
}

impl Valid for DoryLayout {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for DoryLayout {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = u8::deserialize_with_mode(reader, compress, validate)?;
        if value > 1 {
            return Err(SerializationError::InvalidData);
        }
        Ok(DoryLayout::from(value))
    }
}

pub struct Claims<F: JoltField>(pub Openings<F>);

impl<F: JoltField> CanonicalSerialize for Claims<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.len().serialize_with_mode(&mut writer, compress)?;
        for (key, (_opening_point, claim)) in self.0.iter() {
            key.serialize_with_mode(&mut writer, compress)?;
            claim.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut size = self.0.len().serialized_size(compress);
        for (key, (_opening_point, claim)) in self.0.iter() {
            size += key.serialized_size(compress);
            size += claim.serialized_size(compress);
        }
        size
    }
}

impl<F: JoltField> Valid for Claims<F> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: JoltField> CanonicalDeserialize for Claims<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let size = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let mut claims = BTreeMap::new();
        for _ in 0..size {
            let key = OpeningId::deserialize_with_mode(&mut reader, compress, validate)?;
            let claim = F::deserialize_with_mode(&mut reader, compress, validate)?;
            claims.insert(key, (OpeningPoint::default(), claim));
        }

        Ok(Claims(claims))
    }
}

// OpeningId wire encoding (packed, self-describing):
//
// Header byte:
//   bits[7..6] = kind (0..=3)
//   bits[5..0] = sumcheck_id if < 63, otherwise 63 as an escape
//
// If bits[5..0] == 63:
//   next bytes = varint u64 sumcheck_id (must fit in u8 for current SumcheckId)
//
// If kind indicates a polynomial:
//   append polynomial id bytes (CommittedPolynomial / VirtualPolynomial)
//
// This does NOT depend on `SumcheckId::COUNT` range boundaries.
const OPENING_ID_KIND_UNTRUSTED_ADVICE: u8 = 0;
const OPENING_ID_KIND_TRUSTED_ADVICE: u8 = 1;
const OPENING_ID_KIND_COMMITTED: u8 = 2;
const OPENING_ID_KIND_VIRTUAL: u8 = 3;
const OPENING_ID_SUMCHECK_ESCAPE: u8 = 63;

impl CanonicalSerialize for OpeningId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let (kind, sumcheck_u64, poly) = match *self {
            OpeningId::UntrustedAdvice(sumcheck_id) => {
                (OPENING_ID_KIND_UNTRUSTED_ADVICE, sumcheck_id as u64, None)
            }
            OpeningId::TrustedAdvice(sumcheck_id) => {
                (OPENING_ID_KIND_TRUSTED_ADVICE, sumcheck_id as u64, None)
            }
            OpeningId::Polynomial(PolynomialId::Committed(committed_polynomial), sumcheck_id) => (
                OPENING_ID_KIND_COMMITTED,
                sumcheck_id as u64,
                Some(PolynomialId::Committed(committed_polynomial)),
            ),
            OpeningId::Polynomial(PolynomialId::Virtual(virtual_polynomial), sumcheck_id) => (
                OPENING_ID_KIND_VIRTUAL,
                sumcheck_id as u64,
                Some(PolynomialId::Virtual(virtual_polynomial)),
            ),
        };

        let header = if sumcheck_u64 < OPENING_ID_SUMCHECK_ESCAPE as u64 {
            (kind << 6) | (sumcheck_u64 as u8)
        } else {
            (kind << 6) | OPENING_ID_SUMCHECK_ESCAPE
        };
        header.serialize_with_mode(&mut writer, compress)?;
        if (header & 0x3F) == OPENING_ID_SUMCHECK_ESCAPE {
            write_varint_u64(&mut writer, sumcheck_u64)?;
        }
        if let Some(poly) = poly {
            match poly {
                PolynomialId::Committed(p) => p.serialize_with_mode(&mut writer, compress)?,
                PolynomialId::Virtual(p) => p.serialize_with_mode(&mut writer, compress)?,
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let sumcheck_u64 = match self {
            OpeningId::UntrustedAdvice(sumcheck_id)
            | OpeningId::TrustedAdvice(sumcheck_id)
            | OpeningId::Polynomial(_, sumcheck_id) => *sumcheck_id as u64,
        };
        let header_len = 1usize;
        let sumcheck_ext_len = if sumcheck_u64 < OPENING_ID_SUMCHECK_ESCAPE as u64 {
            0usize
        } else {
            transport::varint_u64_len(sumcheck_u64)
        };
        let poly_len = match self {
            OpeningId::UntrustedAdvice(_) | OpeningId::TrustedAdvice(_) => 0,
            OpeningId::Polynomial(PolynomialId::Committed(p), _) => p.serialized_size(compress),
            OpeningId::Polynomial(PolynomialId::Virtual(p), _) => p.serialized_size(compress),
        };
        header_len + sumcheck_ext_len + poly_len
    }
}

impl Valid for OpeningId {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for OpeningId {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let header = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let kind = header >> 6;
        let small = header & 0x3F;

        let sumcheck_u64 = if small == OPENING_ID_SUMCHECK_ESCAPE {
            read_varint_u64(&mut reader)?
        } else {
            small as u64
        };

        let sumcheck_u8 =
            u8::try_from(sumcheck_u64).map_err(|_| SerializationError::InvalidData)?;
        let sumcheck_id =
            SumcheckId::from_u8(sumcheck_u8).ok_or(SerializationError::InvalidData)?;

        match kind {
            OPENING_ID_KIND_UNTRUSTED_ADVICE => Ok(OpeningId::UntrustedAdvice(sumcheck_id)),
            OPENING_ID_KIND_TRUSTED_ADVICE => Ok(OpeningId::TrustedAdvice(sumcheck_id)),
            OPENING_ID_KIND_COMMITTED => {
                let polynomial =
                    CommittedPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::Polynomial(
                    PolynomialId::Committed(polynomial),
                    sumcheck_id,
                ))
            }
            OPENING_ID_KIND_VIRTUAL => {
                let polynomial =
                    VirtualPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::Polynomial(
                    PolynomialId::Virtual(polynomial),
                    sumcheck_id,
                ))
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl CanonicalSerialize for CommittedPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::RdInc => 0u8.serialize_with_mode(writer, compress),
            Self::RamInc => 1u8.serialize_with_mode(writer, compress),
            Self::InstructionRa(i) => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::BytecodeRa(i) => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::RamRa(i) => {
                4u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::TrustedAdvice => 5u8.serialize_with_mode(writer, compress),
            Self::UntrustedAdvice => 6u8.serialize_with_mode(writer, compress),
        }
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        match self {
            Self::RdInc | Self::RamInc | Self::TrustedAdvice | Self::UntrustedAdvice => 1,
            Self::InstructionRa(_) | Self::BytecodeRa(_) | Self::RamRa(_) => 2,
        }
    }
}

impl Valid for CommittedPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for CommittedPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(
            match u8::deserialize_with_mode(&mut reader, compress, validate)? {
                0 => Self::RdInc,
                1 => Self::RamInc,
                2 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::InstructionRa(i as usize)
                }
                3 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::BytecodeRa(i as usize)
                }
                4 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::RamRa(i as usize)
                }
                5 => Self::TrustedAdvice,
                6 => Self::UntrustedAdvice,
                _ => return Err(SerializationError::InvalidData),
            },
        )
    }
}

impl CanonicalSerialize for VirtualPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::PC => 0u8.serialize_with_mode(&mut writer, compress),
            Self::UnexpandedPC => 1u8.serialize_with_mode(&mut writer, compress),
            Self::NextPC => 2u8.serialize_with_mode(&mut writer, compress),
            Self::NextUnexpandedPC => 3u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsNoop => 4u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsVirtual => 5u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsFirstInSequence => 6u8.serialize_with_mode(&mut writer, compress),
            Self::LeftLookupOperand => 7u8.serialize_with_mode(&mut writer, compress),
            Self::RightLookupOperand => 8u8.serialize_with_mode(&mut writer, compress),
            Self::LeftInstructionInput => 9u8.serialize_with_mode(&mut writer, compress),
            Self::RightInstructionInput => 10u8.serialize_with_mode(&mut writer, compress),
            Self::Product => 11u8.serialize_with_mode(&mut writer, compress),
            Self::ShouldJump => 12u8.serialize_with_mode(&mut writer, compress),
            Self::ShouldBranch => 13u8.serialize_with_mode(&mut writer, compress),
            Self::WritePCtoRD => 14u8.serialize_with_mode(&mut writer, compress),
            Self::WriteLookupOutputToRD => 15u8.serialize_with_mode(&mut writer, compress),
            Self::Rd => 16u8.serialize_with_mode(&mut writer, compress),
            Self::Imm => 17u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Value => 18u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Value => 19u8.serialize_with_mode(&mut writer, compress),
            Self::RdWriteValue => 20u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Ra => 21u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Ra => 22u8.serialize_with_mode(&mut writer, compress),
            Self::RdWa => 23u8.serialize_with_mode(&mut writer, compress),
            Self::LookupOutput => 24u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRaf => 25u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRafFlag => 26u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRa(i) => {
                27u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::RegistersVal => 28u8.serialize_with_mode(&mut writer, compress),
            Self::RamAddress => 29u8.serialize_with_mode(&mut writer, compress),
            Self::RamRa => 30u8.serialize_with_mode(&mut writer, compress),
            Self::RamReadValue => 31u8.serialize_with_mode(&mut writer, compress),
            Self::RamWriteValue => 32u8.serialize_with_mode(&mut writer, compress),
            Self::RamVal => 33u8.serialize_with_mode(&mut writer, compress),
            Self::RamValInit => 34u8.serialize_with_mode(&mut writer, compress),
            Self::RamValFinal => 35u8.serialize_with_mode(&mut writer, compress),
            Self::RamHammingWeight => 36u8.serialize_with_mode(&mut writer, compress),
            Self::UnivariateSkip => 37u8.serialize_with_mode(&mut writer, compress),
            Self::OpFlags(flags) => {
                38u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::InstructionFlags(flags) => {
                39u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::LookupTableFlag(flag) => {
                40u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flag).unwrap()).serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        match self {
            Self::PC
            | Self::UnexpandedPC
            | Self::NextPC
            | Self::NextUnexpandedPC
            | Self::NextIsNoop
            | Self::NextIsVirtual
            | Self::NextIsFirstInSequence
            | Self::LeftLookupOperand
            | Self::RightLookupOperand
            | Self::LeftInstructionInput
            | Self::RightInstructionInput
            | Self::Product
            | Self::ShouldJump
            | Self::ShouldBranch
            | Self::WritePCtoRD
            | Self::WriteLookupOutputToRD
            | Self::Rd
            | Self::Imm
            | Self::Rs1Value
            | Self::Rs2Value
            | Self::RdWriteValue
            | Self::Rs1Ra
            | Self::Rs2Ra
            | Self::RdWa
            | Self::LookupOutput
            | Self::InstructionRaf
            | Self::InstructionRafFlag
            | Self::RegistersVal
            | Self::RamAddress
            | Self::RamRa
            | Self::RamReadValue
            | Self::RamWriteValue
            | Self::RamVal
            | Self::RamValInit
            | Self::RamValFinal
            | Self::RamHammingWeight
            | Self::UnivariateSkip => 1,
            Self::InstructionRa(_)
            | Self::OpFlags(_)
            | Self::InstructionFlags(_)
            | Self::LookupTableFlag(_) => 2,
        }
    }
}

impl Valid for VirtualPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for VirtualPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(
            match u8::deserialize_with_mode(&mut reader, compress, validate)? {
                0 => Self::PC,
                1 => Self::UnexpandedPC,
                2 => Self::NextPC,
                3 => Self::NextUnexpandedPC,
                4 => Self::NextIsNoop,
                5 => Self::NextIsVirtual,
                6 => Self::NextIsFirstInSequence,
                7 => Self::LeftLookupOperand,
                8 => Self::RightLookupOperand,
                9 => Self::LeftInstructionInput,
                10 => Self::RightInstructionInput,
                11 => Self::Product,
                12 => Self::ShouldJump,
                13 => Self::ShouldBranch,
                14 => Self::WritePCtoRD,
                15 => Self::WriteLookupOutputToRD,
                16 => Self::Rd,
                17 => Self::Imm,
                18 => Self::Rs1Value,
                19 => Self::Rs2Value,
                20 => Self::RdWriteValue,
                21 => Self::Rs1Ra,
                22 => Self::Rs2Ra,
                23 => Self::RdWa,
                24 => Self::LookupOutput,
                25 => Self::InstructionRaf,
                26 => Self::InstructionRafFlag,
                27 => {
                    let i = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::InstructionRa(i as usize)
                }
                28 => Self::RegistersVal,
                29 => Self::RamAddress,
                30 => Self::RamRa,
                31 => Self::RamReadValue,
                32 => Self::RamWriteValue,
                33 => Self::RamVal,
                34 => Self::RamValInit,
                35 => Self::RamValFinal,
                36 => Self::RamHammingWeight,
                37 => Self::UnivariateSkip,
                38 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = CircuitFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::OpFlags(flags)
                }
                39 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = InstructionFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::InstructionFlags(flags)
                }
                40 => {
                    let flag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::LookupTableFlag(flag as usize)
                }
                _ => return Err(SerializationError::InvalidData),
            },
        )
    }
}

pub fn serialize_and_print_size(
    item_name: &str,
    file_name: &str,
    item: &impl CanonicalSerialize,
) -> Result<(), SerializationError> {
    let mut file = File::create(file_name)?;
    item.serialize_compressed(&mut file)?;
    let file_size_bytes = file.metadata()?.len();
    let file_size_kb = file_size_bytes as f64 / 1024.0;
    tracing::info!("{item_name} Written to {file_name}");
    tracing::info!("{item_name} size: {file_size_kb:.1} kB");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::opening_proof::{OpeningId, PolynomialId, SumcheckId};
    use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};

    #[test]
    fn opening_id_header_is_packed_common_case() {
        let id = OpeningId::UntrustedAdvice(SumcheckId::SpartanOuter);
        let mut bytes = Vec::new();
        id.serialize_compressed(&mut bytes).unwrap();
        assert_eq!(bytes.len(), 1);
        let expected = (OPENING_ID_KIND_UNTRUSTED_ADVICE << 6) | (SumcheckId::SpartanOuter as u8);
        assert_eq!(bytes[0], expected);

        let id = OpeningId::Polynomial(
            PolynomialId::Committed(CommittedPolynomial::RdInc),
            SumcheckId::SpartanOuter,
        );
        let mut bytes = Vec::new();
        id.serialize_compressed(&mut bytes).unwrap();
        assert!(bytes.len() >= 2); // header + poly id
        assert_eq!(bytes[0] >> 6, OPENING_ID_KIND_COMMITTED);
        assert_eq!(bytes[0] & 0x3F, SumcheckId::SpartanOuter as u8);

        let id = OpeningId::Polynomial(
            PolynomialId::Virtual(VirtualPolynomial::PC),
            SumcheckId::SpartanOuter,
        );
        let mut bytes = Vec::new();
        id.serialize_compressed(&mut bytes).unwrap();
        assert!(bytes.len() >= 2); // header + poly id
        assert_eq!(bytes[0] >> 6, OPENING_ID_KIND_VIRTUAL);
        assert_eq!(bytes[0] & 0x3F, SumcheckId::SpartanOuter as u8);
    }

    #[test]
    fn opening_id_roundtrips() {
        let cases = [
            OpeningId::UntrustedAdvice(SumcheckId::SpartanOuter),
            OpeningId::TrustedAdvice(SumcheckId::SpartanOuter),
            OpeningId::Polynomial(
                PolynomialId::Committed(CommittedPolynomial::RdInc),
                SumcheckId::SpartanOuter,
            ),
            OpeningId::Polynomial(
                PolynomialId::Virtual(VirtualPolynomial::PC),
                SumcheckId::SpartanOuter,
            ),
        ];
        for id in cases {
            let mut bytes = Vec::new();
            id.serialize_compressed(&mut bytes).unwrap();
            let decoded = OpeningId::deserialize_compressed(bytes.as_slice()).unwrap();
            assert_eq!(decoded, id);
        }
    }

    #[test]
    fn proof_signature_is_required_and_unknown_tags_reject() {
        // Missing sections should reject cleanly (after signature).
        let mut just_sig = Vec::new();
        just_sig.extend_from_slice(PROOF_SIGNATURE);
        let res = crate::zkvm::RV64IMACProof::deserialize_with_mode(
            std::io::Cursor::new(&just_sig),
            Compress::Yes,
            Validate::Yes,
        )
        .map(|_| ());
        match res {
            Err(SerializationError::InvalidData) | Err(SerializationError::IoError(_)) => {}
            _ => panic!("expected decode error"),
        }

        // Unknown tag should reject.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(PROOF_SIGNATURE);
        transport::write_frame_header(&mut bytes, 99, 0).unwrap();
        let res = crate::zkvm::RV64IMACProof::deserialize_with_mode(
            std::io::Cursor::new(&bytes),
            Compress::Yes,
            Validate::Yes,
        )
        .map(|_| ());
        match res {
            Err(SerializationError::InvalidData) | Err(SerializationError::IoError(_)) => {}
            _ => panic!("expected decode error"),
        }
    }
}
