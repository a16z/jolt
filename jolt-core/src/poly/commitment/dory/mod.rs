//! Dory polynomial commitment scheme
//!
//! This module provides a Dory commitment scheme implementation that bridges
//! between Jolt's types and final-dory's arkworks backend.

use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};

mod commitment_scheme;
mod dory_globals;
mod jolt_dory_routines;
mod wrappers;

#[cfg(test)]
mod tests;

pub use commitment_scheme::DoryCommitmentScheme;
pub use dory_globals::{DoryContext, DoryGlobals, DoryLayout};
pub use jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
pub use wrappers::{
    ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
    JoltFieldWrapper, BN254,
};

/// Deserializes a Dory opening proof with fine-grained cycle markers.
///
/// This is intended for recursion/guest profiling where deserialization costs matter.
/// On non-RISC-V targets, the cycle markers are no-ops.
pub fn deserialize_ark_dory_proof_marked<R: std::io::Read>(
    reader: &mut R,
    compress: ark_serialize::Compress,
    validate: ark_serialize::Validate,
) -> Result<ArkDoryProof, ark_serialize::SerializationError> {
    use ark_serialize::CanonicalDeserialize;
    use dory::messages::{
        FirstReduceMessage, ScalarProductMessage, SecondReduceMessage, VMVMessage,
    };
    use jolt_platform::{end_cycle_tracking, start_cycle_tracking};

    start_cycle_tracking("deserialize dory_proof/vmv_message");
    let c = {
        start_cycle_tracking("deserialize dory_proof/vmv_message.c (GT)");
        let v = CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        end_cycle_tracking("deserialize dory_proof/vmv_message.c (GT)");
        v
    };
    let d2 = {
        start_cycle_tracking("deserialize dory_proof/vmv_message.d2 (GT)");
        let v = CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        end_cycle_tracking("deserialize dory_proof/vmv_message.d2 (GT)");
        v
    };
    let e1 = {
        start_cycle_tracking("deserialize dory_proof/vmv_message.e1 (G1)");
        let v = CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        end_cycle_tracking("deserialize dory_proof/vmv_message.e1 (G1)");
        v
    };
    let vmv_message = VMVMessage { c, d2, e1 };
    end_cycle_tracking("deserialize dory_proof/vmv_message");

    start_cycle_tracking("deserialize dory_proof/num_rounds");
    let num_rounds =
        <u32 as CanonicalDeserialize>::deserialize_with_mode(&mut *reader, compress, validate)?
            as usize;
    end_cycle_tracking("deserialize dory_proof/num_rounds");

    start_cycle_tracking("deserialize dory_proof/first_messages");
    let mut first_messages = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        // Many GT deserializations here; keep a marker per message for readability.
        start_cycle_tracking("deserialize dory_proof/first_messages/round");

        start_cycle_tracking("deserialize dory_proof/first_messages/round/gt_fields");
        let d1_left =
            CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        let d1_right =
            CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        let d2_left =
            CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        let d2_right =
            CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        end_cycle_tracking("deserialize dory_proof/first_messages/round/gt_fields");

        start_cycle_tracking("deserialize dory_proof/first_messages/round/e1_beta (G1)");
        let e1_beta =
            CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        end_cycle_tracking("deserialize dory_proof/first_messages/round/e1_beta (G1)");

        start_cycle_tracking("deserialize dory_proof/first_messages/round/e2_beta (G2)");
        let e2_beta =
            CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        end_cycle_tracking("deserialize dory_proof/first_messages/round/e2_beta (G2)");

        first_messages.push(FirstReduceMessage {
            d1_left,
            d1_right,
            d2_left,
            d2_right,
            e1_beta,
            e2_beta,
        });
        end_cycle_tracking("deserialize dory_proof/first_messages/round");
    }
    end_cycle_tracking("deserialize dory_proof/first_messages");

    start_cycle_tracking("deserialize dory_proof/second_messages");
    let mut second_messages = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        start_cycle_tracking("deserialize dory_proof/second_messages/round");

        start_cycle_tracking("deserialize dory_proof/second_messages/round/gt_fields");
        let c_plus = CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        let c_minus =
            CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        end_cycle_tracking("deserialize dory_proof/second_messages/round/gt_fields");

        start_cycle_tracking("deserialize dory_proof/second_messages/round/g1_fields");
        let e1_plus =
            CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        let e1_minus =
            CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        end_cycle_tracking("deserialize dory_proof/second_messages/round/g1_fields");

        start_cycle_tracking("deserialize dory_proof/second_messages/round/g2_fields");
        let e2_plus =
            CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        let e2_minus =
            CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
        end_cycle_tracking("deserialize dory_proof/second_messages/round/g2_fields");

        second_messages.push(SecondReduceMessage {
            c_plus,
            c_minus,
            e1_plus,
            e1_minus,
            e2_plus,
            e2_minus,
        });
        end_cycle_tracking("deserialize dory_proof/second_messages/round");
    }
    end_cycle_tracking("deserialize dory_proof/second_messages");

    start_cycle_tracking("deserialize dory_proof/final_message");
    start_cycle_tracking("deserialize dory_proof/final_message.e1 (G1)");
    let e1 = CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
    end_cycle_tracking("deserialize dory_proof/final_message.e1 (G1)");
    start_cycle_tracking("deserialize dory_proof/final_message.e2 (G2)");
    let e2 = CanonicalDeserialize::deserialize_with_mode(&mut *reader, compress, validate)?;
    end_cycle_tracking("deserialize dory_proof/final_message.e2 (G2)");
    let final_message = ScalarProductMessage { e1, e2 };
    end_cycle_tracking("deserialize dory_proof/final_message");

    start_cycle_tracking("deserialize dory_proof/nu_sigma");
    let nu = <u32 as CanonicalDeserialize>::deserialize_with_mode(&mut *reader, compress, validate)?
        as usize;
    let sigma =
        <u32 as CanonicalDeserialize>::deserialize_with_mode(&mut *reader, compress, validate)?
            as usize;
    end_cycle_tracking("deserialize dory_proof/nu_sigma");

    Ok(ArkDoryProof {
        vmv_message,
        first_messages,
        second_messages,
        final_message,
        nu,
        sigma,
    })
}

/// Deserializes a Dory verifier setup with fine-grained cycle markers.
///
/// This is useful to pinpoint costs inside `PCS::VerifierSetup` deserialization
/// (which can become significant as recursion verification gets optimized).
pub fn deserialize_arkworks_verifier_setup_marked<R: std::io::Read>(
    reader: &mut R,
    compress: ark_serialize::Compress,
    validate: ark_serialize::Validate,
) -> Result<ArkworksVerifierSetup, ark_serialize::SerializationError> {
    use dory::primitives::serialization::{
        Compress as DoryCompress, DoryDeserialize, Validate as DoryValidate,
    };
    use dory::setup::VerifierSetup;
    use jolt_platform::{end_cycle_tracking, start_cycle_tracking};

    let dory_compress = match compress {
        ark_serialize::Compress::Yes => DoryCompress::Yes,
        ark_serialize::Compress::No => DoryCompress::No,
    };
    let dory_validate = match validate {
        ark_serialize::Validate::Yes => DoryValidate::Yes,
        ark_serialize::Validate::No => DoryValidate::No,
    };

    start_cycle_tracking("deserialize dory_verifier_setup/delta_1l");
    let delta_1l: Vec<ArkGT> =
        DoryDeserialize::deserialize_with_mode(&mut *reader, dory_compress, dory_validate)
            .map_err(|_| ark_serialize::SerializationError::InvalidData)?;
    end_cycle_tracking("deserialize dory_verifier_setup/delta_1l");

    start_cycle_tracking("deserialize dory_verifier_setup/delta_1r");
    let delta_1r: Vec<ArkGT> =
        DoryDeserialize::deserialize_with_mode(&mut *reader, dory_compress, dory_validate)
            .map_err(|_| ark_serialize::SerializationError::InvalidData)?;
    end_cycle_tracking("deserialize dory_verifier_setup/delta_1r");

    start_cycle_tracking("deserialize dory_verifier_setup/delta_2l");
    let delta_2l: Vec<ArkGT> =
        DoryDeserialize::deserialize_with_mode(&mut *reader, dory_compress, dory_validate)
            .map_err(|_| ark_serialize::SerializationError::InvalidData)?;
    end_cycle_tracking("deserialize dory_verifier_setup/delta_2l");

    start_cycle_tracking("deserialize dory_verifier_setup/delta_2r");
    let delta_2r: Vec<ArkGT> =
        DoryDeserialize::deserialize_with_mode(&mut *reader, dory_compress, dory_validate)
            .map_err(|_| ark_serialize::SerializationError::InvalidData)?;
    end_cycle_tracking("deserialize dory_verifier_setup/delta_2r");

    start_cycle_tracking("deserialize dory_verifier_setup/chi");
    let chi: Vec<ArkGT> =
        DoryDeserialize::deserialize_with_mode(&mut *reader, dory_compress, dory_validate)
            .map_err(|_| ark_serialize::SerializationError::InvalidData)?;
    end_cycle_tracking("deserialize dory_verifier_setup/chi");

    start_cycle_tracking("deserialize dory_verifier_setup/g1_0 (G1)");
    let g1_0: ArkG1 =
        DoryDeserialize::deserialize_with_mode(&mut *reader, dory_compress, dory_validate)
            .map_err(|_| ark_serialize::SerializationError::InvalidData)?;
    end_cycle_tracking("deserialize dory_verifier_setup/g1_0 (G1)");

    start_cycle_tracking("deserialize dory_verifier_setup/g2_0 (G2)");
    let g2_0: ArkG2 =
        DoryDeserialize::deserialize_with_mode(&mut *reader, dory_compress, dory_validate)
            .map_err(|_| ark_serialize::SerializationError::InvalidData)?;
    end_cycle_tracking("deserialize dory_verifier_setup/g2_0 (G2)");

    start_cycle_tracking("deserialize dory_verifier_setup/h1 (G1)");
    let h1: ArkG1 =
        DoryDeserialize::deserialize_with_mode(&mut *reader, dory_compress, dory_validate)
            .map_err(|_| ark_serialize::SerializationError::InvalidData)?;
    end_cycle_tracking("deserialize dory_verifier_setup/h1 (G1)");

    start_cycle_tracking("deserialize dory_verifier_setup/h2 (G2)");
    let h2: ArkG2 =
        DoryDeserialize::deserialize_with_mode(&mut *reader, dory_compress, dory_validate)
            .map_err(|_| ark_serialize::SerializationError::InvalidData)?;
    end_cycle_tracking("deserialize dory_verifier_setup/h2 (G2)");

    start_cycle_tracking("deserialize dory_verifier_setup/ht (GT)");
    let ht: ArkGT =
        DoryDeserialize::deserialize_with_mode(&mut *reader, dory_compress, dory_validate)
            .map_err(|_| ark_serialize::SerializationError::InvalidData)?;
    end_cycle_tracking("deserialize dory_verifier_setup/ht (GT)");

    start_cycle_tracking("deserialize dory_verifier_setup/max_log_n (usize)");
    let max_log_n: usize =
        DoryDeserialize::deserialize_with_mode(&mut *reader, dory_compress, dory_validate)
            .map_err(|_| ark_serialize::SerializationError::InvalidData)?;
    end_cycle_tracking("deserialize dory_verifier_setup/max_log_n (usize)");

    Ok(ArkworksVerifierSetup(VerifierSetup::<BN254> {
        delta_1l,
        delta_1r,
        delta_2l,
        delta_2r,
        chi,
        g1_0,
        g2_0,
        h1,
        h2,
        ht,
        max_log_n,
    }))
}

// -------------------------------------------------------------------------------------------------
// Guest (decompressed) encoding helpers
// -------------------------------------------------------------------------------------------------

impl crate::zkvm::guest_serde::GuestSerialize for ArkGT {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        crate::zkvm::guest_serde::GuestSerialize::guest_serialize(&self.0, w)
    }
}
impl crate::zkvm::guest_serde::GuestDeserialize for ArkGT {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(ArkGT(
            crate::zkvm::guest_serde::GuestDeserialize::guest_deserialize(r)?,
        ))
    }
}

impl crate::zkvm::guest_serde::GuestSerialize for ArkG1 {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        use ark_ec::CurveGroup;
        let aff = self.0.into_affine();
        crate::zkvm::guest_serde::GuestSerialize::guest_serialize(&aff, w)
    }
}
impl crate::zkvm::guest_serde::GuestDeserialize for ArkG1 {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        use ark_ec::AffineRepr;
        let aff: ark_bn254::g1::G1Affine =
            crate::zkvm::guest_serde::GuestDeserialize::guest_deserialize(r)?;
        Ok(ArkG1(aff.into_group()))
    }
}

impl crate::zkvm::guest_serde::GuestSerialize for ArkG2 {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        use ark_ec::CurveGroup;
        let aff = self.0.into_affine();
        crate::zkvm::guest_serde::GuestSerialize::guest_serialize(&aff, w)
    }
}
impl crate::zkvm::guest_serde::GuestDeserialize for ArkG2 {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        use ark_ec::AffineRepr;
        let aff: ark_bn254::g2::G2Affine =
            crate::zkvm::guest_serde::GuestDeserialize::guest_deserialize(r)?;
        Ok(ArkG2(aff.into_group()))
    }
}

impl crate::zkvm::guest_serde::GuestSerialize for ArkworksVerifierSetup {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        let s = &self.0;
        s.delta_1l.guest_serialize(w)?;
        s.delta_1r.guest_serialize(w)?;
        s.delta_2l.guest_serialize(w)?;
        s.delta_2r.guest_serialize(w)?;
        s.chi.guest_serialize(w)?;
        s.g1_0.guest_serialize(w)?;
        s.g2_0.guest_serialize(w)?;
        s.h1.guest_serialize(w)?;
        s.h2.guest_serialize(w)?;
        s.ht.guest_serialize(w)?;
        s.max_log_n.guest_serialize(w)?;
        Ok(())
    }
}
impl crate::zkvm::guest_serde::GuestDeserialize for ArkworksVerifierSetup {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        use dory::setup::VerifierSetup;
        Ok(ArkworksVerifierSetup(VerifierSetup::<BN254> {
            delta_1l: Vec::<ArkGT>::guest_deserialize(r)?,
            delta_1r: Vec::<ArkGT>::guest_deserialize(r)?,
            delta_2l: Vec::<ArkGT>::guest_deserialize(r)?,
            delta_2r: Vec::<ArkGT>::guest_deserialize(r)?,
            chi: Vec::<ArkGT>::guest_deserialize(r)?,
            g1_0: ArkG1::guest_deserialize(r)?,
            g2_0: ArkG2::guest_deserialize(r)?,
            h1: ArkG1::guest_deserialize(r)?,
            h2: ArkG2::guest_deserialize(r)?,
            ht: ArkGT::guest_deserialize(r)?,
            max_log_n: usize::guest_deserialize(r)?,
        }))
    }
}

impl crate::zkvm::guest_serde::GuestSerialize for ArkDoryProof {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.vmv_message.guest_serialize(w)?;
        self.first_messages.guest_serialize(w)?;
        self.second_messages.guest_serialize(w)?;
        self.final_message.guest_serialize(w)?;
        (self.nu as u32).guest_serialize(w)?;
        (self.sigma as u32).guest_serialize(w)?;
        Ok(())
    }
}
impl crate::zkvm::guest_serde::GuestDeserialize for ArkDoryProof {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(ArkDoryProof {
            vmv_message: dory::messages::VMVMessage::<ArkG1, ArkGT>::guest_deserialize(r)?,
            first_messages:
                Vec::<dory::messages::FirstReduceMessage<ArkG1, ArkG2, ArkGT>>::guest_deserialize(
                    r,
                )?,
            second_messages:
                Vec::<dory::messages::SecondReduceMessage<ArkG1, ArkG2, ArkGT>>::guest_deserialize(
                    r,
                )?,
            final_message: dory::messages::ScalarProductMessage::<ArkG1, ArkG2>::guest_deserialize(
                r,
            )?,
            nu: u32::guest_deserialize(r)? as usize,
            sigma: u32::guest_deserialize(r)? as usize,
        })
    }
}

impl<G1, GT> crate::zkvm::guest_serde::GuestSerialize for dory::messages::VMVMessage<G1, GT>
where
    G1: crate::zkvm::guest_serde::GuestSerialize,
    GT: crate::zkvm::guest_serde::GuestSerialize,
{
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.c.guest_serialize(w)?;
        self.d2.guest_serialize(w)?;
        self.e1.guest_serialize(w)?;
        Ok(())
    }
}
impl<G1, GT> crate::zkvm::guest_serde::GuestDeserialize for dory::messages::VMVMessage<G1, GT>
where
    G1: crate::zkvm::guest_serde::GuestDeserialize,
    GT: crate::zkvm::guest_serde::GuestDeserialize,
{
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(dory::messages::VMVMessage {
            c: GT::guest_deserialize(r)?,
            d2: GT::guest_deserialize(r)?,
            e1: G1::guest_deserialize(r)?,
        })
    }
}

impl<G1, G2, GT> crate::zkvm::guest_serde::GuestSerialize
    for dory::messages::FirstReduceMessage<G1, G2, GT>
where
    G1: crate::zkvm::guest_serde::GuestSerialize,
    G2: crate::zkvm::guest_serde::GuestSerialize,
    GT: crate::zkvm::guest_serde::GuestSerialize,
{
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.d1_left.guest_serialize(w)?;
        self.d1_right.guest_serialize(w)?;
        self.d2_left.guest_serialize(w)?;
        self.d2_right.guest_serialize(w)?;
        self.e1_beta.guest_serialize(w)?;
        self.e2_beta.guest_serialize(w)?;
        Ok(())
    }
}
impl<G1, G2, GT> crate::zkvm::guest_serde::GuestDeserialize
    for dory::messages::FirstReduceMessage<G1, G2, GT>
where
    G1: crate::zkvm::guest_serde::GuestDeserialize,
    G2: crate::zkvm::guest_serde::GuestDeserialize,
    GT: crate::zkvm::guest_serde::GuestDeserialize,
{
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(dory::messages::FirstReduceMessage {
            d1_left: GT::guest_deserialize(r)?,
            d1_right: GT::guest_deserialize(r)?,
            d2_left: GT::guest_deserialize(r)?,
            d2_right: GT::guest_deserialize(r)?,
            e1_beta: G1::guest_deserialize(r)?,
            e2_beta: G2::guest_deserialize(r)?,
        })
    }
}

impl<G1, G2, GT> crate::zkvm::guest_serde::GuestSerialize
    for dory::messages::SecondReduceMessage<G1, G2, GT>
where
    G1: crate::zkvm::guest_serde::GuestSerialize,
    G2: crate::zkvm::guest_serde::GuestSerialize,
    GT: crate::zkvm::guest_serde::GuestSerialize,
{
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.c_plus.guest_serialize(w)?;
        self.c_minus.guest_serialize(w)?;
        self.e1_plus.guest_serialize(w)?;
        self.e1_minus.guest_serialize(w)?;
        self.e2_plus.guest_serialize(w)?;
        self.e2_minus.guest_serialize(w)?;
        Ok(())
    }
}
impl<G1, G2, GT> crate::zkvm::guest_serde::GuestDeserialize
    for dory::messages::SecondReduceMessage<G1, G2, GT>
where
    G1: crate::zkvm::guest_serde::GuestDeserialize,
    G2: crate::zkvm::guest_serde::GuestDeserialize,
    GT: crate::zkvm::guest_serde::GuestDeserialize,
{
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(dory::messages::SecondReduceMessage {
            c_plus: GT::guest_deserialize(r)?,
            c_minus: GT::guest_deserialize(r)?,
            e1_plus: G1::guest_deserialize(r)?,
            e1_minus: G1::guest_deserialize(r)?,
            e2_plus: G2::guest_deserialize(r)?,
            e2_minus: G2::guest_deserialize(r)?,
        })
    }
}

impl<G1, G2> crate::zkvm::guest_serde::GuestSerialize
    for dory::messages::ScalarProductMessage<G1, G2>
where
    G1: crate::zkvm::guest_serde::GuestSerialize,
    G2: crate::zkvm::guest_serde::GuestSerialize,
{
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.e1.guest_serialize(w)?;
        self.e2.guest_serialize(w)?;
        Ok(())
    }
}
impl<G1, G2> crate::zkvm::guest_serde::GuestDeserialize
    for dory::messages::ScalarProductMessage<G1, G2>
where
    G1: crate::zkvm::guest_serde::GuestDeserialize,
    G2: crate::zkvm::guest_serde::GuestDeserialize,
{
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(dory::messages::ScalarProductMessage {
            e1: G1::guest_deserialize(r)?,
            e2: G2::guest_deserialize(r)?,
        })
    }
}
