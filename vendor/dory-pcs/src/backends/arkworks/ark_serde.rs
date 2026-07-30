//! Manual implementations of Dory serialization traits for arkworks wrapper types
use crate::backends::arkworks::{ArkFr, ArkG1, ArkG2, ArkGT};
use crate::primitives::serialization::{Compress, SerializationError, Valid, Validate};
use crate::primitives::{DoryDeserialize, DorySerialize};
use ark_bn254::Bn254;
use ark_ec::pairing::PairingOutput;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress as ArkCompress,
    SerializationError as ArkSerializationError, Valid as ArkValid, Validate as ArkValidate,
};
use std::io::{Read, Write};

use super::BN254;
use crate::messages::{FirstReduceMessage, SecondReduceMessage, VMVMessage};
use crate::setup::{ProverSetup, VerifierSetup};

impl Valid for ArkFr {
    fn check(&self) -> Result<(), SerializationError> {
        self.0
            .check()
            .map_err(|e| SerializationError::InvalidData(format!("{e:?}")))
    }
}

impl DorySerialize for ArkFr {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match compress {
            Compress::Yes => self
                .0
                .serialize_compressed(writer)
                .map_err(|e| SerializationError::InvalidData(format!("{e}"))),
            Compress::No => self
                .0
                .serialize_uncompressed(writer)
                .map_err(|e| SerializationError::InvalidData(format!("{e}"))),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match compress {
            Compress::Yes => self.0.compressed_size(),
            Compress::No => self.0.uncompressed_size(),
        }
    }
}

impl DoryDeserialize for ArkFr {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let inner = match compress {
            Compress::Yes => ark_bn254::Fr::deserialize_compressed(reader)
                .map_err(|e| SerializationError::InvalidData(format!("{e}")))?,
            Compress::No => ark_bn254::Fr::deserialize_uncompressed(reader)
                .map_err(|e| SerializationError::InvalidData(format!("{e}")))?,
        };

        if matches!(validate, Validate::Yes) {
            inner
                .check()
                .map_err(|e| SerializationError::InvalidData(format!("{e:?}")))?;
        }

        Ok(ArkFr(inner))
    }
}

impl Valid for ArkG1 {
    fn check(&self) -> Result<(), SerializationError> {
        self.0
            .check()
            .map_err(|e| SerializationError::InvalidData(format!("{e:?}")))
    }
}

impl DorySerialize for ArkG1 {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match compress {
            Compress::Yes => self
                .0
                .serialize_compressed(writer)
                .map_err(|e| SerializationError::InvalidData(format!("{e}"))),
            Compress::No => self
                .0
                .serialize_uncompressed(writer)
                .map_err(|e| SerializationError::InvalidData(format!("{e}"))),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match compress {
            Compress::Yes => self.0.compressed_size(),
            Compress::No => self.0.uncompressed_size(),
        }
    }
}

impl DoryDeserialize for ArkG1 {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let inner = match compress {
            Compress::Yes => ark_bn254::G1Projective::deserialize_compressed(reader)
                .map_err(|e| SerializationError::InvalidData(format!("{e}")))?,
            Compress::No => ark_bn254::G1Projective::deserialize_uncompressed(reader)
                .map_err(|e| SerializationError::InvalidData(format!("{e}")))?,
        };

        if matches!(validate, Validate::Yes) {
            inner
                .check()
                .map_err(|e| SerializationError::InvalidData(format!("{e:?}")))?;
        }

        Ok(ArkG1(inner))
    }
}

impl Valid for ArkG2 {
    fn check(&self) -> Result<(), SerializationError> {
        self.0
            .check()
            .map_err(|e| SerializationError::InvalidData(format!("{e:?}")))
    }
}

impl DorySerialize for ArkG2 {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match compress {
            Compress::Yes => self
                .0
                .serialize_compressed(writer)
                .map_err(|e| SerializationError::InvalidData(format!("{e}"))),
            Compress::No => self
                .0
                .serialize_uncompressed(writer)
                .map_err(|e| SerializationError::InvalidData(format!("{e}"))),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match compress {
            Compress::Yes => self.0.compressed_size(),
            Compress::No => self.0.uncompressed_size(),
        }
    }
}

impl DoryDeserialize for ArkG2 {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let inner = match compress {
            Compress::Yes => ark_bn254::G2Projective::deserialize_compressed(reader)
                .map_err(|e| SerializationError::InvalidData(format!("{e}")))?,
            Compress::No => ark_bn254::G2Projective::deserialize_uncompressed(reader)
                .map_err(|e| SerializationError::InvalidData(format!("{e}")))?,
        };

        if matches!(validate, Validate::Yes) {
            inner
                .check()
                .map_err(|e| SerializationError::InvalidData(format!("{e:?}")))?;
        }

        Ok(ArkG2(inner))
    }
}

impl Valid for ArkGT {
    fn check(&self) -> Result<(), SerializationError> {
        self.0
            .check()
            .map_err(|e| SerializationError::InvalidData(format!("{e:?}")))
    }
}

impl DorySerialize for ArkGT {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match compress {
            Compress::Yes => self
                .0
                .serialize_compressed(writer)
                .map_err(|e| SerializationError::InvalidData(format!("{e}"))),
            Compress::No => self
                .0
                .serialize_uncompressed(writer)
                .map_err(|e| SerializationError::InvalidData(format!("{e}"))),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match compress {
            Compress::Yes => self.0.compressed_size(),
            Compress::No => self.0.uncompressed_size(),
        }
    }
}

impl DoryDeserialize for ArkGT {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let inner = match compress {
            Compress::Yes => PairingOutput::<Bn254>::deserialize_compressed(reader)
                .map_err(|e| SerializationError::InvalidData(format!("{e}")))?,
            Compress::No => PairingOutput::<Bn254>::deserialize_uncompressed(reader)
                .map_err(|e| SerializationError::InvalidData(format!("{e}")))?,
        };

        if matches!(validate, Validate::Yes) {
            inner
                .check()
                .map_err(|e| SerializationError::InvalidData(format!("{e:?}")))?;
        }

        Ok(ArkGT(inner))
    }
}

// Arkworks-specific Dory proof type
use super::ArkDoryProof;

mod opt_serde {
    use ark_serialize::{
        CanonicalDeserialize as De, CanonicalSerialize as Ser, Compress, SerializationError, Valid,
        Validate,
    };
    use std::io::{Read, Write};

    pub(super) fn ser_opt<W: Write, T: Ser>(
        v: &Option<T>,
        w: &mut W,
        c: Compress,
    ) -> Result<(), SerializationError> {
        match v {
            Some(val) => {
                Ser::serialize_with_mode(&1u8, &mut *w, c)?;
                Ser::serialize_with_mode(val, w, c)
            }
            None => Ser::serialize_with_mode(&0u8, w, c),
        }
    }

    pub(super) fn de_opt<R: Read, T: De>(
        r: &mut R,
        c: Compress,
        v: Validate,
    ) -> Result<Option<T>, SerializationError> {
        match <u8 as De>::deserialize_with_mode(&mut *r, c, v)? {
            0 => Ok(None),
            1 => Ok(Some(T::deserialize_with_mode(r, c, v)?)),
            _ => Err(SerializationError::InvalidData),
        }
    }

    pub(super) fn size_opt<T: Ser>(v: &Option<T>, c: Compress) -> usize {
        1 + v.as_ref().map_or(0, |val| Ser::serialized_size(val, c))
    }

    macro_rules! impl_serde {
        ($ty:ty, [$($field:ident),+]) => {
            impl Valid for $ty { fn check(&self) -> Result<(), SerializationError> { Ok(()) } }
            impl Ser for $ty {
                fn serialize_with_mode<W: Write>(&self, mut w: W, c: Compress) -> Result<(), SerializationError> {
                    $(Ser::serialize_with_mode(&self.$field, &mut w, c)?;)+
                    Ok(())
                }
                fn serialized_size(&self, c: Compress) -> usize {
                    0 $(+ Ser::serialized_size(&self.$field, c))+
                }
            }
            impl De for $ty {
                fn deserialize_with_mode<R: Read>(mut r: R, c: Compress, v: Validate) -> Result<Self, SerializationError> {
                    Ok(Self { $($field: De::deserialize_with_mode(&mut r, c, v)?),+ })
                }
            }
        };
    }

    use super::{ArkFr, ArkG1, ArkG2, ArkGT};
    use crate::messages::{ScalarProductMessage, ScalarProductProof};
    #[cfg(feature = "zk")]
    use crate::messages::{Sigma1Proof, Sigma2Proof};

    #[cfg(feature = "zk")]
    impl_serde!(Sigma1Proof<ArkG1, ArkG2, ArkFr>, [a1, a2, z1, z2, z3]);
    #[cfg(feature = "zk")]
    impl_serde!(Sigma2Proof<ArkFr, ArkGT>, [a, z1, z2]);
    impl_serde!(ScalarProductMessage<ArkG1, ArkG2>, [e1, e2]);
    impl_serde!(ScalarProductProof<ArkG1, ArkG2, ArkFr, ArkGT>, [p1, p2, q, r, e1, e2, r1, r2, r3]);
}

impl ArkValid for ArkDoryProof {
    fn check(&self) -> Result<(), ArkSerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for ArkDoryProof {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: ArkCompress,
    ) -> Result<(), ArkSerializationError> {
        // Serialize VMV message
        CanonicalSerialize::serialize_with_mode(&self.vmv_message.c, &mut writer, compress)?;
        CanonicalSerialize::serialize_with_mode(&self.vmv_message.d2, &mut writer, compress)?;
        CanonicalSerialize::serialize_with_mode(&self.vmv_message.e1, &mut writer, compress)?;

        // Serialize number of rounds
        let num_rounds = self.first_messages.len() as u32;
        CanonicalSerialize::serialize_with_mode(&num_rounds, &mut writer, compress)?;

        // Serialize first messages
        for msg in &self.first_messages {
            CanonicalSerialize::serialize_with_mode(&msg.d1_left, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.d1_right, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.d2_left, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.d2_right, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.e1_beta, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.e2_beta, &mut writer, compress)?;
        }

        // Serialize second messages
        for msg in &self.second_messages {
            CanonicalSerialize::serialize_with_mode(&msg.c_plus, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.c_minus, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.e1_plus, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.e1_minus, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.e2_plus, &mut writer, compress)?;
            CanonicalSerialize::serialize_with_mode(&msg.e2_minus, &mut writer, compress)?;
        }

        // Serialize final message (present in transparent proofs only)
        opt_serde::ser_opt(&self.final_message, &mut writer, compress)?;

        // Serialize nu and sigma
        CanonicalSerialize::serialize_with_mode(&(self.nu as u32), &mut writer, compress)?;
        CanonicalSerialize::serialize_with_mode(&(self.sigma as u32), &mut writer, compress)?;

        #[cfg(feature = "zk")]
        {
            opt_serde::ser_opt(&self.e2, &mut writer, compress)?;
            opt_serde::ser_opt(&self.y_com, &mut writer, compress)?;
            opt_serde::ser_opt(&self.sigma1_proof, &mut writer, compress)?;
            opt_serde::ser_opt(&self.sigma2_proof, &mut writer, compress)?;
            opt_serde::ser_opt(&self.scalar_product_proof, &mut writer, compress)?;
        }

        Ok(())
    }

    fn serialized_size(&self, compress: ArkCompress) -> usize {
        let mut size = 0;

        // VMV message
        size += CanonicalSerialize::serialized_size(&self.vmv_message.c, compress);
        size += CanonicalSerialize::serialized_size(&self.vmv_message.d2, compress);
        size += CanonicalSerialize::serialized_size(&self.vmv_message.e1, compress);

        // Number of rounds
        size += 4; // u32

        // First messages
        for msg in &self.first_messages {
            size += CanonicalSerialize::serialized_size(&msg.d1_left, compress);
            size += CanonicalSerialize::serialized_size(&msg.d1_right, compress);
            size += CanonicalSerialize::serialized_size(&msg.d2_left, compress);
            size += CanonicalSerialize::serialized_size(&msg.d2_right, compress);
            size += CanonicalSerialize::serialized_size(&msg.e1_beta, compress);
            size += CanonicalSerialize::serialized_size(&msg.e2_beta, compress);
        }

        // Second messages
        for msg in &self.second_messages {
            size += CanonicalSerialize::serialized_size(&msg.c_plus, compress);
            size += CanonicalSerialize::serialized_size(&msg.c_minus, compress);
            size += CanonicalSerialize::serialized_size(&msg.e1_plus, compress);
            size += CanonicalSerialize::serialized_size(&msg.e1_minus, compress);
            size += CanonicalSerialize::serialized_size(&msg.e2_plus, compress);
            size += CanonicalSerialize::serialized_size(&msg.e2_minus, compress);
        }

        // Final message (present in transparent proofs only)
        size += opt_serde::size_opt(&self.final_message, compress);

        // nu and sigma
        size += 8; // 2 * u32

        #[cfg(feature = "zk")]
        {
            size += opt_serde::size_opt(&self.e2, compress);
            size += opt_serde::size_opt(&self.y_com, compress);
            size += opt_serde::size_opt(&self.sigma1_proof, compress);
            size += opt_serde::size_opt(&self.sigma2_proof, compress);
            size += opt_serde::size_opt(&self.scalar_product_proof, compress);
        }

        size
    }
}

impl CanonicalDeserialize for ArkDoryProof {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: ArkCompress,
        validate: ArkValidate,
    ) -> Result<Self, ArkSerializationError> {
        // Deserialize VMV message
        let c = CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let d2 = CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let e1 = CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let vmv_message = VMVMessage { c, d2, e1 };

        // Deserialize number of rounds
        let num_rounds =
            <u32 as CanonicalDeserialize>::deserialize_with_mode(&mut reader, compress, validate)?
                as usize;

        // Deserialize first messages
        let mut first_messages = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            let d1_left =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let d1_right =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let d2_left =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let d2_right =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let e1_beta =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let e2_beta =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            first_messages.push(FirstReduceMessage {
                d1_left,
                d1_right,
                d2_left,
                d2_right,
                e1_beta,
                e2_beta,
            });
        }

        // Deserialize second messages
        let mut second_messages = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            let c_plus =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let c_minus =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let e1_plus =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let e1_minus =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let e2_plus =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            let e2_minus =
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
            second_messages.push(SecondReduceMessage {
                c_plus,
                c_minus,
                e1_plus,
                e1_minus,
                e2_plus,
                e2_minus,
            });
        }

        // Deserialize final message (present in transparent proofs only)
        let final_message = opt_serde::de_opt(&mut reader, compress, validate)?;

        // Deserialize nu and sigma
        let nu =
            <u32 as CanonicalDeserialize>::deserialize_with_mode(&mut reader, compress, validate)?
                as usize;
        let sigma =
            <u32 as CanonicalDeserialize>::deserialize_with_mode(&mut reader, compress, validate)?
                as usize;

        Ok(ArkDoryProof {
            vmv_message,
            first_messages,
            second_messages,
            final_message,
            nu,
            sigma,
            #[cfg(feature = "zk")]
            e2: opt_serde::de_opt(&mut reader, compress, validate)?,
            #[cfg(feature = "zk")]
            y_com: opt_serde::de_opt(&mut reader, compress, validate)?,
            #[cfg(feature = "zk")]
            sigma1_proof: opt_serde::de_opt(&mut reader, compress, validate)?,
            #[cfg(feature = "zk")]
            sigma2_proof: opt_serde::de_opt(&mut reader, compress, validate)?,
            #[cfg(feature = "zk")]
            scalar_product_proof: opt_serde::de_opt(&mut reader, compress, validate)?,
        })
    }
}

// Setup wrapper types (declared in ark_setup.rs, serialization impls here)
use super::{ArkworksProverSetup, ArkworksVerifierSetup};

impl ArkValid for ArkworksProverSetup {
    fn check(&self) -> Result<(), ArkSerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for ArkworksProverSetup {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: ArkCompress,
    ) -> Result<(), ArkSerializationError> {
        let dory_compress = match compress {
            ArkCompress::Yes => Compress::Yes,
            ArkCompress::No => Compress::No,
        };

        DorySerialize::serialize_with_mode(&self.0, &mut writer, dory_compress)
            .map_err(|_| ArkSerializationError::InvalidData)
    }

    fn serialized_size(&self, compress: ArkCompress) -> usize {
        let dory_compress = match compress {
            ArkCompress::Yes => Compress::Yes,
            ArkCompress::No => Compress::No,
        };
        DorySerialize::serialized_size(&self.0, dory_compress)
    }
}

impl CanonicalDeserialize for ArkworksProverSetup {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: ArkCompress,
        validate: ArkValidate,
    ) -> Result<Self, ArkSerializationError> {
        let dory_compress = match compress {
            ArkCompress::Yes => Compress::Yes,
            ArkCompress::No => Compress::No,
        };

        let dory_validate = match validate {
            ArkValidate::Yes => Validate::Yes,
            ArkValidate::No => Validate::No,
        };

        let setup =
            ProverSetup::<BN254>::deserialize_with_mode(&mut reader, dory_compress, dory_validate)
                .map_err(|_| ArkSerializationError::InvalidData)?;

        Ok(Self(setup))
    }
}

impl ArkValid for ArkworksVerifierSetup {
    fn check(&self) -> Result<(), ArkSerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for ArkworksVerifierSetup {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: ArkCompress,
    ) -> Result<(), ArkSerializationError> {
        let dory_compress = match compress {
            ArkCompress::Yes => Compress::Yes,
            ArkCompress::No => Compress::No,
        };

        DorySerialize::serialize_with_mode(&self.0, &mut writer, dory_compress)
            .map_err(|_| ArkSerializationError::InvalidData)
    }

    fn serialized_size(&self, compress: ArkCompress) -> usize {
        let dory_compress = match compress {
            ArkCompress::Yes => Compress::Yes,
            ArkCompress::No => Compress::No,
        };
        DorySerialize::serialized_size(&self.0, dory_compress)
    }
}

impl CanonicalDeserialize for ArkworksVerifierSetup {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: ArkCompress,
        validate: ArkValidate,
    ) -> Result<Self, ArkSerializationError> {
        let dory_compress = match compress {
            ArkCompress::Yes => Compress::Yes,
            ArkCompress::No => Compress::No,
        };

        let dory_validate = match validate {
            ArkValidate::Yes => Validate::Yes,
            ArkValidate::No => Validate::No,
        };

        let setup = VerifierSetup::<BN254>::deserialize_with_mode(
            &mut reader,
            dory_compress,
            dory_validate,
        )
        .map_err(|_| ArkSerializationError::InvalidData)?;

        Ok(Self(setup))
    }
}
