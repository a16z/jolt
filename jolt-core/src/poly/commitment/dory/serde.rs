use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress as ArkCompress,
    SerializationError as ArkSerializationError, Valid as ArkValid, Validate as ArkValidate,
};
use dory::{
    primitives::{Compress, DoryDeserialize, DorySerialize, Validate},
    ProverSetup, VerifierSetup,
};
use std::io::{Read, Write};

use crate::poly::commitment::dory::{
    setup::{DoryProverSetup, DoryVerifierSetup},
    wrappers::JoltBn254,
};

impl CanonicalSerialize for DoryProverSetup {
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

impl ArkValid for DoryProverSetup {
    fn check(&self) -> Result<(), ArkSerializationError> {
        Ok(())
    }
}

impl ArkValid for DoryVerifierSetup {
    fn check(&self) -> Result<(), ArkSerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for DoryProverSetup {
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

        let setup = ProverSetup::<JoltBn254>::deserialize_with_mode(
            &mut reader,
            dory_compress,
            dory_validate,
        )
        .map_err(|_| ArkSerializationError::InvalidData)?;

        Ok(Self(setup))
    }
}

impl CanonicalSerialize for DoryVerifierSetup {
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

impl CanonicalDeserialize for DoryVerifierSetup {
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

        let setup = VerifierSetup::<JoltBn254>::deserialize_with_mode(
            &mut reader,
            dory_compress,
            dory_validate,
        )
        .map_err(|_| ArkSerializationError::InvalidData)?;

        Ok(Self(setup))
    }
}
