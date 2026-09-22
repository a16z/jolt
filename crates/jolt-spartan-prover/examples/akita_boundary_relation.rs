//! Build the fixed partial Akita boundary relation from trusted typed artifacts.
//! This checks the assignment and public partition; it does not generate a SNARK.
use blake2::{digest::consts::U32, Blake2b, Digest};
use common::jolt_device::JoltDevice;
use jolt_field::CanonicalBytes;
use jolt_prover_legacy::zkvm::packed::{AkitaJoltProof, AkitaScheme, AkitaVc};
use jolt_r1cs::R1csBuilder;
use jolt_spartan_verifier::{preprocessed::MatrixApplicationIds, SpartanKey};
use jolt_verifier::{
    r1cs::{AkitaStage1BoundaryShape, Stage1RemainderShape},
    JoltVerifierPreprocessing,
};
use serde::de::DeserializeOwned;
use std::{error::Error, path::PathBuf};

struct Application;
impl Application {
    fn digest(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
        Blake2b::<U32>::new()
            .chain_update(domain)
            .chain_update([0])
            .chain_update((bytes.len() as u64).to_le_bytes())
            .chain_update(bytes)
            .finalize()
            .into()
    }
    fn identities(
        shape: &AkitaStage1BoundaryShape,
        remainder: &Stage1RemainderShape,
    ) -> MatrixApplicationIds {
        let setup = Self::digest(
            b"jolt-boundary-setup-v1",
            shape.commitment_shape().setup_encoding(),
        );
        let mut profile = shape.profile_descriptor(setup);
        profile.extend(remainder.profile_descriptor());
        MatrixApplicationIds {
            circuit: Self::digest(
                b"jolt-boundary-circuit-v1",
                b"preamble+packed-commitment+complete-stage1",
            ),
            profile: Self::digest(b"jolt-boundary-profile-v1", &profile),
            public_schema: Self::digest(
                b"jolt-boundary-public-v1",
                b"input3-output1-nonzero-panic1-le33",
            ),
            table: Self::digest(
                b"jolt-boundary-table-v1",
                shape.commitment_shape().catalog(),
            ),
        }
    }
    fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, Box<dyn Error>> {
        let (value, used) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
        if used != bytes.len() {
            return Err("trailing artifact bytes".into());
        }
        Ok(value)
    }
}

#[expect(
    clippy::print_stdout,
    reason = "intentional partial-relation compiler reports the public key descriptors and matrix geometry"
)]
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let directory = PathBuf::from(
        args.next()
            .ok_or("expected trusted proof/preprocessing/public-io directory")?,
    );
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let preprocessing: JoltVerifierPreprocessing<AkitaScheme, AkitaVc> =
        Application::decode(&std::fs::read(directory.join("preprocessing.bin"))?)?;
    let io: JoltDevice = Application::decode(&std::fs::read(directory.join("public-io.bin"))?)?;
    let proof: AkitaJoltProof = Application::decode(&std::fs::read(directory.join("proof.bin"))?)?;
    let shape = AkitaStage1BoundaryShape::new(&preprocessing, &io, &proof)?;
    let witness = shape.witness(&io, &proof)?;
    let remainder = Stage1RemainderShape::new()?;
    let remainder_witness = remainder.witness(&proof)?;
    let ids = Application::identities(&shape, &remainder);
    let mut builder = R1csBuilder::new();
    let mut handles = shape.constrain(&mut builder, Some(&witness))?;
    let _stage1 = remainder.constrain(&mut builder, &mut handles, Some(&remainder_witness))?;
    if handles.public.index() != 1 {
        return Err("unexpected public column".into());
    }
    let assignment = builder.witness()?;
    let public = witness.public.packed()?;
    if assignment.get(handles.public.index()) != Some(&public) {
        return Err("assignment does not bind the external public value".into());
    }
    let key = SpartanKey::new(builder.into_matrices(), 1, ids.profile)?;
    key.validate_public_inputs(&[public])?;
    key.matrices()
        .check_witness(&assignment)
        .map_err(|row| format!("unsatisfied boundary row {row}"))?;
    // Spartan reconstructs z=[ONE,p,private]; the witness never supplies ONE.
    if key.public_columns() != 2 {
        return Err("unexpected public partition".into());
    }
    let nonzeros = [&key.matrices().a, &key.matrices().b, &key.matrices().c]
        .into_iter()
        .flat_map(|matrix| matrix.iter())
        .map(Vec::len)
        .sum::<usize>();
    println!("matrix nonzeros={nonzeros}");
    println!("complete stage1, later stages absent: rows={} variables={} public_columns={} public={:?} identities={ids:?}", key.matrices().num_constraints, key.matrices().num_vars, key.public_columns(), witness.public.packed()?.to_bytes_le_vec());
    Ok(())
}
