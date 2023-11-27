use ark_ec::{pairing::Pairing, CurveGroup};

use crate::msm::VariableBaseMSM;

/// MINIMAL KZG IMPLEMENTATION BASED ON ARKWORKS. PROVIDES METHOD OF CONVERTING FROM LAGRANGE TO MONOMIAL BASIS AND CREATES A lazy_static reference used directly in the setup ceremony.
// Questions:
// Should we encapsulate this the proof and commitment into there own structs?
//
//
pub fn commit<P: Pairing>(powers: &[P::G1Affine], scalars: &[P::ScalarField]) -> P::G1Affine {
  <P::G1 as VariableBaseMSM>::msm(powers, &scalars)
    .unwrap()
    .into_affine()
}
