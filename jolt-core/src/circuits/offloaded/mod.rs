use crate::snark::OffloadedDataCircuit;
use ark_ec::pairing::Pairing;
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::{One, PrimeField};
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::fields::FieldVar;
use ark_r1cs_std::groups::CurveVar;
use ark_r1cs_std::{R1CSVar, ToConstraintFieldGadget};
use ark_relations::ns;
use ark_relations::r1cs::{ConstraintSystemRef, Namespace, SynthesisError};
use std::marker::PhantomData;

pub struct OffloadedMSMGadget<'a, FVar, G, GVar, Circuit>
where
    Circuit: OffloadedDataCircuit<G>,
    FVar: FieldVar<G::ScalarField, G::ScalarField> + ToConstraintFieldGadget<G::ScalarField>,
    G: CurveGroup,
    GVar: CurveVar<G, G::ScalarField> + ToConstraintFieldGadget<G::ScalarField>,
{
    _params: PhantomData<(FVar, G, GVar)>,
    circuit: &'a Circuit,
}

impl<'a, FVar, G, GVar, Circuit> OffloadedMSMGadget<'a, FVar, G, GVar, Circuit>
where
    Circuit: OffloadedDataCircuit<G>,
    FVar: FieldVar<G::ScalarField, G::ScalarField> + ToConstraintFieldGadget<G::ScalarField>,
    G: CurveGroup,
    GVar: CurveVar<G, G::ScalarField> + ToConstraintFieldGadget<G::ScalarField>,
{
    pub fn new(circuit: &'a Circuit) -> Self {
        Self {
            _params: PhantomData,
            circuit,
        }
    }
}

impl<'a, FVar, G, GVar, Circuit> MSMGadget<FVar, G, GVar>
    for OffloadedMSMGadget<'a, FVar, G, GVar, Circuit>
where
    Circuit: OffloadedDataCircuit<G>,
    FVar: FieldVar<G::ScalarField, G::ScalarField> + ToConstraintFieldGadget<G::ScalarField>,
    G: CurveGroup,
    GVar: CurveVar<G, G::ScalarField> + ToConstraintFieldGadget<G::ScalarField>,
{
    fn msm(
        &self,
        cs: impl Into<Namespace<G::ScalarField>>,
        g1s: &[GVar],
        scalars: &[FVar],
    ) -> Result<GVar, SynthesisError> {
        let ns = cs.into();
        let cs = ns.cs();

        let g1_values = g1s
            .iter()
            .map(|g1| g1.value().ok().map(|g1| g1.into_affine()))
            .collect::<Option<Vec<_>>>();

        let scalar_values = scalars
            .iter()
            .map(|s| s.value().ok())
            .collect::<Option<Vec<_>>>();

        let (full_msm_value, msm_g1_value) = g1_values
            .zip(scalar_values)
            .map(|(g1s, scalars)| {
                let r_g1 = G::msm_unchecked(&g1s, &scalars);
                let minus_one = -G::ScalarField::one();
                (
                    (
                        [g1s, vec![r_g1.into()]].concat(),
                        [scalars, vec![minus_one]].concat(),
                    ),
                    r_g1,
                )
            })
            .unzip();

        let msm_g1_var = GVar::new_witness(ns!(cs, "msm_g1"), || {
            msm_g1_value.ok_or(SynthesisError::AssignmentMissing)
        })?;

        {
            let g1s = g1s.to_vec();
            let scalars = scalars.to_vec();
            let msm_g1_var = msm_g1_var.clone();
            let ns = ns!(cs, "deferred_msm");
            let cs = ns.cs();

            self.circuit.defer_msm(move || {
                // write scalars to public_input
                for x in scalars {
                    let scalar_input = FVar::new_input(ns!(cs, "scalar"), || x.value())?;
                    scalar_input.enforce_equal(&x)?;
                }
                dbg!(cs.num_constraints());

                // write g1s to public_input
                for g1 in g1s {
                    let f_vec = g1.to_constraint_field()?;

                    for f in f_vec.iter() {
                        let f_input = FpVar::new_input(ns!(cs, "g1s"), || f.value())?;
                        f_input.enforce_equal(f)?;
                    }
                }
                dbg!(cs.num_constraints());

                // write msm_g1 to public_input
                {
                    let f_vec = msm_g1_var.to_constraint_field()?;

                    for f in f_vec.iter() {
                        let f_input = FpVar::new_input(ns!(cs, "msm_g1"), || f.value())?;
                        f_input.enforce_equal(f)?;
                    }
                }
                dbg!(cs.num_constraints());
                dbg!(cs.num_instance_variables());

                Ok(full_msm_value)
            })
        };
        dbg!(cs.num_constraints());

        Ok(msm_g1_var)
    }
}

pub trait MSMGadget<FVar, G, GVar>
where
    FVar: FieldVar<G::ScalarField, G::ScalarField> + ToConstraintFieldGadget<G::ScalarField>,
    G: CurveGroup,
    GVar: CurveVar<G, G::ScalarField> + ToConstraintFieldGadget<G::ScalarField>,
{
    fn msm(
        &self,
        cs: impl Into<Namespace<G::ScalarField>>,
        g1s: &[GVar],
        scalars: &[FVar],
    ) -> Result<GVar, SynthesisError>;
}
