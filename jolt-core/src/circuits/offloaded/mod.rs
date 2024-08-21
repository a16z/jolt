use crate::snark::{DeferredOpData, OffloadedDataCircuit};
use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_r1cs_std::{
    alloc::AllocVar, eq::EqGadget, fields::fp::FpVar, fields::FieldVar, groups::CurveVar, R1CSVar,
    ToConstraintFieldGadget,
};
use ark_relations::{
    ns,
    r1cs::{Namespace, SynthesisError},
};
use ark_std::{One, Zero};
use std::marker::PhantomData;

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

pub struct OffloadedMSMGadget<'a, FVar, E, GVar, Circuit>
where
    Circuit: OffloadedDataCircuit<E>,
    FVar: FieldVar<E::ScalarField, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    E: Pairing,
    GVar: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    _params: PhantomData<(FVar, E, GVar)>,
    circuit: &'a Circuit,
}

impl<'a, FVar, E, GVar, Circuit> OffloadedMSMGadget<'a, FVar, E, GVar, Circuit>
where
    Circuit: OffloadedDataCircuit<E>,
    FVar: FieldVar<E::ScalarField, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    E: Pairing,
    GVar: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    pub fn new(circuit: &'a Circuit) -> Self {
        Self {
            _params: PhantomData,
            circuit,
        }
    }
}

impl<'a, FVar, E, GVar, Circuit> MSMGadget<FVar, E::G1, GVar>
    for OffloadedMSMGadget<'a, FVar, E, GVar, Circuit>
where
    Circuit: OffloadedDataCircuit<E>,
    FVar: FieldVar<E::ScalarField, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    E: Pairing,
    GVar: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    fn msm(
        &self,
        cs: impl Into<Namespace<E::ScalarField>>,
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
                let r_g1 = E::G1::msm_unchecked(&g1s, &scalars);
                let minus_one = -E::ScalarField::one();
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

            self.circuit.defer_op(move || {
                // write scalars to public_input
                for x in scalars {
                    let scalar_input = FVar::new_input(ns!(cs, "scalar"), || x.value())?;
                    scalar_input.enforce_equal(&x)?;
                }

                // write g1s to public_input
                for g1 in g1s {
                    let f_vec = g1.to_constraint_field()?;

                    for f in f_vec.iter() {
                        let f_input = FpVar::new_input(ns!(cs, "g1s"), || f.value())?;
                        f_input.enforce_equal(f)?;
                    }
                }

                // write msm_g1 to public_input
                {
                    dbg!(cs.num_instance_variables() - 1);
                    let f_vec = msm_g1_var.to_constraint_field()?;

                    for f in f_vec.iter() {
                        let f_input = FpVar::new_input(ns!(cs, "msm_g1"), || f.value())?;
                        f_input.enforce_equal(f)?;
                    }
                }
                dbg!(cs.num_constraints());
                dbg!(cs.num_instance_variables());

                Ok(DeferredOpData::MSM(full_msm_value))
            })
        };
        dbg!(cs.num_constraints());

        Ok(msm_g1_var)
    }
}

pub trait PairingGadget<E, G1Var>
where
    E: Pairing,
    G1Var: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    fn multi_pairing_is_zero(
        &self,
        cs: impl Into<Namespace<E::ScalarField>>,
        g1s: &[G1Var],
        g2s: &[E::G2Affine],
    ) -> Result<(), SynthesisError>;
}

pub struct OffloadedPairingGadget<'a, E, FVar, GVar, Circuit>
where
    E: Pairing,
    Circuit: OffloadedDataCircuit<E>,
    FVar: FieldVar<E::ScalarField, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    GVar: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    _params: PhantomData<(E, FVar, GVar)>,
    circuit: &'a Circuit,
}

impl<'a, E, FVar, GVar, Circuit> OffloadedPairingGadget<'a, E, FVar, GVar, Circuit>
where
    E: Pairing,
    Circuit: OffloadedDataCircuit<E>,
    FVar: FieldVar<E::ScalarField, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    GVar: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    pub(crate) fn new(circuit: &'a Circuit) -> Self {
        Self {
            _params: PhantomData,
            circuit,
        }
    }
}

impl<'a, E, FVar, GVar, Circuit> PairingGadget<E, GVar>
    for OffloadedPairingGadget<'a, E, FVar, GVar, Circuit>
where
    E: Pairing,
    Circuit: OffloadedDataCircuit<E>,
    FVar: FieldVar<E::ScalarField, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
    GVar: CurveVar<E::G1, E::ScalarField> + ToConstraintFieldGadget<E::ScalarField>,
{
    fn multi_pairing_is_zero(
        &self,
        cs: impl Into<Namespace<E::ScalarField>>,
        g1s: &[GVar],
        g2s: &[E::G2Affine],
    ) -> Result<(), SynthesisError> {
        let ns = cs.into();
        let cs = ns.cs();

        let g1_values_opt = g1s
            .iter()
            .map(|g1| g1.value().ok().map(|g1| g1.into_affine()))
            .collect::<Option<Vec<_>>>();

        let g2_values = g2s;

        for g1_values in g1_values_opt.iter() {
            if !E::multi_pairing(g1_values, g2_values).is_zero() {
                return Err(SynthesisError::Unsatisfiable);
            }
        }

        {
            let g1_values_opt = g1_values_opt;
            let g2_values = g2_values.to_vec();
            let g1s = g1s.to_vec();
            let ns = ns!(cs, "deferred_pairing");
            let cs = ns.cs();

            self.circuit.defer_op(move || {
                // write g1s to public_input
                for g1 in g1s {
                    let f_vec = g1.to_constraint_field()?;

                    for f in f_vec.iter() {
                        let f_input = FpVar::new_input(ns!(cs, "g1s"), || f.value())?;
                        f_input.enforce_equal(f)?;
                    }
                }

                dbg!(cs.num_constraints());
                dbg!(cs.num_instance_variables());

                Ok(DeferredOpData::Pairing(g1_values_opt, g2_values))
            })
        }

        Ok(())
    }
}
