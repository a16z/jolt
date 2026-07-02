use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_openings::{EvaluationClaim, OpeningsError, PrefixPacking};
use jolt_poly::Polynomial;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

/// Logical ids for the shared mixed-arity packed fixture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PackedId {
    Constant,
    NarrowA,
    NarrowB,
    Medium,
    Wide,
    Unused,
}

/// Mixed-arity fixture: arities 3, 2, 1, 1, 0 packing into 5 variables.
pub fn packed_polynomials() -> Vec<(PackedId, Polynomial<Fr>)> {
    let mut rng = ChaCha20Rng::seed_from_u64(0x0a_11_ce);
    vec![
        (PackedId::Wide, Polynomial::<Fr>::random(3, &mut rng)),
        (PackedId::Medium, Polynomial::<Fr>::random(2, &mut rng)),
        (PackedId::NarrowB, Polynomial::<Fr>::random(1, &mut rng)),
        (PackedId::NarrowA, Polynomial::<Fr>::random(1, &mut rng)),
        (PackedId::Constant, Polynomial::new(vec![Fr::from_u64(41)])),
    ]
}

pub fn build_packed(
    polynomials: &[(PackedId, Polynomial<Fr>)],
) -> MaterializedPackedWitness<PackedId, Fr> {
    materialize_packed(polynomials).expect("packed witness should build")
}

/// Honest claims for every slot at the suffixes of one shared packed point.
pub fn packed_claims(
    polynomials: &[(PackedId, Polynomial<Fr>)],
    packing: &PrefixPacking<PackedId>,
    packed_point: &[Fr],
) -> Vec<(PackedId, EvaluationClaim<Fr>)> {
    polynomials
        .iter()
        .map(|(id, polynomial)| {
            let logical_point = packing
                .logical_point(id, packed_point)
                .expect("packed point should produce logical suffix");
            (
                *id,
                EvaluationClaim::new(logical_point.clone(), polynomial.evaluate(&logical_point)),
            )
        })
        .collect()
}

/// Honest claims for every slot at mutually independent random points.
pub fn independent_claims(
    polynomials: &[(PackedId, Polynomial<Fr>)],
    rng: &mut ChaCha20Rng,
) -> Vec<(PackedId, EvaluationClaim<Fr>)> {
    polynomials
        .iter()
        .map(|(id, polynomial)| {
            let logical_point = (0..polynomial.num_vars())
                .map(|_| Fr::random(rng))
                .collect::<Vec<_>>();
            (
                *id,
                EvaluationClaim::new(logical_point.clone(), polynomial.evaluate(&logical_point)),
            )
        })
        .collect()
}

use jolt_field::Field;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterializedPackedWitness<Id, F> {
    pub packing: PrefixPacking<Id>,
    pub polynomial: Polynomial<F>,
}

pub fn materialize_packed<Id, F>(
    polynomials: &[(Id, Polynomial<F>)],
) -> Result<MaterializedPackedWitness<Id, F>, OpeningsError>
where
    Id: Clone + Ord,
    F: Field,
{
    let packing = PrefixPacking::new(
        polynomials
            .iter()
            .map(|(id, polynomial)| (id.clone(), polynomial.num_vars())),
    )?;
    let packed_len = domain_size(packing.packed_num_vars)?;
    let mut packed_evaluations = vec![F::zero(); packed_len];

    for (id, polynomial) in polynomials {
        let offset = packing[id].packed_offset();
        for (local_index, evaluation) in polynomial.evaluations().iter().copied().enumerate() {
            packed_evaluations[offset + local_index] = evaluation;
        }
    }

    Ok(MaterializedPackedWitness {
        packing,
        polynomial: Polynomial::new(packed_evaluations),
    })
}

fn domain_size(num_vars: usize) -> Result<usize, OpeningsError> {
    1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| OpeningsError::InvalidSetup("polynomial domain size overflow".to_owned()))
}
