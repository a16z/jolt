use jolt_field::Field;
use jolt_openings::{OpeningsError, PrefixPacking};
use jolt_poly::Polynomial;

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
        let slot = &packing[id];
        let offset = prefix_index(&slot.prefix) << slot.num_vars;
        for (local_index, evaluation) in polynomial.evaluations().iter().copied().enumerate() {
            packed_evaluations[offset + local_index] = evaluation;
        }
    }

    Ok(MaterializedPackedWitness {
        packing,
        polynomial: Polynomial::new(packed_evaluations),
    })
}

fn prefix_index(prefix: &[bool]) -> usize {
    prefix
        .iter()
        .fold(0usize, |acc, bit| (acc << 1) | usize::from(*bit))
}

fn domain_size(num_vars: usize) -> Result<usize, OpeningsError> {
    1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| OpeningsError::InvalidSetup("polynomial domain size overflow".to_owned()))
}
