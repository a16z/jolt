use crate::{field::JoltField, poly::multilinear_polynomial::MultilinearPolynomial};

#[allow(dead_code)]
pub struct CommittedPolynomials<F: JoltField> {
    bytecode_ra: MultilinearPolynomial<F>,
    rd_inc: MultilinearPolynomial<F>,
    rd_wv: MultilinearPolynomial<F>,
    ram_inc: MultilinearPolynomial<F>,
    ram_wv: MultilinearPolynomial<F>,
    ram_ra: MultilinearPolynomial<F>,
    instruction_ra: [MultilinearPolynomial<F>; 4],
}

pub trait JoltWitness {
    fn is_virtual() -> bool;
    fn as_polynomial<F: JoltField>(&self) -> MultilinearPolynomial<F>;
}
