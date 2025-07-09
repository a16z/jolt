///
pub mod poseidon;

///
pub mod kzg;

///
pub mod planner;
use halo2_proofs::{
    circuit::Layouter,
    plonk::{ConstraintSystem, Error},
};
use halo2curves::ff::PrimeField;
pub use planner::*;

use crate::tensor::{TensorType, ValTensor};

/// Module trait used to extend ezkl functionality
pub trait Module<F: PrimeField + TensorType + PartialOrd> {
    /// Config
    type Config;
    /// The return type after an input assignment
    type InputAssignments;
    /// The inputs used in the run function
    type RunInputs;
    /// The params used in configure
    type Params;

    /// construct new module from config
    fn new(config: Self::Config) -> Self;
    /// Configure
    fn configure(meta: &mut ConstraintSystem<F>, params: Self::Params) -> Self::Config;
    /// Name
    fn name(&self) -> &'static str;
    /// Run the operation the module represents
    fn run(input: Self::RunInputs) -> Result<Vec<Vec<F>>, Box<dyn std::error::Error>>;
    /// Layout inputs
    fn layout_inputs(
        &self,
        layouter: &mut impl Layouter<F>,
        input: &[ValTensor<F>],
    ) -> Result<Self::InputAssignments, Error>;
    /// Layout
    fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        input: &[ValTensor<F>],
        row_offset: usize,
    ) -> Result<ValTensor<F>, Error>;
    /// Number of instance values the module uses every time it is applied
    fn instance_increment_input(&self) -> Vec<usize>;
    /// Number of rows used by the module
    fn num_rows(input_len: usize) -> usize;
}
