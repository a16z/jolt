//! Witness dimensions and program facts without execution rows or polynomial tables.

use std::ops::Range;

use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
use jolt_field::Field;
use jolt_program::preprocess::JoltProgramPreprocessing;

use super::JoltVmWitnessConfig;
use crate::{ChunkVisitor, JoltWitnessOracle, ProgramSource, RowSource, Shape, WitnessError};

/// Witness shapes derived from configuration and program preprocessing.
/// This view stores no execution rows or polynomial tables. Data queries return
/// [`WitnessError::UnavailableView`]; shape and program queries remain available.
pub struct JoltVmWitnessMetadata<'a> {
    pub(super) config: &'a JoltVmWitnessConfig,
    pub(super) preprocessing: &'a JoltProgramPreprocessing,
}

impl<'a> JoltVmWitnessMetadata<'a> {
    pub fn new(
        config: &'a JoltVmWitnessConfig,
        preprocessing: &'a JoltProgramPreprocessing,
    ) -> Self {
        Self {
            config,
            preprocessing,
        }
    }
}

impl<F: Field> JoltWitnessOracle<F> for JoltVmWitnessMetadata<'_> {
    fn shape(&self, id: JoltPolynomialId) -> Result<Shape, WitnessError> {
        self.shape(id)
    }

    fn committed_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
        self.committed_polynomial_order()
    }

    fn oracle_table(&self, id: JoltPolynomialId) -> Result<Vec<F>, WitnessError> {
        let _ = self.shape(id)?;
        Err(WitnessError::UnavailableView {
            label: "witness metadata has no polynomial tables",
        })
    }
}

impl ProgramSource for JoltVmWitnessMetadata<'_> {
    fn program_preprocessing(&self) -> &JoltProgramPreprocessing {
        self.preprocessing
    }
}

impl RowSource for JoltVmWitnessMetadata<'_> {
    fn visit_chunks(
        &self,
        _range: Range<usize>,
        _chunk_size: usize,
        _visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError> {
        Err(WitnessError::UnavailableView {
            label: "witness metadata has no trace rows",
        })
    }
}
