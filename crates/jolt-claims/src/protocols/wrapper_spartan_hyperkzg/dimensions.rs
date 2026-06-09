use super::WrapperSpartanHyperKzgFactsError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WrapperRelationDimensions {
    pub variables: usize,
    pub constraints: usize,
    pub public_inputs: usize,
}

impl WrapperRelationDimensions {
    pub fn new(variables: usize, constraints: usize, public_inputs: usize) -> Self {
        Self {
            variables,
            constraints,
            public_inputs,
        }
    }

    pub fn spartan_dimensions(
        self,
    ) -> Result<SpartanRelationDimensions, WrapperSpartanHyperKzgFactsError> {
        SpartanRelationDimensions::from_relation_dimensions(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaddedR1csDimension {
    raw: usize,
    padded: usize,
    log_padded: usize,
}

impl PaddedR1csDimension {
    pub fn new(
        dimension: &'static str,
        raw: usize,
    ) -> Result<Self, WrapperSpartanHyperKzgFactsError> {
        let padded = raw.max(1).checked_next_power_of_two().ok_or(
            WrapperSpartanHyperKzgFactsError::DimensionOverflow {
                dimension,
                value: raw,
            },
        )?;
        Ok(Self {
            raw,
            padded,
            log_padded: padded.trailing_zeros() as usize,
        })
    }

    pub fn raw(self) -> usize {
        self.raw
    }

    pub fn padded(self) -> usize {
        self.padded
    }

    pub fn log_padded(self) -> usize {
        self.log_padded
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicInputSegment {
    start: usize,
    len: usize,
}

impl PublicInputSegment {
    pub fn new(start: usize, len: usize) -> Result<Self, WrapperSpartanHyperKzgFactsError> {
        let _ = start
            .checked_add(len)
            .ok_or(WrapperSpartanHyperKzgFactsError::PublicInputLayoutOverflow { start, len })?;
        Ok(Self { start, len })
    }

    pub fn start(self) -> usize {
        self.start
    }

    pub fn len(self) -> usize {
        self.len
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    pub fn end(self) -> usize {
        self.start + self.len
    }

    pub fn indices(self) -> std::ops::Range<usize> {
        self.start..self.end()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrapperPublicInputLayout {
    wrapper_inputs: PublicInputSegment,
    total: usize,
}

impl WrapperPublicInputLayout {
    pub fn contiguous(public_inputs: usize) -> Result<Self, WrapperSpartanHyperKzgFactsError> {
        let wrapper_inputs = PublicInputSegment::new(0, public_inputs)?;
        Self::new(wrapper_inputs, public_inputs)
    }

    pub fn new(
        wrapper_inputs: PublicInputSegment,
        total: usize,
    ) -> Result<Self, WrapperSpartanHyperKzgFactsError> {
        let end = wrapper_inputs.end();
        if end != total {
            return Err(WrapperSpartanHyperKzgFactsError::PublicInputLayoutMismatch { total, end });
        }
        Ok(Self {
            wrapper_inputs,
            total,
        })
    }

    pub fn wrapper_inputs(self) -> PublicInputSegment {
        self.wrapper_inputs
    }

    pub fn total(self) -> usize {
        self.total
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpartanRelationDimensions {
    variables: PaddedR1csDimension,
    constraints: PaddedR1csDimension,
    public_inputs: WrapperPublicInputLayout,
}

impl SpartanRelationDimensions {
    pub fn from_relation_dimensions(
        dimensions: WrapperRelationDimensions,
    ) -> Result<Self, WrapperSpartanHyperKzgFactsError> {
        Ok(Self {
            variables: PaddedR1csDimension::new("variables", dimensions.variables)?,
            constraints: PaddedR1csDimension::new("constraints", dimensions.constraints)?,
            public_inputs: WrapperPublicInputLayout::contiguous(dimensions.public_inputs)?,
        })
    }

    pub fn variables(self) -> PaddedR1csDimension {
        self.variables
    }

    pub fn constraints(self) -> PaddedR1csDimension {
        self.constraints
    }

    pub fn public_inputs_layout(self) -> WrapperPublicInputLayout {
        self.public_inputs
    }

    pub fn num_vars(self) -> usize {
        self.variables.raw()
    }

    pub fn num_vars_padded(self) -> usize {
        self.variables.padded()
    }

    pub fn num_var_rounds(self) -> usize {
        self.variables.log_padded()
    }

    pub fn num_constraints(self) -> usize {
        self.constraints.raw()
    }

    pub fn num_constraints_padded(self) -> usize {
        self.constraints.padded()
    }

    pub fn num_constraint_rounds(self) -> usize {
        self.constraints.log_padded()
    }

    pub fn num_public_inputs(self) -> usize {
        self.public_inputs.total()
    }
}

impl TryFrom<WrapperRelationDimensions> for SpartanRelationDimensions {
    type Error = WrapperSpartanHyperKzgFactsError;

    fn try_from(dimensions: WrapperRelationDimensions) -> Result<Self, Self::Error> {
        Self::from_relation_dimensions(dimensions)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests assert successful construction")]
mod tests {
    use super::{
        PaddedR1csDimension, PublicInputSegment, SpartanRelationDimensions,
        WrapperPublicInputLayout, WrapperRelationDimensions,
    };
    use crate::protocols::wrapper_spartan_hyperkzg::WrapperSpartanHyperKzgFactsError;

    #[test]
    fn padded_dimensions_track_raw_padded_and_log_values() {
        let dimension = PaddedR1csDimension::new("variables", 17).unwrap();

        assert_eq!(dimension.raw(), 17);
        assert_eq!(dimension.padded(), 32);
        assert_eq!(dimension.log_padded(), 5);
    }

    #[test]
    fn zero_dimension_uses_singleton_domain() {
        let dimension = PaddedR1csDimension::new("constraints", 0).unwrap();

        assert_eq!(dimension.raw(), 0);
        assert_eq!(dimension.padded(), 1);
        assert_eq!(dimension.log_padded(), 0);
    }

    #[test]
    fn public_input_layout_is_contiguous_in_wrapper_input_order() {
        let layout = WrapperPublicInputLayout::contiguous(3).unwrap();

        assert_eq!(layout.total(), 3);
        assert_eq!(
            layout.wrapper_inputs().indices().collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn public_input_layout_rejects_mismatched_total() {
        let segment = PublicInputSegment::new(1, 3).unwrap();

        assert_eq!(
            WrapperPublicInputLayout::new(segment, 3),
            Err(WrapperSpartanHyperKzgFactsError::PublicInputLayoutMismatch { total: 3, end: 4 })
        );
    }

    #[test]
    fn wrapper_dimensions_convert_to_spartan_shape() {
        let wrapper = WrapperRelationDimensions::new(17, 23, 5);
        let spartan = SpartanRelationDimensions::try_from(wrapper).unwrap();

        assert_eq!(spartan.num_vars(), wrapper.variables);
        assert_eq!(spartan.num_vars_padded(), 32);
        assert_eq!(spartan.num_var_rounds(), 5);
        assert_eq!(spartan.num_constraints(), wrapper.constraints);
        assert_eq!(spartan.num_constraints_padded(), 32);
        assert_eq!(spartan.num_constraint_rounds(), 5);
        assert_eq!(spartan.num_public_inputs(), wrapper.public_inputs);
    }
}
