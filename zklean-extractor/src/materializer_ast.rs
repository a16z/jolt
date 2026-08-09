use jolt_lookup_tables::MaterializerBackend;

/// A Boolean expression extracted from a lookup materializer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoolExpr {
    Input(usize),
    And(Box<Self>, Box<Self>),
}

impl BoolExpr {
    fn evaluate(&self, point: &[bool]) -> bool {
        match self {
            Self::Input(index) => point[*index],
            Self::And(left, right) => left.evaluate(point) && right.evaluate(point),
        }
    }

    fn format_for_lean(&self, num_inputs: usize) -> Result<String, String> {
        match self {
            Self::Input(index) => {
                if *index >= num_inputs {
                    return Err(format!(
                        "materializer input {index} is outside expression arity {num_inputs}"
                    ));
                }
                Ok(format!("(.input {index})"))
            }
            Self::And(left, right) => Ok(format!(
                "(.conj {} {})",
                left.format_for_lean(num_inputs)?,
                right.format_for_lean(num_inputs)?
            )),
        }
    }
}

/// A natural number expression extracted from a lookup materializer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NatExpr {
    OfBitsBe(Vec<BoolExpr>),
}

impl NatExpr {
    pub fn evaluate(&self, point: &[bool]) -> u64 {
        match self {
            Self::OfBitsBe(bits) => bits.iter().fold(0, |value, bit| {
                (value << 1) | u64::from(bit.evaluate(point))
            }),
        }
    }

    pub fn format_for_lean(&self, num_inputs: usize) -> Result<String, String> {
        match self {
            Self::OfBitsBe(bits) => {
                let bits = bits
                    .iter()
                    .map(|bit| bit.format_for_lean(num_inputs))
                    .map(|bit| bit.map(|bit| format!("    {bit},")))
                    .collect::<Result<Vec<_>, _>>()?
                    .join("\n");
                Ok(format!("(.ofBitsBE [\n{bits}\n  ])"))
            }
        }
    }
}

/// Symbolic backend for the shared Rust materializer semantics.
pub struct MaterializerAst {
    num_inputs: usize,
}

impl MaterializerAst {
    pub fn new(num_inputs: usize) -> Self {
        Self { num_inputs }
    }
}

impl MaterializerBackend for MaterializerAst {
    type Bit = BoolExpr;
    type Output = NatExpr;

    fn input_bit(&mut self, index: usize) -> Self::Bit {
        assert!(
            index < self.num_inputs,
            "materializer input is out of range"
        );
        BoolExpr::Input(index)
    }

    fn and(&mut self, left: Self::Bit, right: Self::Bit) -> Self::Bit {
        BoolExpr::And(Box::new(left), Box::new(right))
    }

    fn bits_be(&mut self, bits: Vec<Self::Bit>) -> Self::Output {
        NatExpr::OfBitsBe(bits)
    }
}

#[cfg(test)]
mod tests {
    use jolt_lookup_tables::{tables::and::AndTable, LookupMaterializer, LookupTable};

    use super::*;

    #[test]
    fn extracts_and_from_shared_materializer() {
        let expression = AndTable::<2>.materialize(&mut MaterializerAst::new(4));
        assert_eq!(
            expression.format_for_lean(4).unwrap(),
            "(.ofBitsBE [\n    (.conj (.input 0) (.input 1)),\n    \
             (.conj (.input 2) (.input 3)),\n  ])"
        );
    }

    #[test]
    fn extracted_and_interpreter_matches_concrete_backend() {
        const XLEN: usize = 4;
        let expression = AndTable::<XLEN>.materialize(&mut MaterializerAst::new(2 * XLEN));

        for index in 0..(1u128 << (2 * XLEN)) {
            let point = (0..2 * XLEN)
                .map(|i| (index >> (2 * XLEN - 1 - i)) & 1 == 1)
                .collect::<Vec<_>>();
            assert_eq!(
                expression.evaluate(&point),
                AndTable::<XLEN>.materialize_entry(index)
            );
        }
    }
}
