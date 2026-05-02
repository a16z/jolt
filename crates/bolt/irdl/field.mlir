irdl.dialect @field {
  irdl.type @scalar
  irdl.operation @define {
    %sym = irdl.any
    %modulus_bits = irdl.any
    %role = irdl.any
    irdl.attributes {"sym_name" = %sym, "modulus_bits" = %modulus_bits, "role" = %role}
  }
  irdl.operation @const {
    %scalar = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    %field = irdl.any
    %value = irdl.any
    irdl.attributes {"sym_name" = %sym, "field" = %field, "value" = %value}
    irdl.results(value: %scalar)
  }
  irdl.operation @zero {
    %scalar = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    %field = irdl.any
    irdl.attributes {"sym_name" = %sym, "field" = %field}
    irdl.results(value: %scalar)
  }
  irdl.operation @one {
    %scalar = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    %field = irdl.any
    irdl.attributes {"sym_name" = %sym, "field" = %field}
    irdl.results(value: %scalar)
  }
  irdl.operation @add {
    %scalar = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    irdl.attributes {"sym_name" = %sym}
    irdl.operands(lhs: %scalar, rhs: %scalar)
    irdl.results(value: %scalar)
  }
  irdl.operation @sub {
    %scalar = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    irdl.attributes {"sym_name" = %sym}
    irdl.operands(lhs: %scalar, rhs: %scalar)
    irdl.results(value: %scalar)
  }
  irdl.operation @neg {
    %scalar = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    irdl.attributes {"sym_name" = %sym}
    irdl.operands(input: %scalar)
    irdl.results(value: %scalar)
  }
  irdl.operation @mul {
    %scalar = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    irdl.attributes {"sym_name" = %sym}
    irdl.operands(lhs: %scalar, rhs: %scalar)
    irdl.results(value: %scalar)
  }
  irdl.operation @pow {
    %scalar = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    %exponent = irdl.any
    irdl.attributes {"sym_name" = %sym, "exponent" = %exponent}
    irdl.operands(input: %scalar)
    irdl.results(value: %scalar)
  }
}
