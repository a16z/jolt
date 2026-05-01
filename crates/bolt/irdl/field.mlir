irdl.dialect @field {
  irdl.type @scalar
  irdl.type @challenge
  irdl.operation @define {
    %sym = irdl.any
    %modulus_bits = irdl.any
    %role = irdl.any
    irdl.attributes {"sym_name" = %sym, "modulus_bits" = %modulus_bits, "role" = %role}
  }
  irdl.operation @constant {
    %scalar = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    %field = irdl.any
    %value = irdl.any
    irdl.attributes {"sym_name" = %sym, "field" = %field, "value" = %value}
    irdl.results(value: %scalar)
  }
  irdl.operation @challenge_extract {
    %challenge = irdl.parametric @field::@challenge<>
    %scalar = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    %source = irdl.any
    %index = irdl.any
    irdl.attributes {"sym_name" = %sym, "source" = %source, "index" = %index}
    irdl.operands(challenge: %challenge)
    irdl.results(value: %scalar)
  }
  irdl.operation @expr {
    %scalar = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    %kind = irdl.any
    %formula = irdl.any
    %operands = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "kind" = %kind,
      "formula" = %formula,
      "operands" = %operands
    }
    irdl.operands(inputs: variadic %scalar)
    irdl.results(value: %scalar)
  }
}
