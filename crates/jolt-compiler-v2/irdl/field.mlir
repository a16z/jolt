irdl.dialect @field {
  irdl.type @scalar
  irdl.operation @define {
    %sym = irdl.any
    %modulus_bits = irdl.any
    %role = irdl.any
    irdl.attributes {"sym_name" = %sym, "modulus_bits" = %modulus_bits, "role" = %role}
  }
}
