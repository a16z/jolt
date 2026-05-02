irdl.dialect @poly {
  irdl.type @domain_type
  irdl.type @oracle
  irdl.type @point
  irdl.operation @domain {
    %sym = irdl.any
    %field = irdl.any
    %log_size = irdl.any
    irdl.attributes {"sym_name" = %sym, "field" = %field, "log_size" = %log_size}
  }
  irdl.operation @point_slice {
    %input = irdl.parametric @poly::@point<>
    %output = irdl.parametric @poly::@point<>
    %sym = irdl.any
    %source = irdl.any
    %offset = irdl.any
    %length = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "source" = %source,
      "offset" = %offset,
      "length" = %length
    }
    irdl.operands(input: %input)
    irdl.results(output: %output)
  }
  irdl.operation @point_concat {
    %input = irdl.parametric @poly::@point<>
    %output = irdl.parametric @poly::@point<>
    %sym = irdl.any
    %layout = irdl.any
    %arity = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "layout" = %layout,
      "arity" = %arity
    }
    irdl.operands(inputs: variadic %input)
    irdl.results(output: %output)
  }
  irdl.operation @lagrange_basis_eval {
    %scalar = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    %domain_start = irdl.any
    %domain_size = irdl.any
    %index = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "domain_start" = %domain_start,
      "domain_size" = %domain_size,
      "index" = %index
    }
    irdl.operands(point: %scalar)
    irdl.results(value: %scalar)
  }
}
