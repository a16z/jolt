irdl.dialect @transcript {
  irdl.type @state_type
  irdl.operation @scheme {
    %sym = irdl.any
    %hash = irdl.any
    irdl.attributes {"sym_name" = %sym, "hash" = %hash}
  }
  irdl.operation @state {
    %state = irdl.parametric @transcript::@state_type<>
    %sym = irdl.any
    %scheme = irdl.any
    irdl.attributes {"sym_name" = %sym, "scheme" = %scheme}
    irdl.results(state: %state)
  }
  irdl.operation @absorb {
    %state = irdl.parametric @transcript::@state_type<>
    %artifact = irdl.parametric @commit::@artifact<>
    %sym = irdl.any
    %label = irdl.any
    irdl.attributes {"sym_name" = %sym, "label" = %label}
    irdl.operands(input: %state, artifact: %artifact)
    irdl.results(output: %state)
  }
  irdl.operation @absorb_optional {
    %state = irdl.parametric @transcript::@state_type<>
    %artifact = irdl.parametric @commit::@artifact<>
    %sym = irdl.any
    %label = irdl.any
    irdl.attributes {"sym_name" = %sym, "label" = %label}
    irdl.operands(input: %state, artifact: %artifact)
    irdl.results(output: %state)
  }
  irdl.operation @squeeze {
    %state = irdl.parametric @transcript::@state_type<>
    %challenge = irdl.parametric @field::@challenge<>
    %sym = irdl.any
    %label = irdl.any
    %kind = irdl.any
    %count = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "label" = %label,
      "kind" = %kind,
      "count" = %count
    }
    irdl.operands(input: %state)
    irdl.results(output: %state, challenge: %challenge)
  }
}
