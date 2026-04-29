irdl.dialect @pcs {
  irdl.type @scheme_type
  irdl.operation @scheme {
    %sym = irdl.any
    %field = irdl.any
    irdl.attributes {"sym_name" = %sym, "field" = %field}
  }
  irdl.operation @commit_batch {
    %artifact = irdl.parametric @commit::@artifact<>
    %sym = irdl.any
    %scheme = irdl.any
    irdl.attributes {"sym_name" = %sym, "scheme" = %scheme}
    irdl.operands(commitment: %artifact)
  }
}
