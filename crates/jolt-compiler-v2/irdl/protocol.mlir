irdl.dialect @protocol {
  irdl.operation @params {
    %sym = irdl.any
    %field = irdl.any
    %pcs = irdl.any
    %transcript = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "field" = %field,
      "pcs" = %pcs,
      "transcript" = %transcript
    }
  }
  irdl.operation @boundary {
    %sym = irdl.any
    %roles = irdl.any
    irdl.attributes {"sym_name" = %sym, "roles" = %roles}
  }
}
