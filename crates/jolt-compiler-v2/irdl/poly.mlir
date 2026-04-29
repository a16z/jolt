irdl.dialect @poly {
  irdl.type @domain_type
  irdl.type @oracle
  irdl.operation @domain {
    %sym = irdl.any
    %field = irdl.any
    %log_size = irdl.any
    irdl.attributes {"sym_name" = %sym, "field" = %field, "log_size" = %log_size}
  }
}
