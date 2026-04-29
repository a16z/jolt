irdl.dialect @piop {
  irdl.operation @oracle {
    %sym = irdl.any
    %field = irdl.any
    %domain = irdl.any
    %commit_domain = irdl.any
    %visibility = irdl.any
    %layout = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "field" = %field,
      "domain" = %domain,
      "commit_domain" = %commit_domain,
      "visibility" = %visibility,
      "layout" = %layout
    }
  }
  irdl.operation @oracle_family {
    %sym = irdl.any
    %ordered_oracles = irdl.any
    %visibility = irdl.any
    %count = irdl.any
    %domain = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "ordered_oracles" = %ordered_oracles,
      "visibility" = %visibility,
      "count" = %count,
      "domain" = %domain
    }
  }
}
