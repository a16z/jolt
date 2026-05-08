irdl.dialect @commit {
  irdl.type @artifact
  irdl.operation @publish_batch {
    %artifact_type = irdl.parametric @commit::@artifact<>
    %sym = irdl.any
    %oracle_family = irdl.any
    %label = irdl.any
    irdl.attributes {"sym_name" = %sym, "oracle_family" = %oracle_family, "label" = %label}
    irdl.results(artifact: %artifact_type)
  }
  irdl.operation @publish_optional {
    %artifact_type = irdl.parametric @commit::@artifact<>
    %sym = irdl.any
    %oracle = irdl.any
    %label = irdl.any
    %skip_policy = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "oracle" = %oracle,
      "label" = %label,
      "skip_policy" = %skip_policy
    }
    irdl.results(artifact: %artifact_type)
  }
}
