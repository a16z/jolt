irdl.dialect @pcs {
  irdl.type @scheme_type
  irdl.type @opening_claim_type
  irdl.type @opening_batch_type
  irdl.type @opening_proof_type
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
  irdl.operation @opening_claim {
    %point = irdl.parametric @poly::@point<>
    %eval = irdl.parametric @field::@scalar<>
    %claim = irdl.parametric @pcs::@opening_claim_type<>
    %sym = irdl.any
    %oracle = irdl.any
    %family = irdl.any
    %domain = irdl.any
    %point_arity = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "oracle" = %oracle,
      "family" = %family,
      "domain" = %domain,
      "point_arity" = %point_arity
    }
    irdl.operands(point: %point, eval: %eval)
    irdl.results(claim: %claim)
  }
  irdl.operation @opening_batch {
    %claim = irdl.parametric @pcs::@opening_claim_type<>
    %batch = irdl.parametric @pcs::@opening_batch_type<>
    %sym = irdl.any
    %proof_slot = irdl.any
    %policy = irdl.any
    %count = irdl.any
    %ordered_claims = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "proof_slot" = %proof_slot,
      "policy" = %policy,
      "count" = %count,
      "ordered_claims" = %ordered_claims
    }
    irdl.operands(claims: variadic %claim)
    irdl.results(batch: %batch)
  }
  irdl.operation @batch_open {
    %state = irdl.parametric @transcript::@state_type<>
    %batch = irdl.parametric @pcs::@opening_batch_type<>
    %proof = irdl.parametric @pcs::@opening_proof_type<>
    %sym = irdl.any
    %pcs = irdl.any
    %proof_slot = irdl.any
    %transcript_label = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "pcs" = %pcs,
      "proof_slot" = %proof_slot,
      "transcript_label" = %transcript_label
    }
    irdl.operands(input: %state, batch: %batch)
    irdl.results(output: %state, proof: %proof)
  }
  irdl.operation @batch_verify {
    %state = irdl.parametric @transcript::@state_type<>
    %batch = irdl.parametric @pcs::@opening_batch_type<>
    %proof = irdl.parametric @pcs::@opening_proof_type<>
    %sym = irdl.any
    %pcs = irdl.any
    %proof_slot = irdl.any
    %transcript_label = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "pcs" = %pcs,
      "proof_slot" = %proof_slot,
      "transcript_label" = %transcript_label
    }
    irdl.operands(input: %state, batch: %batch)
    irdl.results(output: %state, proof: %proof)
  }
}
