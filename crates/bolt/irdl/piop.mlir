irdl.dialect @piop {
  irdl.type @stage_type
  irdl.type @sumcheck_claim_type
  irdl.type @sumcheck_batch_type
  irdl.type @sumcheck_result_type
  irdl.type @sumcheck_proof_type
  irdl.type @opening_claim_type
  irdl.type @opening_batch_type

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
  irdl.operation @stage {
    %stage_type = irdl.parametric @piop::@stage_type<>
    %sym = irdl.any
    %name = irdl.any
    %order = irdl.any
    %roles = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "name" = %name,
      "order" = %order,
      "roles" = %roles
    }
    irdl.results(stage: %stage_type)
  }
  irdl.operation @relation {
    %sym = irdl.any
    %kind = irdl.any
    %domain = irdl.any
    %num_rounds = irdl.any
    %degree = irdl.any
    %output_count = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "kind" = %kind,
      "domain" = %domain,
      "num_rounds" = %num_rounds,
      "degree" = %degree,
      "output_count" = %output_count
    }
  }
  irdl.operation @sumcheck_claim {
    %input_claim = irdl.parametric @field::@scalar<>
    %opening_claim = irdl.parametric @piop::@opening_claim_type<>
    %claim_type = irdl.parametric @piop::@sumcheck_claim_type<>
    %sym = irdl.any
    %stage = irdl.any
    %domain = irdl.any
    %num_rounds = irdl.any
    %degree = irdl.any
    %claim = irdl.any
    %relation = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "stage" = %stage,
      "domain" = %domain,
      "num_rounds" = %num_rounds,
      "degree" = %degree,
      "claim" = %claim,
      "relation" = %relation
    }
    irdl.operands(input_claim: %input_claim, inputs: variadic %opening_claim)
    irdl.results(claim: %claim_type)
  }
  irdl.operation @opening_input {
    %point = irdl.parametric @poly::@point<>
    %eval = irdl.parametric @field::@scalar<>
    %claim = irdl.parametric @piop::@opening_claim_type<>
    %sym = irdl.any
    %source_stage = irdl.any
    %source_claim = irdl.any
    %oracle = irdl.any
    %domain = irdl.any
    %point_arity = irdl.any
    %claim_kind = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "source_stage" = %source_stage,
      "source_claim" = %source_claim,
      "oracle" = %oracle,
      "domain" = %domain,
      "point_arity" = %point_arity,
      "claim_kind" = %claim_kind
    }
    irdl.results(point: %point, eval: %eval, claim: %claim)
  }
  irdl.operation @sumcheck_batch {
    %stage_type = irdl.parametric @piop::@stage_type<>
    %claim_type = irdl.parametric @piop::@sumcheck_claim_type<>
    %batch_type = irdl.parametric @piop::@sumcheck_batch_type<>
    %sym = irdl.any
    %stage = irdl.any
    %proof_slot = irdl.any
    %policy = irdl.any
    %count = irdl.any
    %ordered_claims = irdl.any
    %claim_label = irdl.any
    %round_label = irdl.any
    %round_schedule = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "stage" = %stage,
      "proof_slot" = %proof_slot,
      "policy" = %policy,
      "count" = %count,
      "ordered_claims" = %ordered_claims,
      "claim_label" = %claim_label,
      "round_label" = %round_label,
      "round_schedule" = %round_schedule
    }
    irdl.operands(stage: %stage_type, claims: variadic %claim_type)
    irdl.results(batch: %batch_type)
  }
  irdl.operation @sumcheck {
    %state = irdl.parametric @transcript::@state_type<>
    %batch_type = irdl.parametric @piop::@sumcheck_batch_type<>
    %point = irdl.parametric @poly::@point<>
    %result = irdl.parametric @piop::@sumcheck_result_type<>
    %proof = irdl.parametric @piop::@sumcheck_proof_type<>
    %sym = irdl.any
    %stage = irdl.any
    %proof_slot = irdl.any
    %relation = irdl.any
    %policy = irdl.any
    %round_schedule = irdl.any
    %claim_label = irdl.any
    %round_label = irdl.any
    %num_rounds = irdl.any
    %degree = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "stage" = %stage,
      "proof_slot" = %proof_slot,
      "relation" = %relation,
      "policy" = %policy,
      "round_schedule" = %round_schedule,
      "claim_label" = %claim_label,
      "round_label" = %round_label,
      "num_rounds" = %num_rounds,
      "degree" = %degree
    }
    irdl.operands(input: %state, batch: %batch_type)
    irdl.results(output: %state, point: %point, result: %result, proof: %proof)
  }
  irdl.operation @sumcheck_eval {
    %result = irdl.parametric @piop::@sumcheck_result_type<>
    %eval = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    %source = irdl.any
    %name = irdl.any
    %index = irdl.any
    %oracle = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "source" = %source,
      "name" = %name,
      "index" = %index,
      "oracle" = %oracle
    }
    irdl.operands(result: %result)
    irdl.results(eval: %eval)
  }
  irdl.operation @sumcheck_instance_result {
    %input_point = irdl.parametric @poly::@point<>
    %output_point = irdl.parametric @poly::@point<>
    %input_result = irdl.parametric @piop::@sumcheck_result_type<>
    %output_result = irdl.parametric @piop::@sumcheck_result_type<>
    %sym = irdl.any
    %source = irdl.any
    %claim = irdl.any
    %relation = irdl.any
    %index = irdl.any
    %point_arity = irdl.any
    %num_rounds = irdl.any
    %round_offset = irdl.any
    %point_order = irdl.any
    %degree = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "source" = %source,
      "claim" = %claim,
      "relation" = %relation,
      "index" = %index,
      "point_arity" = %point_arity,
      "num_rounds" = %num_rounds,
      "round_offset" = %round_offset,
      "point_order" = %point_order,
      "degree" = %degree
    }
    irdl.operands(input_point: %input_point, input_result: %input_result)
    irdl.results(instance_point: %output_point, instance_result: %output_result)
  }
  irdl.operation @sumcheck_output_value {
    %point = irdl.parametric @poly::@point<>
    %value = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    %kind = irdl.any
    %local_point_segment = irdl.any
    %local_point_length = irdl.any
    %local_point_order = irdl.any
    %opening_point_segment = irdl.any
    %opening_point_length = irdl.any
    %opening_point_order = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "kind" = %kind,
      "local_point_segment" = %local_point_segment,
      "local_point_length" = %local_point_length,
      "local_point_order" = %local_point_order,
      "opening_point_segment" = %opening_point_segment,
      "opening_point_length" = %opening_point_length,
      "opening_point_order" = %opening_point_order
    }
    irdl.operands(local_point: %point, opening_point: %point)
    irdl.results(value: %value)
  }
  irdl.operation @sumcheck_output_claim {
    %value = irdl.parametric @field::@scalar<>
    %sym = irdl.any
    %stage = irdl.any
    %relation = irdl.any
    %count = irdl.any
    %local_values = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "stage" = %stage,
      "relation" = %relation,
      "count" = %count,
      "local_values" = %local_values
    }
    irdl.operands(claim_value: %value, local_values: variadic %value)
  }
  irdl.operation @opening_claim {
    %point = irdl.parametric @poly::@point<>
    %eval = irdl.parametric @field::@scalar<>
    %claim = irdl.parametric @piop::@opening_claim_type<>
    %sym = irdl.any
    %oracle = irdl.any
    %domain = irdl.any
    %point_arity = irdl.any
    %claim_kind = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "oracle" = %oracle,
      "domain" = %domain,
      "point_arity" = %point_arity,
      "claim_kind" = %claim_kind
    }
    irdl.operands(point: %point, eval: %eval)
    irdl.results(claim: %claim)
  }
  irdl.operation @opening_claim_equal {
    %claim = irdl.parametric @piop::@opening_claim_type<>
    %sym = irdl.any
    %mode = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "mode" = %mode
    }
    irdl.operands(left: %claim, right: %claim)
  }
  irdl.operation @opening_batch {
    %claim = irdl.parametric @piop::@opening_claim_type<>
    %batch = irdl.parametric @piop::@opening_batch_type<>
    %sym = irdl.any
    %stage = irdl.any
    %proof_slot = irdl.any
    %policy = irdl.any
    %count = irdl.any
    %ordered_claims = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "stage" = %stage,
      "proof_slot" = %proof_slot,
      "policy" = %policy,
      "count" = %count,
      "ordered_claims" = %ordered_claims
    }
    irdl.operands(claims: variadic %claim)
    irdl.results(batch: %batch)
  }
}
