irdl.dialect @compute {
  irdl.type @commitment_artifact
  irdl.type @transcript_state
  irdl.type @oracle_buffer
  irdl.type @oracle_family
  irdl.type @field_value
  irdl.type @point
  irdl.type @sumcheck_claim_type
  irdl.type @sumcheck_batch_type
  irdl.type @sumcheck_result_type
  irdl.type @sumcheck_proof_type
  irdl.type @opening_claim_type
  irdl.type @opening_batch_type
  irdl.type @opening_proof_type

  irdl.operation @params {
    %sym = irdl.any
    %field = irdl.any
    %pcs = irdl.any
    %transcript = irdl.any
    irdl.attributes {"sym_name" = %sym, "field" = %field, "pcs" = %pcs, "transcript" = %transcript}
  }
  irdl.operation @function {
    %sym = irdl.any
    %source = irdl.any
    irdl.attributes {"sym_name" = %sym, "source" = %source}
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
  irdl.operation @kernel {
    %sym = irdl.any
    %relation = irdl.any
    %kind = irdl.any
    %backend = irdl.any
    %abi = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "relation" = %relation,
      "kind" = %kind,
      "backend" = %backend,
      "abi" = %abi
    }
  }
  irdl.operation @oracle_dense_trace {
    %buffer = irdl.parametric @compute::@oracle_buffer<>
    %sym = irdl.any
    %oracle = irdl.any
    %source = irdl.any
    %domain = irdl.any
    %num_vars = irdl.any
    %padding = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "oracle" = %oracle,
      "source" = %source,
      "domain" = %domain,
      "num_vars" = %num_vars,
      "padding" = %padding
    }
    irdl.results(buffer: %buffer)
  }
  irdl.operation @oracle_one_hot_chunk {
    %buffer = irdl.parametric @compute::@oracle_buffer<>
    %sym = irdl.any
    %oracle = irdl.any
    %source = irdl.any
    %domain = irdl.any
    %num_vars = irdl.any
    %trace_num_vars = irdl.any
    %chunk = irdl.any
    %num_chunks = irdl.any
    %chunk_bits = irdl.any
    %padding = irdl.any
    %layout = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "oracle" = %oracle,
      "source" = %source,
      "domain" = %domain,
      "num_vars" = %num_vars,
      "trace_num_vars" = %trace_num_vars,
      "chunk" = %chunk,
      "num_chunks" = %num_chunks,
      "chunk_bits" = %chunk_bits,
      "padding" = %padding,
      "layout" = %layout
    }
    irdl.results(buffer: %buffer)
  }
  irdl.operation @oracle_optional_advice {
    %buffer = irdl.parametric @compute::@oracle_buffer<>
    %sym = irdl.any
    %oracle = irdl.any
    %source = irdl.any
    %domain = irdl.any
    %num_vars = irdl.any
    %skip_policy = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "oracle" = %oracle,
      "source" = %source,
      "domain" = %domain,
      "num_vars" = %num_vars,
      "skip_policy" = %skip_policy
    }
    irdl.results(buffer: %buffer)
  }
  irdl.operation @oracle_ref {
    %buffer = irdl.parametric @compute::@oracle_buffer<>
    %sym = irdl.any
    %oracle = irdl.any
    %domain = irdl.any
    %num_vars = irdl.any
    irdl.attributes {"sym_name" = %sym, "oracle" = %oracle, "domain" = %domain, "num_vars" = %num_vars}
    irdl.results(buffer: %buffer)
  }
  irdl.operation @oracle_family_init {
    %family_type = irdl.parametric @compute::@oracle_family<>
    %sym = irdl.any
    %family = irdl.any
    %count = irdl.any
    irdl.attributes {"sym_name" = %sym, "family" = %family, "count" = %count}
    irdl.results(family: %family_type)
  }
  irdl.operation @oracle_family_append {
    %family_type = irdl.parametric @compute::@oracle_family<>
    %buffer = irdl.parametric @compute::@oracle_buffer<>
    %sym = irdl.any
    %family = irdl.any
    %oracle = irdl.any
    %index = irdl.any
    irdl.attributes {"sym_name" = %sym, "family" = %family, "oracle" = %oracle, "index" = %index}
    irdl.operands(input: %family_type, oracle_buffer: %buffer)
    irdl.results(output: %family_type)
  }
  irdl.operation @pcs_commit_batch {
    %artifact_type = irdl.parametric @compute::@commitment_artifact<>
    %family_type = irdl.parametric @compute::@oracle_family<>
    %sym = irdl.any
    %artifact = irdl.any
    %pcs = irdl.any
    %oracle_family = irdl.any
    %ordered_oracles = irdl.any
    %label = irdl.any
    %domain = irdl.any
    %num_vars = irdl.any
    %count = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "artifact" = %artifact,
      "pcs" = %pcs,
      "oracle_family" = %oracle_family,
      "ordered_oracles" = %ordered_oracles,
      "label" = %label,
      "domain" = %domain,
      "num_vars" = %num_vars,
      "count" = %count
    }
    irdl.operands(oracles: %family_type)
    irdl.results(artifact: %artifact_type)
  }
  irdl.operation @pcs_commit_optional {
    %artifact_type = irdl.parametric @compute::@commitment_artifact<>
    %buffer = irdl.parametric @compute::@oracle_buffer<>
    %sym = irdl.any
    %artifact = irdl.any
    %pcs = irdl.any
    %oracle = irdl.any
    %label = irdl.any
    %domain = irdl.any
    %num_vars = irdl.any
    %skip_policy = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "artifact" = %artifact,
      "pcs" = %pcs,
      "oracle" = %oracle,
      "label" = %label,
      "domain" = %domain,
      "num_vars" = %num_vars,
      "skip_policy" = %skip_policy
    }
    irdl.operands(oracle_buffer: %buffer)
    irdl.results(artifact: %artifact_type)
  }
  irdl.operation @pcs_receive_batch {
    %artifact_type = irdl.parametric @compute::@commitment_artifact<>
    %family_type = irdl.parametric @compute::@oracle_family<>
    %sym = irdl.any
    %artifact = irdl.any
    %pcs = irdl.any
    %oracle_family = irdl.any
    %ordered_oracles = irdl.any
    %label = irdl.any
    %domain = irdl.any
    %num_vars = irdl.any
    %count = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "artifact" = %artifact,
      "pcs" = %pcs,
      "oracle_family" = %oracle_family,
      "ordered_oracles" = %ordered_oracles,
      "label" = %label,
      "domain" = %domain,
      "num_vars" = %num_vars,
      "count" = %count
    }
    irdl.operands(oracles: %family_type)
    irdl.results(artifact: %artifact_type)
  }
  irdl.operation @pcs_receive_optional {
    %artifact_type = irdl.parametric @compute::@commitment_artifact<>
    %buffer = irdl.parametric @compute::@oracle_buffer<>
    %sym = irdl.any
    %artifact = irdl.any
    %pcs = irdl.any
    %oracle = irdl.any
    %label = irdl.any
    %domain = irdl.any
    %num_vars = irdl.any
    %skip_policy = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "artifact" = %artifact,
      "pcs" = %pcs,
      "oracle" = %oracle,
      "label" = %label,
      "domain" = %domain,
      "num_vars" = %num_vars,
      "skip_policy" = %skip_policy
    }
    irdl.operands(oracle_buffer: %buffer)
    irdl.results(artifact: %artifact_type)
  }
  irdl.operation @transcript_init {
    %state = irdl.parametric @compute::@transcript_state<>
    %sym = irdl.any
    %scheme = irdl.any
    irdl.attributes {"sym_name" = %sym, "scheme" = %scheme}
    irdl.results(state: %state)
  }
  irdl.operation @transcript_absorb {
    %state = irdl.parametric @compute::@transcript_state<>
    %artifact = irdl.parametric @compute::@commitment_artifact<>
    %sym = irdl.any
    %label = irdl.any
    %optional = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "label" = %label,
      "optional" = %optional
    }
    irdl.operands(input: %state, artifact: %artifact)
    irdl.results(output: %state)
  }
  irdl.operation @transcript_absorb_bytes {
    %state = irdl.parametric @compute::@transcript_state<>
    %sym = irdl.any
    %label = irdl.any
    %payload = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "label" = %label,
      "payload" = %payload
    }
    irdl.operands(input: %state)
    irdl.results(output: %state)
  }
  irdl.operation @transcript_squeeze {
    %state = irdl.parametric @compute::@transcript_state<>
    %challenge = irdl.any
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
  irdl.operation @opening_input {
    %point = irdl.parametric @compute::@point<>
    %eval = irdl.parametric @compute::@field_value<>
    %claim = irdl.parametric @compute::@opening_claim_type<>
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
  irdl.operation @point_slice {
    %input = irdl.parametric @compute::@point<>
    %output = irdl.parametric @compute::@point<>
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
  irdl.operation @point_zero {
    %output = irdl.parametric @compute::@point<>
    %sym = irdl.any
    %field = irdl.any
    %arity = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "field" = %field,
      "arity" = %arity
    }
    irdl.results(output: %output)
  }
  irdl.operation @point_concat {
    %input = irdl.parametric @compute::@point<>
    %output = irdl.parametric @compute::@point<>
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
  irdl.operation @field_const {
    %value_type = irdl.parametric @compute::@field_value<>
    %sym = irdl.any
    %field = irdl.any
    %value = irdl.any
    irdl.attributes {"sym_name" = %sym, "field" = %field, "value" = %value}
    irdl.results(value: %value_type)
  }
  irdl.operation @field_zero {
    %value_type = irdl.parametric @compute::@field_value<>
    %sym = irdl.any
    %field = irdl.any
    irdl.attributes {"sym_name" = %sym, "field" = %field}
    irdl.results(value: %value_type)
  }
  irdl.operation @field_one {
    %value_type = irdl.parametric @compute::@field_value<>
    %sym = irdl.any
    %field = irdl.any
    irdl.attributes {"sym_name" = %sym, "field" = %field}
    irdl.results(value: %value_type)
  }
  irdl.operation @field_add {
    %value_type = irdl.parametric @compute::@field_value<>
    %sym = irdl.any
    irdl.attributes {"sym_name" = %sym}
    irdl.operands(lhs: %value_type, rhs: %value_type)
    irdl.results(value: %value_type)
  }
  irdl.operation @field_sub {
    %value_type = irdl.parametric @compute::@field_value<>
    %sym = irdl.any
    irdl.attributes {"sym_name" = %sym}
    irdl.operands(lhs: %value_type, rhs: %value_type)
    irdl.results(value: %value_type)
  }
  irdl.operation @field_neg {
    %value_type = irdl.parametric @compute::@field_value<>
    %sym = irdl.any
    irdl.attributes {"sym_name" = %sym}
    irdl.operands(input: %value_type)
    irdl.results(value: %value_type)
  }
  irdl.operation @field_mul {
    %value_type = irdl.parametric @compute::@field_value<>
    %sym = irdl.any
    irdl.attributes {"sym_name" = %sym}
    irdl.operands(lhs: %value_type, rhs: %value_type)
    irdl.results(value: %value_type)
  }
  irdl.operation @field_pow {
    %value_type = irdl.parametric @compute::@field_value<>
    %sym = irdl.any
    %exponent = irdl.any
    irdl.attributes {"sym_name" = %sym, "exponent" = %exponent}
    irdl.operands(input: %value_type)
    irdl.results(value: %value_type)
  }
  irdl.operation @poly_lagrange_basis_eval {
    %value_type = irdl.parametric @compute::@field_value<>
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
    irdl.operands(point: %value_type)
    irdl.results(value: %value_type)
  }
  irdl.operation @sumcheck_claim {
    %input_claim = irdl.parametric @compute::@field_value<>
    %opening_claim = irdl.parametric @compute::@opening_claim_type<>
    %claim_type = irdl.parametric @compute::@sumcheck_claim_type<>
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
  irdl.operation @sumcheck_kernel_claim {
    %input_claim = irdl.parametric @compute::@field_value<>
    %opening_claim = irdl.parametric @compute::@opening_claim_type<>
    %claim_type = irdl.parametric @compute::@sumcheck_claim_type<>
    %sym = irdl.any
    %stage = irdl.any
    %domain = irdl.any
    %num_rounds = irdl.any
    %degree = irdl.any
    %claim = irdl.any
    %kernel = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "stage" = %stage,
      "domain" = %domain,
      "num_rounds" = %num_rounds,
      "degree" = %degree,
      "claim" = %claim,
      "kernel" = %kernel
    }
    irdl.operands(input_claim: %input_claim, inputs: variadic %opening_claim)
    irdl.results(claim: %claim_type)
  }
  irdl.operation @sumcheck_verify_claim {
    %input_claim = irdl.parametric @compute::@field_value<>
    %opening_claim = irdl.parametric @compute::@opening_claim_type<>
    %claim_type = irdl.parametric @compute::@sumcheck_claim_type<>
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
  irdl.operation @sumcheck_batch {
    %claim_type = irdl.parametric @compute::@sumcheck_claim_type<>
    %batch_type = irdl.parametric @compute::@sumcheck_batch_type<>
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
    irdl.operands(claims: variadic %claim_type)
    irdl.results(batch: %batch_type)
  }
  irdl.operation @sumcheck_driver {
    %state = irdl.parametric @compute::@transcript_state<>
    %batch_type = irdl.parametric @compute::@sumcheck_batch_type<>
    %point = irdl.parametric @compute::@point<>
    %result = irdl.parametric @compute::@sumcheck_result_type<>
    %proof = irdl.parametric @compute::@sumcheck_proof_type<>
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
  irdl.operation @sumcheck_kernel_driver {
    %state = irdl.parametric @compute::@transcript_state<>
    %batch_type = irdl.parametric @compute::@sumcheck_batch_type<>
    %point = irdl.parametric @compute::@point<>
    %result = irdl.parametric @compute::@sumcheck_result_type<>
    %proof = irdl.parametric @compute::@sumcheck_proof_type<>
    %sym = irdl.any
    %stage = irdl.any
    %proof_slot = irdl.any
    %kernel = irdl.any
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
      "kernel" = %kernel,
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
  irdl.operation @sumcheck_verify {
    %state = irdl.parametric @compute::@transcript_state<>
    %batch_type = irdl.parametric @compute::@sumcheck_batch_type<>
    %point = irdl.parametric @compute::@point<>
    %result = irdl.parametric @compute::@sumcheck_result_type<>
    %proof = irdl.parametric @compute::@sumcheck_proof_type<>
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
    %result = irdl.parametric @compute::@sumcheck_result_type<>
    %eval = irdl.parametric @compute::@field_value<>
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
    %input_point = irdl.parametric @compute::@point<>
    %output_point = irdl.parametric @compute::@point<>
    %input_result = irdl.parametric @compute::@sumcheck_result_type<>
    %output_result = irdl.parametric @compute::@sumcheck_result_type<>
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
  irdl.operation @structured_polynomial_eval {
    %point = irdl.parametric @compute::@point<>
    %value = irdl.parametric @compute::@field_value<>
    %sym = irdl.any
    %polynomial = irdl.any
    %x_point_segment = irdl.any
    %x_point_length = irdl.any
    %x_point_order = irdl.any
    %y_point_segment = irdl.any
    %y_point_length = irdl.any
    %y_point_order = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "polynomial" = %polynomial,
      "x_point_segment" = %x_point_segment,
      "x_point_length" = %x_point_length,
      "x_point_order" = %x_point_order,
      "y_point_segment" = %y_point_segment,
      "y_point_length" = %y_point_length,
      "y_point_order" = %y_point_order
    }
    irdl.operands(x_point: %point, y_point: %point)
    irdl.results(value: %value)
  }
  irdl.operation @sumcheck_output_eval_family {
    %value = irdl.parametric @compute::@field_value<>
    %sym = irdl.any
    %power_stride = irdl.any
    %value_term_offsets = irdl.any
    %shared_term_offsets = irdl.any
    %item_term_offsets = irdl.any
    %evals = irdl.any
    %shared_terms = irdl.any
    %item_terms = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "power_stride" = %power_stride,
      "value_term_offsets" = %value_term_offsets,
      "shared_term_offsets" = %shared_term_offsets,
      "item_term_offsets" = %item_term_offsets,
      "evals" = %evals,
      "shared_terms" = %shared_terms,
      "item_terms" = %item_terms
    }
    irdl.operands(gamma: %value, inputs: variadic %value)
    irdl.results(value: %value)
  }
  irdl.operation @sumcheck_output_product_family {
    %value = irdl.parametric @compute::@field_value<>
    %sym = irdl.any
    %term_gamma_power_offsets = irdl.any
    %term_eval_counts = irdl.any
    %term_factor_counts = irdl.any
    %evals = irdl.any
    %factors = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "term_gamma_power_offsets" = %term_gamma_power_offsets,
      "term_eval_counts" = %term_eval_counts,
      "term_factor_counts" = %term_factor_counts,
      "evals" = %evals,
      "factors" = %factors
    }
    irdl.operands(gamma: %value, inputs: variadic %value)
    irdl.results(value: %value)
  }
  irdl.operation @sumcheck_output_function_family {
    %value = irdl.parametric @compute::@field_value<>
    %sym = irdl.any
    %term_gamma_power_offsets = irdl.any
    %term_functions = irdl.any
    %term_factor_counts = irdl.any
    %evals = irdl.any
    %factors = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "term_gamma_power_offsets" = %term_gamma_power_offsets,
      "term_functions" = %term_functions,
      "term_factor_counts" = %term_factor_counts,
      "evals" = %evals,
      "factors" = %factors
    }
    irdl.operands(gamma: %value, inputs: variadic %value)
    irdl.results(value: %value)
  }
  irdl.operation @sumcheck_output_claim {
    %value = irdl.parametric @compute::@field_value<>
    %sym = irdl.any
    %stage = irdl.any
    %relation = irdl.any
    %count = irdl.any
    %polynomial_evals = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "stage" = %stage,
      "relation" = %relation,
      "count" = %count,
      "polynomial_evals" = %polynomial_evals
    }
    irdl.operands(claim_value: %value, polynomial_evals: variadic %value)
  }
  irdl.operation @opening_claim {
    %point = irdl.parametric @compute::@point<>
    %eval = irdl.parametric @compute::@field_value<>
    %claim = irdl.parametric @compute::@opening_claim_type<>
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
    %claim = irdl.parametric @compute::@opening_claim_type<>
    %sym = irdl.any
    %mode = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "mode" = %mode
    }
    irdl.operands(left: %claim, right: %claim)
  }
  irdl.operation @opening_batch {
    %claim = irdl.parametric @compute::@opening_claim_type<>
    %batch = irdl.parametric @compute::@opening_batch_type<>
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
  irdl.operation @pcs_opening_claim {
    %point = irdl.parametric @compute::@point<>
    %eval = irdl.parametric @compute::@field_value<>
    %claim = irdl.parametric @compute::@opening_claim_type<>
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
  irdl.operation @pcs_opening_batch {
    %claim = irdl.parametric @compute::@opening_claim_type<>
    %batch = irdl.parametric @compute::@opening_batch_type<>
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
  irdl.operation @pcs_batch_open {
    %state = irdl.parametric @compute::@transcript_state<>
    %batch = irdl.parametric @compute::@opening_batch_type<>
    %proof = irdl.parametric @compute::@opening_proof_type<>
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
  irdl.operation @pcs_batch_verify {
    %state = irdl.parametric @compute::@transcript_state<>
    %batch = irdl.parametric @compute::@opening_batch_type<>
    %proof = irdl.parametric @compute::@opening_proof_type<>
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
  irdl.operation @generate_oracle {
    %sym = irdl.any
    %oracle = irdl.any
    %source = irdl.any
    %generation = irdl.any
    irdl.attributes {"sym_name" = %sym, "oracle" = %oracle, "source" = %source, "generation" = %generation}
  }
  irdl.operation @generate_oracle_family {
    %sym = irdl.any
    %family = irdl.any
    %source = irdl.any
    %generation = irdl.any
    irdl.attributes {"sym_name" = %sym, "family" = %family, "source" = %source, "generation" = %generation}
  }
}
