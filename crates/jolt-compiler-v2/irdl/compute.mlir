irdl.dialect @compute {
  irdl.type @commitment_artifact
  irdl.type @transcript_state
  irdl.type @oracle_buffer
  irdl.type @oracle_family

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
