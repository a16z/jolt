irdl.dialect @protocol {
  irdl.operation @params {
    %sym = irdl.any
    %field = irdl.any
    %pcs = irdl.any
    %transcript = irdl.any
    %xlen = irdl.any
    %log_t = irdl.any
    %trace_length = irdl.any
    %log_k_bytecode = irdl.any
    %bytecode_k = irdl.any
    %log_k_ram = irdl.any
    %ram_k = irdl.any
    %log_k_chunk = irdl.any
    %k_chunk = irdl.any
    %instruction_log_k = irdl.any
    %register_log_k = irdl.any
    %lookup_table_count = irdl.any
    %instruction_d = irdl.any
    %bytecode_d = irdl.any
    %ram_d = irdl.any
    %num_committed = irdl.any
    %num_r1cs_constraints = irdl.any
    %num_r1cs_inputs = irdl.any
    %num_vars_padded = irdl.any
    irdl.attributes {
      "sym_name" = %sym,
      "field" = %field,
      "pcs" = %pcs,
      "transcript" = %transcript,
      "xlen" = %xlen,
      "log_t" = %log_t,
      "trace_length" = %trace_length,
      "log_k_bytecode" = %log_k_bytecode,
      "bytecode_k" = %bytecode_k,
      "log_k_ram" = %log_k_ram,
      "ram_k" = %ram_k,
      "log_k_chunk" = %log_k_chunk,
      "k_chunk" = %k_chunk,
      "instruction_log_k" = %instruction_log_k,
      "register_log_k" = %register_log_k,
      "lookup_table_count" = %lookup_table_count,
      "instruction_d" = %instruction_d,
      "bytecode_d" = %bytecode_d,
      "ram_d" = %ram_d,
      "num_committed" = %num_committed,
      "num_r1cs_constraints" = %num_r1cs_constraints,
      "num_r1cs_inputs" = %num_r1cs_inputs,
      "num_vars_padded" = %num_vars_padded
    }
  }
  irdl.operation @boundary {
    %sym = irdl.any
    %roles = irdl.any
    irdl.attributes {"sym_name" = %sym, "roles" = %roles}
  }
}
