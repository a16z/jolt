#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn int_to_string(n: i32) -> String {
    n.to_string()
}

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn string_concat(n: i32) -> String {
    let mut res = String::new();
    for i in 0..n {
        res += &i.to_string();
    }

    res
}
