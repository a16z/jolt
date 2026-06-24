use jolt_field::{Field, Fr};

use crate::cuda::Gather8Inputs;

pub(crate) fn materialize_gather8<F: Field, R: AsRef<[Option<u8>]>>(
    table_groups: &[&Vec<Vec<F>>; 8],
    indices: &[R],
) -> Option<Vec<Vec<F>>> {
    let ctx = crate::cuda::shared_ctx()?;
    let num_chunks = indices.len();
    if num_chunks == 0 {
        return None;
    }
    let new_len = indices[0].as_ref().len() / 8;
    let table_len = table_groups[0].first().map_or(0, Vec::len);
    if table_len == 0 || new_len == 0 {
        return None;
    }
    for group in table_groups {
        if group.len() != num_chunks || group.iter().any(|table| table.len() != table_len) {
            return None;
        }
    }

    let mut flat_groups: Vec<Vec<Fr>> = Vec::with_capacity(8);
    for group in table_groups {
        let mut flat = Vec::with_capacity(num_chunks * table_len);
        for table in *group {
            flat.extend_from_slice(crate::cuda::as_fr_slice(table)?);
        }
        flat_groups.push(flat);
    }
    let table_refs: [&[Fr]; 8] = std::array::from_fn(|g| flat_groups[g].as_slice());

    let mut flat_indices: Vec<i16> = Vec::with_capacity(num_chunks * new_len * 8);
    for chunk in indices {
        for entry in chunk.as_ref() {
            flat_indices.push(entry.map_or(-1, i16::from));
        }
    }

    let dense = ctx
        .gather8_materialize(Gather8Inputs {
            table_groups: table_refs,
            indices: &flat_indices,
            num_chunks,
            table_len,
            new_len,
        })
        .ok()?;

    dense
        .into_iter()
        .map(|chunk| {
            chunk
                .into_iter()
                .map(crate::cuda::fr_into::<F>)
                .collect::<Option<Vec<F>>>()
        })
        .collect::<Option<Vec<Vec<F>>>>()
}
