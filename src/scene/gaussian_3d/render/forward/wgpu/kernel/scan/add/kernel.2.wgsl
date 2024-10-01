// [N]
@group(0) @binding(0) var<storage, read_write>
values: array<u32>;

// [N']
@group(0) @binding(1) var<storage, read_write>
values_next: array<u32>;

// N / N'
const GROUP_SIZE: u32 = 256u;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    // Specifying the index

    // (0 ~ N)
    let global_index = global_id.x;
    // (0 ~ N')
    let group_index = group_id.x;
    if global_index >= arrayLength(&values) {
        return;
    }

    // Propagating values backwards

    values[global_index] += values_next[group_index];
}
