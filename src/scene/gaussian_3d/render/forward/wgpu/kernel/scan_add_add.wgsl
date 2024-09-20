// [N]
@group(0) @binding(0)
var<storage, read_write> sums: array<u32>;
// [N']
@group(0) @binding(1)
var<storage, read_write> sums_next: array<u32>;

// N / N'
const GROUP_SIZE: u32 = 256u;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    // Specifying the index

    // (0 ~ N')
    let group_index = group_id.x;
    // (0 ~ N)
    let index = global_id.x;

    if index >= arrayLength(&sums) {
        return;
    }

    // Adding sums back
    sums[index] += sums_next[group_index];
}
