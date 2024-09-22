// [N]
@group(0) @binding(0) var<storage, read_write>
sums: array<u32>;
// [N']
@group(0) @binding(1) var<storage, read_write>
sums_next: array<u32>;

// [N / N']
var<workgroup>
sums_in_group: array<u32, GROUP_SIZE>;

// N / N'
const GROUP_SIZE: u32 = 256u;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    // Specifying the index

    // (0 ~ N)
    let global_index = global_id.x;
    // (0 ~ N')
    let group_index = group_id.x;

    // Specifying the parameters

    let is_invocation_valid = global_index < arrayLength(&sums);
    let is_largest_local_index = local_index + 1u == GROUP_SIZE;
    var sum_original = 0u;
    var sum_exclusive = 0u;

    if is_invocation_valid {
        sum_original = sums[global_index];
        if !is_largest_local_index {
            // Shifting by one for exclusive scan
            sums_in_group[local_index + 1u] = sum_original;
        }
    }

    // Scanning the sums in the group exclusively

    for (var stride = 1u; stride < GROUP_SIZE; stride <<= 1u) {
        workgroupBarrier();
        sum_exclusive = sums_in_group[local_index];
        if local_index >= stride {
            sum_exclusive += sums_in_group[local_index - stride];
        }

        workgroupBarrier();
        sums_in_group[local_index] = sum_exclusive;
    }

    // Specifying the result of sums for the next pass

    if is_largest_local_index {
        let sum_inclusive = sum_exclusive + sum_original;
        sums_next[group_index] = sum_inclusive;
    }

    // Specifying the result of sums

    if is_invocation_valid {
        sums[global_index] = sum_exclusive;
    }
}
